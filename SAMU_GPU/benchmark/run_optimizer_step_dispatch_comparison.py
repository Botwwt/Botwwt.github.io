"""Gate-4 optimizer-step A/B for strengthened RG-LRU and SAMU dispatches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import RecurrentBlock, StudyConfig, set_seed


CASES = (
    ("state2048_short", 4, 2048,
     dict(width=2048, rnn_width=2048, depth=1),
     "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_chunk32"),
    ("state2560_medium", 1, 8192,
     dict(width=2560, rnn_width=2560, depth=1),
     "fused_grouped_prefix32_hierarchical_chunk16", 256,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
    ("400m_block_long", 1, 32768,
     dict(width=1536, rnn_width=2048, depth=12),
     "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
)


def timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples):
    median = statistics.median(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "mad_ms": statistics.median(abs(value - median) for value in samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=75619)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "protocol": {
            "scope": "complete recurrent block forward, input/parameter backward and optimizer update",
            "optimizer": "torch.optim.AdamW(fused=True), states initialized before timing",
            "dtype": "BF16 autocast, FP32 parameters and recurrent carry",
            "order": "AB/BA",
            "compile_and_optimizer_state_initialization_excluded": True,
        },
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "cases": [],
    }
    for case_index, (case, batch, length, dimensions, rg_backend, rg_block,
                     samu_backend) in enumerate(CASES):
        print(case, flush=True)
        config = StudyConfig(vocab_size=32000, **dimensions)
        modules = {}
        for architecture in ("rglru", "samu"):
            set_seed(args.seed + case_index * 11 + (architecture == "samu"))
            modules[architecture] = RecurrentBlock(config, architecture).cuda().train()
        modules["rglru"].mixer.set_scan_backend(rg_backend)
        modules["rglru"].mixer.set_fused_training_block_size(rg_block)
        modules["samu"].mixer.set_scan_backend(samu_backend)
        modules["samu"].mixer.set_tiled_training_block_size(256)
        modules["samu"].mixer.set_controller_projection_dtype("fp32_fused_coords")
        with torch.no_grad():
            modules["samu"].mixer.phase_amplitude.fill_(0.7)
            modules["samu"].mixer.radial_amplitude.fill_(0.5)
        optimizers = {
            name: torch.optim.AdamW(module.parameters(), lr=1.0e-5, fused=True)
            for name, module in modules.items()
        }
        x = torch.randn(batch, length, config.width, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        cotangent = torch.randn_like(x)

        def step(name):
            optimizer, module = optimizers[name], modules[name]
            optimizer.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(x)[0]
            (output.float() * cotangent.float()).sum().backward()
            optimizer.step()

        # Compile kernels and allocate every AdamW state before measurement.
        for name in modules:
            step(name)
        raw = {name: [] for name in modules}
        names = tuple(modules)
        for round_index in range(args.rounds):
            order = names if (round_index + case_index) % 2 == 0 else tuple(reversed(names))
            for name in order:
                for _ in range(args.warmup):
                    step(name)
                samples = [timed(lambda n=name: step(n))
                           for _ in range(args.repetitions)]
                raw[name].extend(samples)
                print(name, statistics.median(samples), flush=True)
        rows = []
        for name, module in modules.items():
            optimizers[name].zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            step(name)
            torch.cuda.synchronize()
            rows.append({
                "architecture": name,
                "backend": rg_backend if name == "rglru" else samu_backend,
                "controller": "fp32_fused_coords" if name == "samu" else None,
                "optimizer_step": summarize(raw[name]),
                "baseline_allocated_bytes": allocated,
                "baseline_reserved_bytes": reserved,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - allocated,
                "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - reserved,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            })
        result["cases"].append({
            "case": case, "batch": batch, "length": length,
            "config": dimensions, "rows": rows,
        })
        atomic_json(args.output, result)
        del modules, optimizers, x, cotangent
        torch.cuda.empty_cache()
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
