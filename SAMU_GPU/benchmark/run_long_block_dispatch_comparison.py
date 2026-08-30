"""Gate-4 long-sequence recurrent-block A/B for selected mixer dispatches."""

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
    ("state1024_long", 1, 32768,
     dict(width=1024, rnn_width=1024, depth=1),
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


def copy_shared(reference, target):
    source, destination = reference.state_dict(), target.state_dict()
    copied = 0
    for name, value in destination.items():
        candidate = source.get(name)
        if candidate is not None and candidate.shape == value.shape and not name.startswith("mixer."):
            destination[name] = candidate.detach().clone()
            copied += value.numel()
    target.load_state_dict(destination)
    return copied


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
    parser.add_argument("--seed", type=int, default=75601)
    parser.add_argument(
        "--case", action="append", choices=[case[0] for case in CASES],
        help="run only the named case; repeat to select more than one",
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "protocol": {
            "scope": "complete recurrent block including input gradient",
            "components": "norm, x/y projections, depthwise conv, mixer, join/output, norm and gated MLP",
            "dtype": "BF16 autocast, FP32 recurrent carry",
            "order": "AB/BA",
            "compile_excluded": True,
        },
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "cases": [],
    }
    selected_cases = tuple(
        entry for entry in CASES
        if args.case is None or entry[0] in args.case
    )
    for case_index, (case, batch, length, dimensions, rg_backend, rg_block,
                     samu_backend) in enumerate(selected_cases):
        print(case, flush=True)
        config = StudyConfig(vocab_size=32000, **dimensions)
        set_seed(args.seed + case_index)
        rglru = RecurrentBlock(config, "rglru").cuda().train()
        set_seed(args.seed + case_index + 101)
        samu = RecurrentBlock(config, "samu").cuda().train()
        copied = copy_shared(rglru, samu)
        rglru.mixer.set_scan_backend(rg_backend)
        rglru.mixer.set_fused_training_block_size(rg_block)
        samu.mixer.set_scan_backend(samu_backend)
        samu.mixer.set_tiled_training_block_size(256)
        samu.mixer.set_controller_projection_dtype("fp32_fused_coords")
        with torch.no_grad():
            samu.mixer.phase_amplitude.fill_(0.7)
            samu.mixer.radial_amplitude.fill_(0.5)
        modules = {"rglru": rglru, "samu": samu}
        x = torch.randn(batch, length, config.width, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        cotangent = torch.randn_like(x)

        def forward(name):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return modules[name](x)[0]

        def backward(name):
            modules[name].zero_grad(set_to_none=True)
            x.grad = None
            output = forward(name)
            return timed(lambda o=output: (o.float() * cotangent.float()).sum().backward())

        def forward_backward(name):
            modules[name].zero_grad(set_to_none=True)
            x.grad = None
            output = forward(name)
            (output.float() * cotangent.float()).sum().backward()

        for name in modules:
            forward_backward(name)
        raw = {name: {"forward": [], "backward": [], "forward_backward": []}
               for name in modules}
        for round_index in range(args.rounds):
            order = list(modules)
            if (round_index + case_index) % 2:
                order.reverse()
            for name in order:
                for _ in range(args.warmup):
                    forward(name)
                raw[name]["forward"].extend(
                    timed(lambda n=name: forward(n)) for _ in range(args.repetitions)
                )
                for index in range(args.warmup + args.repetitions):
                    value = backward(name)
                    if index >= args.warmup:
                        raw[name]["backward"].append(value)
                for _ in range(args.warmup):
                    forward_backward(name)
                raw[name]["forward_backward"].extend(
                    timed(lambda n=name: forward_backward(n))
                    for _ in range(args.repetitions)
                )
                print(name, statistics.median(raw[name]["forward"][-args.repetitions:]),
                      statistics.median(raw[name]["backward"][-args.repetitions:]),
                      statistics.median(raw[name]["forward_backward"][-args.repetitions:]),
                      flush=True)
        rows = []
        for name, module in modules.items():
            module.zero_grad(set_to_none=True); x.grad = None
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            forward_backward(name); torch.cuda.synchronize()
            rows.append({
                "architecture": name,
                "backend": rg_backend if name == "rglru" else samu_backend,
                "controller": "fp32_fused_coords" if name == "samu" else None,
                "forward": summarize(raw[name]["forward"]),
                "backward": summarize(raw[name]["backward"]),
                "forward_backward": summarize(raw[name]["forward_backward"]),
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
            "config": dimensions, "shared_parameters_copied": copied,
            "rows": rows,
        })
        atomic_json(args.output, result)
        del rglru, samu, modules, x, cotangent
        torch.cuda.empty_cache()
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
