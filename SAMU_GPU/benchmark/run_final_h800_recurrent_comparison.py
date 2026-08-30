"""Final selected-backend mixer width scaling and full-block H800 comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import (
    RecurrentBlock,
    RGLRUMixer,
    SAMUMixer,
    StudyConfig,
    set_seed,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
    }


def measure_pair(modules, x, gradient, rounds, repetitions, scope):
    def forward(name):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = modules[name](x)
            if isinstance(output, tuple):
                output = output[0]
        return output

    def forward_backward(name):
        modules[name].zero_grad(set_to_none=True)
        output = forward(name)
        (output.float() * gradient.float()).sum().backward()

    for name in modules:
        forward_backward(name)
    torch.cuda.synchronize()
    raw = {name: {"forward": [], "forward_backward": [], "rounds": []}
           for name in modules}
    names = list(modules)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for order_index, name in enumerate(order):
            for _ in range(2): forward(name)
            fwd = [timed(lambda n=name: forward(n)) for _ in range(repetitions)]
            for _ in range(2): forward_backward(name)
            fb = [timed(lambda n=name: forward_backward(n)) for _ in range(repetitions)]
            raw[name]["forward"].extend(fwd)
            raw[name]["forward_backward"].extend(fb)
            raw[name]["rounds"].append({
                "round": round_index, "order_index": order_index,
                "forward_ms": fwd, "forward_backward_ms": fb,
            })
            print(scope, name, statistics.median(fwd), statistics.median(fb), flush=True)
    rows = []
    for name in modules:
        modules[name].zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        forward_backward(name); torch.cuda.synchronize()
        rows.append({
            "architecture": name,
            "forward": summarize(raw[name]["forward"]),
            "forward_backward": summarize(raw[name]["forward_backward"]),
            "rounds": raw[name]["rounds"],
            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - allocated,
            "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - reserved,
        })
    return rows


def copy_shared_block(reference, target):
    source, destination = reference.state_dict(), target.state_dict()
    copied = 0
    for name, value in destination.items():
        candidate = source.get(name)
        if candidate is not None and candidate.shape == value.shape and not name.startswith("mixer."):
            destination[name] = candidate.detach().clone()
            copied += value.numel()
    target.load_state_dict(destination)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", action="append", type=int,
                        default=[256, 512, 1024, 2048, 2560, 4096])
    parser.add_argument("--width-length", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74001)
    args = parser.parse_args()
    widths = tuple(dict.fromkeys(args.width))
    set_seed(args.seed)
    width_rows = []
    for width in widths:
        if width % 16 or width % 2:
            raise ValueError(width)
        x = torch.randn(1, args.width_length, width, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn_like(x)
        rglru = RGLRUMixer(width, 16).cuda().train()
        rglru.set_scan_backend("fused_chunk16")
        rglru.set_fused_training_block_size(64)
        samu = SAMUMixer(width).cuda().train()
        samu.set_scan_backend("chunk16")
        samu.set_controller_projection_dtype("fp32")
        with torch.no_grad():
            samu.phase_amplitude.fill_(0.7); samu.radial_amplitude.fill_(0.5)
        rows = measure_pair(
            {"rglru": rglru, "samu": samu}, x, gradient,
            args.rounds, args.repetitions, f"mixer_D{width}",
        )
        width_rows.append({
            "width": width, "batch": 1, "length": args.width_length,
            "dynamic_control_values_per_token": {"rglru": 2 * width, "samu": 2},
            "rows": rows,
        })
        del x, gradient, rglru, samu
        torch.cuda.empty_cache()

    block_rows = []
    scales = (
        ("400m", dict(width=1536, rnn_width=2048, depth=12)),
        ("1.3b", dict(width=2048, rnn_width=2560, depth=24)),
    )
    for scale, dimensions in scales:
        config = StudyConfig(vocab_size=32000, **dimensions)
        set_seed(args.seed + len(block_rows) * 17)
        rglru = RecurrentBlock(config, "rglru").cuda().train()
        set_seed(args.seed + len(block_rows) * 17 + 1)
        samu = RecurrentBlock(config, "samu").cuda().train()
        copied = copy_shared_block(rglru, samu)
        rglru.mixer.set_scan_backend("fused_chunk16")
        rglru.mixer.set_fused_training_block_size(64)
        samu.mixer.set_scan_backend("chunk16")
        samu.mixer.set_controller_projection_dtype("fp32")
        with torch.no_grad():
            samu.mixer.phase_amplitude.fill_(0.7); samu.mixer.radial_amplitude.fill_(0.5)
        x = torch.randn(4, 2048, config.width, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn_like(x)
        rows = measure_pair(
            {"rglru": rglru, "samu": samu}, x, gradient,
            args.rounds, args.repetitions, f"block_{scale}",
        )
        block_rows.append({
            "scale": scale, "batch": 4, "length": 2048,
            "config": dimensions, "shared_parameters_copied": copied,
            "rows": rows,
        })
        del x, gradient, rglru, samu
        torch.cuda.empty_cache()

    atomic_json(args.output, {
        "schema_version": 1,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "protocol": {
            "dtype": "BF16 autocast, FP32 recurrent accumulation",
            "selected_backends": {"rglru": "fused_chunk16:block64",
                                  "samu": "chunk16:FP32-controller"},
            "order": "AB/BA",
            "compile_excluded": True,
            "width_scaling": "B=1,L=2048,state-byte matched",
            "full_block": "RMSNorm, parallel branches, Conv1D, recurrence, join, output projection, channel RMSNorm and gated MLP",
        },
        "width_scaling": width_rows,
        "full_block": block_rows,
    })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
