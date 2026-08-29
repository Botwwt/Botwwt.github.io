"""Full-forward companion to the Griffin Figure 8 scan experiment.

This random-weight proxy asks whether a recurrence-only scan advantage remains
visible after the shared embedding, MLP and normalization shell is included.
It is a forward/prefill measurement, not a training step: there is no backward,
optimizer, all-reduce or ZeRO timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from run_equal_kernel_benchmarks import stabilize_gpu
from run_griffin_candidate_inference import (
    RGLRU_COMMIT,
    build_model,
    correctness,
    make_shell,
    prefill,
)
from run_griffin_section5 import percentile


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def timed_prefill(model, token_ids: torch.Tensor, repeats: int) -> list[float]:
    # The warm-up is outside every recorded sample so JIT compilation, allocator
    # growth and first-use library initialization cannot enter the result.
    output, caches = prefill(model, token_ids)
    del output, caches
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, caches = prefill(model, token_ids)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output, caches
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    audit = correctness(args.rglru_source)
    if not audit["passed"]:
        raise RuntimeError(f"correctness failed: {audit}")
    stabilization = stabilize_gpu()
    kinds = ("samu_grouped16_direct", "rglru_official16")
    print("[1/3] building the exact shared 1B-scale shell", flush=True)
    shared_shell = make_shell(2048, 24, 3, 32000, args.seed)
    models = {
        kind: build_model(
            kind, 2048, 24, 3, 32000, args.rglru_source, args.seed,
            shared_shell=shared_shell,
        )
        for kind in kinds
    }

    rows = []
    lengths = [2048, 4096, 8192, 16384]
    for length in lengths:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + length)
        token_ids = torch.randint(
            0, 32000, (8, length), device="cuda", generator=generator
        )
        print(f"[2/3] full forward B=8 L={length}", flush=True)
        for order_name, order in (("AB", kinds), ("BA", tuple(reversed(kinds)))):
            stabilize_gpu(seconds=0.5)
            for kind in order:
                print(f"  order={order_name} model={kind}", flush=True)
                samples = timed_prefill(models[kind], token_ids, args.repeats)
                median = statistics.median(samples)
                rows.append({
                    "model": kind,
                    "measurement_order": order_name,
                    "batch": 8,
                    "sequence_length": length,
                    "median_ms": median,
                    "p10_ms": percentile(samples, 0.10),
                    "p95_ms": percentile(samples, 0.95),
                    "raw_samples_ms": samples,
                    "tokens_per_second": 8 * length * 1000.0 / median,
                })
        del token_ids
        torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "title": "Griffin Figure 8(b)-style full-forward systems proxy",
        "scope": "random-weight full forward/prefill; no backward or model-quality claim",
        "paper_mapping": {
            "matched_axis": "B=8 and sequence lengths 2K, 4K, 8K and 16K",
            "shared_model": "the exact same embedding, 24-layer MLP/norm shell and final norm tensors",
            "important_difference": "H800 instead of TPU-v3; SAMU versus RG-LRU-16 instead of Hawk scan implementations",
            "not_measured": ["backward", "optimizer", "training step", "all-reduce", "ZeRO"],
            "repeats_per_order": args.repeats,
        },
        "environment": {
            "hostname": platform.node(),
            "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "rglru_commit": RGLRU_COMMIT,
        },
        "gpu_stabilization": stabilization,
        "correctness": audit,
        "models": {
            kind: {
                "parameter_count": models[kind].parameter_count,
                "state_bytes_per_sequence": models[kind].state_bytes_per_sequence,
            }
            for kind in kinds
        },
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(f"[3/3] wrote {args.output}", flush=True)
    del models, shared_shell
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
