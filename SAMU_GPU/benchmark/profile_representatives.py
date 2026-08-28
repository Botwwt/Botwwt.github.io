"""Collect Torch-profiler kernel breakdowns for representative measured cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile

from run_benchmarks import base_config, make_mamba3_case, make_rglru_case, make_samu_case, synchronize_result


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def profile_case(name, operation, config):
    for _ in range(3):
        synchronize_result(operation())
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        value = operation()
        synchronize_result(value)
        torch.cuda.synchronize()
    events = []
    for event in prof.key_averages():
        cuda_us = float(getattr(event, "self_cuda_time_total", 0.0) or 0.0)
        if cuda_us <= 0:
            continue
        events.append({
            "name": event.key,
            "count": int(event.count),
            "self_cuda_time_us": cuda_us,
            "cuda_time_us": float(getattr(event, "cuda_time_total", 0.0) or 0.0),
            "cpu_time_us": float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
            "flops_reported": int(getattr(event, "flops", 0) or 0),
        })
    events.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    total = sum(row["self_cuda_time_us"] for row in events)
    for row in events:
        row["self_cuda_share"] = row["self_cuda_time_us"] / total if total else None
    return {
        "name": name,
        "config": config,
        "profiler": "torch.profiler",
        "scope": "one operation after three warmup operations",
        "self_cuda_total_us": total,
        "kernel_event_count": sum(row["count"] for row in events),
        "events": events[:30],
        "unavailable_metrics": [
            "profiler_measured_dram_bytes", "profiler_measured_l2_bytes", "occupancy",
            "registers_per_thread", "tensor_core_utilization", "sfu_utilization", "sm_utilization"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mamba-source", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    args = parser.parse_args()
    protocol = "same_width+state_bytes+parameter_approx"
    configs = [
        ("samu_tree_prefill", base_config("samu", "tree_materialized_factorized", "forward", "bf16", 1, 2048, 64, 128, protocol)),
        ("samu_chunk16_prefill", base_config("samu", "chunked_compressed", "forward", "bf16", 1, 2048, 64, 128, protocol, chunk_size=16)),
        ("rglru_official_prefill", base_config("rglru", "official_pytorch_serial", "forward", "bf16", 1, 2048, 64, 128, protocol, num_heads=2)),
        ("mamba3_official_prefill", base_config("mamba3", "official_siso_triton", "forward", "bf16", 1, 2048, 64, 128, "same_width+best_native", d_state=64, chunk_size=64)),
        ("samu_decode_total", base_config("samu", "decode_direct", "decode", "bf16", 1, 1, 64, 128, protocol)),
        ("samu_decode_recurrence_only", base_config("samu", "decode_precomputed_factorized", "decode", "bf16", 1, 1, 64, 128, protocol)),
    ]
    rows = []
    for name, config in configs:
        print("profiling", name, flush=True)
        if config["model"] == "samu":
            operation, _ = make_samu_case(config, "cuda")
        elif config["model"] == "rglru":
            operation, _ = make_rglru_case(config, "cuda", args.rglru_source)
        else:
            operation, _ = make_mamba3_case(config, "cuda", args.mamba_source)
        rows.append(profile_case(name, operation, config))
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warning": "Torch profiler event times are measured; unavailable hardware counters remain null/N/A.",
        "profiles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
