"""Focused SAMU Triton vs official-source PyTorch RG-LRU benchmark.

The comparison keeps model width and recurrent-state bytes equal:
SAMU M complex FP32 modes == RG-LRU 2M real FP32 channels.  It intentionally
compares against the unmodified RGLRU layer from the pinned RecurrentGemma
repository, not against a hypothetical custom RG-LRU kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from kernels import load_official_rglru, make_samu_parameters
from triton_samu import (
    pack_samu_parameters,
    samu_packed_reference,
    samu_triton_auto,
    samu_triton_chunked,
    samu_triton_decode,
    samu_triton_serial,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"
PROTOCOL = "same_model_width+same_fp32_recurrent_state_bytes"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=15).strip()
    except Exception:
        return None


def source_digest() -> str:
    path = Path(__file__).with_name("triton_samu.py")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    import triton
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_and_board": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "source_commits": {"samu_triton": source_digest(), "recurrentgemma": RGLRU_COMMIT},
    }


def row_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{config['model']}-{config['backend']}-{config['workload']}-B{config['batch']}-L{config['length']}-M{config['modes']}-{digest}"


def measure(operation: Callable[[], Any], *, warmup: int, samples: int,
            inner_iterations: int = 1) -> tuple[dict[str, Any], float]:
    torch.cuda.empty_cache()
    start_setup = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    compile_setup_seconds = time.perf_counter() - start_setup
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _inner in range(inner_iterations):
            operation()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) / inner_iterations)
    result = {
        "raw_samples_ms": durations,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "p10_ms": percentile(durations, .10),
        "p90_ms": percentile(durations, .90),
        "p95_ms": percentile(durations, .95),
        "std_ms": statistics.pstdev(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "inner_iterations": inner_iterations,
        "samples": samples,
        "timing": "CUDA events; synchronized; first-call compile/setup excluded",
    }
    return result, compile_setup_seconds


def base_config(model: str, backend: str, workload: str, batch: int, length: int,
                modes: int = 64, d_model: int | None = None, **extra) -> dict[str, Any]:
    config = {
        "model": model,
        "backend": backend,
        "workload": workload,
        "dtype": "bf16",
        "batch": batch,
        "length": length,
        "modes": modes,
        "d_model": d_model or 2 * modes,
        "matching_protocol": PROTOCOL,
        "chunk_size": None,
    }
    config.update(extra)
    return config


def configs() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    # Primary equal-state length sweep.
    for length in (128, 512, 2048, 8192, 32768, 65536):
        values.append(base_config("samu", "triton_auto", "forward", 1, length))
        values.append(base_config("rglru", "official_pytorch_source", "forward", 1, length))
    # Expose the crossover that motivates the auto policy.
    for length in (128, 512, 2048, 8192, 32768, 65536):
        values.append(base_config("samu", "triton_serial", "forward", 1, length))
    for length in (512, 2048, 8192, 32768, 65536):
        for chunk in (8, 16, 32):
            values.append(base_config("samu", f"triton_chunk_c{chunk}", "forward", 1, length, chunk_size=chunk))
    # Batch scaling at the medium prefill point.
    for batch in (2, 4, 8, 16):
        values.append(base_config("samu", "triton_auto", "forward", batch, 512))
        values.append(base_config("rglru", "official_pytorch_source", "forward", batch, 512))
    # Equal state-byte / width scaling.
    for modes in (128, 256):
        values.append(base_config("samu", "triton_auto", "forward", 1, 512, modes=modes))
        values.append(base_config("rglru", "official_pytorch_source", "forward", 1, 512, modes=modes))
    # One-token recurrent update, including projection and control.
    for batch in (1, 4, 16, 64):
        values.append(base_config("samu", "triton_fused_decode", "decode", batch, 1))
        values.append(base_config("rglru", "official_pytorch_source", "decode", batch, 1))
    # Stable de-duplication while preserving presentation order.
    seen, result = set(), []
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def correctness() -> dict[str, Any]:
    torch.manual_seed(1729)
    p = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    packed = pack_samu_parameters(p)
    u = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    reference = samu_packed_reference(u, p, packed)
    serial = samu_triton_serial(u, p, packed)
    chunk = samu_triton_chunked(u, p, 16, packed)
    state = (torch.randn(2, 64, device="cuda"), torch.randn(2, 64, device="cuda"))
    token = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    decode = samu_triton_decode(token, state, p, packed)
    decode_reference = samu_packed_reference(token[:, None], p, packed, initial=state)[:, 0]
    decode_stacked = torch.stack(decode, dim=-1).to(torch.bfloat16)
    return {
        "reference_policy": "one packed BF16 projection; FP32 bounded control and complex recurrence",
        "serial_max_abs": float((serial - reference).abs().max()),
        "serial_mean_abs": float((serial - reference).abs().float().mean()),
        "chunk16_max_abs": float((chunk - reference).abs().max()),
        "chunk16_mean_abs": float((chunk - reference).abs().float().mean()),
        "decode_max_abs": float((decode_stacked - decode_reference).abs().max()),
        "decode_mean_abs": float((decode_stacked - decode_reference).abs().float().mean()),
    }


def make_samu_case(config: dict[str, Any]):
    torch.manual_seed(1729)
    batch, length, modes, d_model = (config[k] for k in ("batch", "length", "modes", "d_model"))
    p = make_samu_parameters(d_model, modes, "cuda", torch.bfloat16)
    packed = pack_samu_parameters(p)
    parameter_count = sum(value.numel() for value in p.__dict__.values())
    if config["workload"] == "decode":
        token = torch.randn(batch, d_model, device="cuda", dtype=torch.bfloat16)
        state = (torch.randn(batch, modes, device="cuda"), torch.randn(batch, modes, device="cuda"))
        operation = lambda: samu_triton_decode(token, state, p, packed)
    else:
        u = torch.randn(batch, length, d_model, device="cuda", dtype=torch.bfloat16)
        backend = config["backend"]
        if backend == "triton_auto":
            operation = lambda: samu_triton_auto(u, p, packed)
        elif backend == "triton_serial":
            operation = lambda: samu_triton_serial(u, p, packed)
        elif backend.startswith("triton_chunk_c"):
            operation = lambda: samu_triton_chunked(u, p, int(config["chunk_size"]), packed)
        else:
            raise ValueError(backend)
    return operation, {
        "parameter_count": parameter_count,
        "source_commit": source_digest(),
        "implementation_class": "custom_triton_inference_kernel",
        "real_state_scalars_per_batch": 2 * modes,
        "state_bytes_per_batch": 2 * modes * 4,
        "precision_policy": "BF16 input/packed projection/output; FP32 recurrent state and accumulation",
        "dispatch_policy": "serial L<=256; C16 256<L<=512; C32 L>512" if config["backend"] == "triton_auto" else None,
    }


def make_rglru_case(config: dict[str, Any], source_root: Path):
    width, batch, length = config["d_model"], config["batch"], config["length"]
    if width != 2 * config["modes"]:
        raise ValueError("equal-state protocol requires width == 2*modes")
    RGLRU = load_official_rglru(source_root)
    torch.manual_seed(1729)
    model = RGLRU(width=width, num_heads=2, device="cuda", dtype=torch.bfloat16).cuda().eval()
    if config["workload"] == "decode":
        token = torch.randn(batch, 1, width, device="cuda", dtype=torch.bfloat16)
        position = torch.ones(batch, 1, device="cuda", dtype=torch.long)
        cache = torch.randn(batch, width, device="cuda", dtype=torch.float32)
        operation = lambda: model(token, position, cache=cache, return_cache=True)
    else:
        u = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(length, device="cuda").unsqueeze(0).expand(batch, -1)
        operation = lambda: model(u, positions, cache=None, return_cache=True)[0]
    return operation, {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_commit": RGLRU_COMMIT,
        "implementation_class": "official_recurrentgemma_pytorch_source_unmodified",
        "real_state_scalars_per_batch": width,
        "state_bytes_per_batch": width * 4,
        "precision_policy": "official BF16 layer with FP32 rnn_scan accumulator/cache",
        "fairness_note": "Official public source execution; this is not a custom optimized RG-LRU kernel.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["id", "status", "model", "backend", "workload", "dtype", "batch", "length",
            "d_model", "modes", "chunk_size", "median_ms", "p10_ms", "p95_ms",
            "tokens_per_second", "compile_setup_seconds", "state_bytes_per_batch",
            "parameter_count", "implementation_class", "source_commit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../benchmark_results_samu_rg"))
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", choices=("all", "samu", "rglru"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    env = environment()
    atomic_json(output / "environment.json", env)
    correctness_result = correctness()
    atomic_json(output / "correctness.json", correctness_result)
    print("correctness", json.dumps(correctness_result), flush=True)
    selected = [c for c in configs() if args.only == "all" or c["model"] == args.only]
    for index, config in enumerate(selected, 1):
        identifier = row_id(config)
        raw_path = raw_dir / f"{identifier}.json"
        if args.resume and raw_path.exists():
            print(f"[{index}/{len(selected)}] cached {identifier}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] run {identifier}", flush=True)
        try:
            if config["model"] == "samu":
                operation, metadata = make_samu_case(config)
            else:
                operation, metadata = make_rglru_case(config, args.rglru_source)
            is_decode = config["workload"] == "decode"
            if config["model"] == "rglru" and config["length"] >= 32768:
                samples = 7
            else:
                samples = 30 if config["model"] == "samu" else 15
            result, setup_seconds = measure(operation, warmup=5, samples=samples,
                                            inner_iterations=100 if is_decode else 1)
            token_count = config["batch"] if is_decode else config["batch"] * config["length"]
            row = {
                "id": identifier,
                "status": "measured",
                **config,
                **metadata,
                **result,
                "compile_setup_seconds": setup_seconds,
                "tokens_per_second": token_count * 1000.0 / result["median_ms"],
            }
        except Exception as error:
            row = {"id": identifier, "status": "failed", **config,
                   "error": f"{type(error).__name__}: {error}"}
        atomic_json(raw_path, row)
        print(json.dumps({k: row.get(k) for k in ("model", "backend", "workload", "batch", "length", "modes", "status", "median_ms", "error")}), flush=True)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_dir.glob("*.json"))]
    rows.sort(key=lambda r: (r.get("workload", ""), r.get("batch", 0), r.get("length", 0), r.get("modes", 0), r.get("model", ""), r.get("backend", "")))
    summary = {
        "schema_version": 1,
        "title": "SAMU fused Triton vs official-source RG-LRU",
        "comparison_scope": "inference forward/prefill and one-token recurrent update",
        "fairness": "Equal d_model and equal FP32 recurrent-state bytes. RG-LRU is the pinned official PyTorch source, not a custom kernel; implementation maturity is part of the measured result.",
        "correctness": correctness_result,
        "environment": env,
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", rows)
    atomic_json(output / "run_manifest.json", {
        "command": " ".join(sys.argv),
        "config_count": len(selected),
        "measured": sum(row.get("status") == "measured" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "comparison": "SAMU only vs official RecurrentGemma RGLRU",
    })


if __name__ == "__main__":
    main()
