"""Equal-maturity SAMU / RG-LRU Triton benchmarks plus official baselines.

Track A compares custom inference kernels with equal width, parameter count, and
FP32 recurrent-state bytes.  Track B reports pinned official RecurrentGemma and
state-spaces Mamba-3 implementations without pretending their state geometry or
parameter counts are matched.
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

from kernels import load_official_mamba3, load_official_rglru, make_samu_parameters
from triton_rglru import (
    pack_rglru,
    rglru_triton_auto,
    rglru_triton_chunked,
    rglru_triton_decode,
    rglru_triton_serial,
)
from triton_samu import (
    pack_samu_parameters,
    samu_packed_reference,
    samu_triton_auto,
    samu_triton_chunked,
    samu_triton_decode,
    samu_triton_serial,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"
MAMBA_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
EQUAL_PROTOCOL = "equal_d_model+parameters_within_4+equal_fp32_state_bytes"
MAMBA_PROTOCOL = "same_d_model+official_best_native_not_state_or_parameter_matched"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    import triton
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    benchmark_dir = Path(__file__).parent
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
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "registers_per_sm": getattr(props, "regs_per_multiprocessor", None),
        "shared_memory_per_sm": getattr(props, "shared_memory_per_multiprocessor", None),
        "driver_and_board": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "ncu": command_output(["bash", "-lc", "command -v ncu || true"]) or "unavailable",
        "nsys": command_output(["bash", "-lc", "command -v nsys || true"]) or "unavailable",
        "source_commits": {
            "samu_triton": digest(benchmark_dir / "triton_samu.py"),
            "rglru_triton": digest(benchmark_dir / "triton_rglru.py"),
            "recurrentgemma": RGLRU_COMMIT,
            "mamba": MAMBA_COMMIT,
        },
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def measure(operation: Callable[[], Any], *, warmup: int, samples: int,
            inner: int = 1) -> tuple[dict[str, Any], float]:
    torch.cuda.empty_cache()
    compile_start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _inner in range(inner):
            operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / inner)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, .10),
        "p90_ms": percentile(values, .90),
        "p95_ms": percentile(values, .95),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values),
        "raw_samples_ms": values,
        "samples": samples,
        "inner_iterations": inner,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "timing": "CUDA events; synchronized; first-call compile/setup excluded",
    }, compile_seconds


def base_config(track: str, model: str, backend: str, workload: str,
                batch: int, length: int, **extra) -> dict[str, Any]:
    value = {
        "track": track,
        "model": model,
        "backend": backend,
        "workload": workload,
        "dtype": "bf16",
        "batch": batch,
        "length": length,
        "d_model": 128,
        "modes": 64,
        "chunk_size": None,
        "matching_protocol": EQUAL_PROTOCOL if track == "A_equal_triton" else MAMBA_PROTOCOL,
    }
    value.update(extra)
    return value


def configs() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for length in (128, 512, 2048, 8192, 32768, 65536):
        result.append(base_config("A_equal_triton", "samu", "triton_auto", "prefill", 1, length))
        result.append(base_config("A_equal_triton", "rglru", "triton_auto", "prefill", 1, length))
        result.append(base_config("B_official", "mamba3", "official_siso_triton", "prefill", 1, length))
        result.append(base_config("B_official", "rglru", "official_pytorch_source", "prefill", 1, length,
                                  matching_protocol=EQUAL_PROTOCOL))
    for length in (2048, 65536):
        for chunk in (8, 16, 32):
            result.append(base_config("A_equal_triton", "samu", f"triton_chunk_c{chunk}", "prefill", 1, length, chunk_size=chunk))
            result.append(base_config("A_equal_triton", "rglru", f"triton_chunk_c{chunk}", "prefill", 1, length, chunk_size=chunk))
    for batch in (4, 16, 64):
        result.append(base_config("A_equal_triton", "samu", "triton_auto", "prefill", batch, 512))
        result.append(base_config("A_equal_triton", "rglru", "triton_auto", "prefill", batch, 512))
    for batch in (1, 4, 16, 64):
        result.append(base_config("A_equal_triton", "samu", "triton_fused_decode", "decode", batch, 1))
        result.append(base_config("A_equal_triton", "rglru", "triton_fused_decode", "decode", batch, 1))
        result.append(base_config("B_official", "rglru", "official_pytorch_source", "decode", batch, 1,
                                  matching_protocol=EQUAL_PROTOCOL))
        result.append(base_config("B_official", "mamba3", "official_cute_step", "decode", batch, 1))
    return result


def row_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{config['model']}-{config['backend']}-{config['workload']}-B{config['batch']}-L{config['length']}-{suffix}"


def logical_metrics(config: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    """Algorithmic lower-bound traffic and transcendental counts.

    These are source-derived operation counts, not hardware performance-counter
    measurements.  Weight/cache reuse and allocator traffic are deliberately
    excluded.
    """
    b, l, d, m = config["batch"], config["length"], config["d_model"], config["modes"]
    if config["model"] == "mamba3":
        return None, None
    if config["workload"] == "decode":
        if config["model"] == "samu":
            bytes_ = b * (2 * d * 2 + 2 * d * m * 2 + 4 * d * 4 + 4 * m * 4)
            return bytes_, {"exp_or_sigmoid_per_token": m + 3, "sqrt_per_token": 0,
                            "trig_sfu_per_token": 0, "phase_polynomial": "sin/cos d, bounded Taylor degree 7/6"}
        bytes_ = b * (d * 2 + 2 * (d * d // 2) * 2 + d * 4 + d * 2 + d * 4)
        return bytes_, {"exp_or_sigmoid_per_token": 4 * d, "sqrt_per_token": d,
                        "trig_sfu_per_token": 0}
    if config["model"] == "samu":
        projected = 2 * m + 2
        passes = 1 if config["backend"] == "triton_serial" or (config["backend"] == "triton_auto" and l <= 256) else 2
        dynamic = b * l * (d * 2 + projected * 2 * (1 + passes) + 2 * m * 2)
        weights = d * math.ceil(projected / 16) * 16 * 2
        chunks = math.ceil(l / (config.get("chunk_size") or (16 if l <= 512 else 32))) if passes == 2 else 0
        metadata = b * chunks * 48 * m if passes == 2 else 0
        return dynamic + weights + metadata, {
            "exp_or_sigmoid_per_token": passes * (m + 3), "sqrt_per_token": 0,
            "trig_sfu_per_token": 0, "transition_evaluations_per_token": passes,
        }
    if config["backend"] == "official_pytorch_source":
        passes = 1
    else:
        passes = 1 if config["backend"] == "triton_serial" else 2
    dynamic = b * l * (d * 2 + (2 * d) * 2 * (1 + passes) + passes * d * 2 + d * 2)
    weights = d * d * 2
    chunk = config.get("chunk_size") or (8 if l <= 256 else 16 if l <= 1024 else 32)
    chunks = math.ceil(l / chunk) if passes == 2 else 0
    metadata = b * chunks * 24 * d if passes == 2 else 0
    return dynamic + weights + metadata, {
        "exp_or_sigmoid_per_token": passes * 4 * d, "sqrt_per_token": passes * d,
        "trig_sfu_per_token": 0, "transition_evaluations_per_token": passes,
    }


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return 0


def make_case(config: dict[str, Any], rglru_root: Path, mamba_root: Path):
    torch.manual_seed(1729)
    b, l, d, m = config["batch"], config["length"], config["d_model"], config["modes"]
    if config["model"] == "samu":
        params = make_samu_parameters(d, m, "cuda", torch.bfloat16)
        packed = pack_samu_parameters(params)
        parameter_count = sum(value.numel() for value in params.__dict__.values())
        if config["workload"] == "decode":
            token = torch.randn(b, d, device="cuda", dtype=torch.bfloat16)
            state = (torch.randn(b, m, device="cuda"), torch.randn(b, m, device="cuda"))
            operation = lambda: samu_triton_decode(token, state, params, packed)
            launches = 1
        else:
            x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
            if config["backend"] == "triton_auto":
                operation = lambda: samu_triton_auto(x, params, packed, return_cache=True)
                launches = 2 if l <= 256 else 4
            else:
                chunk = int(config["chunk_size"])
                operation = lambda: samu_triton_chunked(x, params, chunk, packed, return_cache=True)
                launches = 4
        return operation, {
            "parameter_count": parameter_count,
            "state_bytes_per_batch": 2 * m * 4,
            "implementation_class": "custom_triton_inference",
            "source_commit": digest(Path(__file__).with_name("triton_samu.py")),
            "precision_policy": "BF16 projection/output; FP32 recurrent state/cache",
            "expected_cuda_launches": launches,
        }
    if config["model"] == "rglru":
        RGLRU = load_official_rglru(rglru_root)
        model = RGLRU(width=d, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        if config["backend"] == "official_pytorch_source":
            if config["workload"] == "decode":
                x = torch.randn(b, 1, d, device="cuda", dtype=torch.bfloat16)
                positions = torch.ones(b, 1, device="cuda", dtype=torch.long)
                cache = torch.randn(b, d, device="cuda")
                operation = lambda: model(x, positions, cache=cache, return_cache=True)
            else:
                x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
                positions = torch.arange(l, device="cuda").unsqueeze(0).expand(b, -1)
                operation = lambda: model(x, positions, cache=None, return_cache=True)
            return operation, {
                "parameter_count": parameter_count, "state_bytes_per_batch": d * 4,
                "implementation_class": "official_recurrentgemma_pytorch_source_unmodified",
                "source_commit": RGLRU_COMMIT,
                "precision_policy": "official BF16 layer; FP32 rnn_scan accumulator/cache",
                "expected_cuda_launches": None,
            }
        packed = pack_rglru(model)
        if config["workload"] == "decode":
            x = torch.randn(b, 1, d, device="cuda", dtype=torch.bfloat16)
            positions = torch.ones(b, 1, device="cuda", dtype=torch.long)
            cache = torch.randn(b, d, device="cuda")
            operation = lambda: rglru_triton_decode(x, positions, packed, cache)
            launches = 1
        else:
            x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(l, device="cuda").unsqueeze(0).expand(b, -1)
            if config["backend"] == "triton_auto":
                operation = lambda: rglru_triton_auto(x, positions, packed)
            elif config["backend"] == "triton_serial":
                operation = lambda: rglru_triton_serial(x, positions, packed)
            else:
                operation = lambda: rglru_triton_chunked(x, positions, packed, int(config["chunk_size"]))
            launches = 4 if config["backend"] != "triton_serial" else 2
        return operation, {
            "parameter_count": parameter_count, "state_bytes_per_batch": d * 4,
            "implementation_class": "custom_triton_official_equations",
            "source_commit": digest(Path(__file__).with_name("triton_rglru.py")),
            "official_equation_source_commit": RGLRU_COMMIT,
            "precision_policy": "official eager-BF16 pointwise rounding; FP32 recurrent state/cache",
            "expected_cuda_launches": launches,
        }
    Mamba3 = load_official_mamba3(mamba_root)
    model = Mamba3(d_model=d, d_state=m, headdim=64, chunk_size=64,
                   is_mimo=False, dtype=torch.bfloat16, device="cuda").eval()
    if config["workload"] == "decode":
        raise RuntimeError("official Mamba-3 CuTeDSL step kernel is unavailable on this RTX 3090 environment")
    x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
    cache_bytes = None
    try:
        cache_bytes = tensor_bytes(model.allocate_inference_cache(1, 1, device="cuda", dtype=torch.bfloat16))
    except Exception:
        pass
    return lambda: model(x), {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_bytes_per_batch": cache_bytes,
        "implementation_class": "official_state_spaces_mamba3_siso_triton",
        "source_commit": MAMBA_COMMIT,
        "precision_policy": "official Mamba-3 BF16 SISO Triton path",
        "expected_cuda_launches": None,
        "fairness_note": "Same d_model only; native Mamba-3 state and parameter geometry are not matched to SAMU/RG-LRU.",
    }


def correctness(rglru_root: Path) -> dict[str, Any]:
    torch.manual_seed(1729)
    RGLRU = load_official_rglru(rglru_root)
    model = RGLRU(width=128, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    packed_rg = pack_rglru(model)
    x = torch.randn(2, 129, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(129, device="cuda").repeat(2, 1) + 1
    positions[0, 0], positions[0, 5], positions[1, 77] = 3, 0, 0
    cache = torch.randn(2, 128, device="cuda")
    official, official_cache = model(x, positions, cache=cache, return_cache=True)
    serial, serial_cache = rglru_triton_serial(x, positions, packed_rg, cache)
    chunk, chunk_cache = rglru_triton_chunked(x, positions, packed_rg, 16, cache)
    decode_x, decode_pos = x[:, :1], torch.tensor([[1], [0]], device="cuda")
    official_decode, official_decode_cache = model(decode_x, decode_pos, cache=cache, return_cache=True)
    triton_decode, triton_decode_cache = rglru_triton_decode(decode_x, decode_pos, packed_rg, cache)

    samu_params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    packed_samu = pack_samu_parameters(samu_params)
    samu_x = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    samu_reference, samu_reference_cache = samu_packed_reference(
        samu_x, samu_params, packed_samu, return_cache=True
    )
    samu_serial, samu_serial_cache = samu_triton_serial(samu_x, samu_params, packed_samu, return_cache=True)
    samu_chunk, samu_chunk_cache = samu_triton_chunked(samu_x, samu_params, 16, packed_samu, return_cache=True)

    bound = packed_samu.phase_scale
    sample_d = torch.linspace(-bound, bound, 10001, device="cuda")
    d2, d4 = sample_d.square(), sample_d.square().square()
    d6 = d4 * d2
    cos_poly = 1 - .5 * d2 + d4 / 24 - d6 / 720
    sin_poly = sample_d * (1 - d2 / 6 + d4 / 120 - d6 / 5040)
    return {
        "rglru_reference": "official recurrentgemma.torch.layers.RGLRU at pinned commit",
        "rglru_reset_cases": ["nonzero first position with nonzero h0", "mid-sequence reset t=5", "mid-sequence reset t=77", "decode reset/non-reset"],
        "rglru_serial_max_abs": float((serial - official).abs().max()),
        "rglru_serial_mean_abs": float((serial - official).abs().float().mean()),
        "rglru_serial_cache_max_abs": float((serial_cache - official_cache).abs().max()),
        "rglru_chunk16_max_abs": float((chunk - official).abs().max()),
        "rglru_chunk16_mean_abs": float((chunk - official).abs().float().mean()),
        "rglru_chunk16_cache_max_abs": float((chunk_cache - official_cache).abs().max()),
        "rglru_decode_max_abs": float((triton_decode - official_decode).abs().max()),
        "rglru_decode_cache_max_abs": float((triton_decode_cache - official_decode_cache).abs().max()),
        "samu_serial_max_abs": float((samu_serial - samu_reference).abs().max()),
        "samu_chunk16_max_abs": float((samu_chunk - samu_reference).abs().max()),
        "samu_serial_cache_max_abs": float((torch.stack(samu_serial_cache, -1) - torch.stack(samu_reference_cache, -1)).abs().max()),
        "samu_chunk16_cache_max_abs": float((torch.stack(samu_chunk_cache, -1) - torch.stack(samu_reference_cache, -1)).abs().max()),
        "samu_phase_delta_bound_rad": bound,
        "samu_sin_polynomial_max_abs": float((sin_poly - torch.sin(sample_d)).abs().max()),
        "samu_cos_polynomial_max_abs": float((cos_poly - torch.cos(sample_d)).abs().max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["id", "status", "track", "model", "backend", "workload", "batch", "length",
            "d_model", "modes", "chunk_size", "median_ms", "p10_ms", "p95_ms",
            "tokens_per_second", "logical_bytes_lower_bound", "logical_effective_gbps",
            "expected_cuda_launches", "state_bytes_per_batch", "parameter_count",
            "implementation_class", "source_commit", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../benchmark_results_equal_kernel"))
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--mamba-source", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-model", choices=("samu", "rglru", "mamba3"))
    args = parser.parse_args()
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    env = environment()
    atomic_json(output / "environment.json", env)
    audit = correctness(args.rglru_source)
    atomic_json(output / "correctness.json", audit)
    print("correctness", json.dumps(audit), flush=True)
    selected = [config for config in configs()
                if args.only_model is None or config["model"] == args.only_model]
    for index, config in enumerate(selected, 1):
        identifier = row_id(config)
        path = raw / f"{identifier}.json"
        if args.resume and path.exists():
            print(f"[{index}/{len(selected)}] cached {identifier}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] run {identifier}", flush=True)
        logical_bytes, sfu = logical_metrics(config)
        try:
            operation, metadata = make_case(config, args.rglru_source, args.mamba_source)
            is_decode = config["workload"] == "decode"
            if config["backend"] == "official_pytorch_source":
                samples = 5 if config["length"] >= 8192 else 9
                warmup = 2
            elif config["model"] == "mamba3":
                samples, warmup = 11, 3
            else:
                samples, warmup = 21, 5
            timing, compile_seconds = measure(operation, warmup=warmup, samples=samples,
                                               inner=100 if is_decode else 1)
            tokens = config["batch"] if is_decode else config["batch"] * config["length"]
            row = {
                "id": identifier, "status": "measured", **config, **metadata, **timing,
                "compile_setup_seconds": compile_seconds,
                "tokens_per_second": tokens * 1000 / timing["median_ms"],
                "logical_bytes_lower_bound": logical_bytes,
                "logical_effective_gbps": (logical_bytes / timing["median_ms"] / 1e6) if logical_bytes else None,
                "sfu_static_counts": sfu,
                "hardware_counter_scope": "N/A: Nsight Compute unavailable; logical traffic and source-derived SFU counts only",
            }
        except Exception as error:
            unsupported = config["model"] == "mamba3" and config["workload"] == "decode"
            row = {
                "id": identifier, "status": "unsupported" if unsupported else "failed",
                **config, "logical_bytes_lower_bound": logical_bytes,
                "sfu_static_counts": sfu,
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_json(path, row)
        print(json.dumps({key: row.get(key) for key in ("model", "backend", "workload", "batch", "length", "status", "median_ms", "error")}), flush=True)
        torch.cuda.empty_cache()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))]
    rows.sort(key=lambda row: (row.get("workload", ""), row.get("batch", 0), row.get("length", 0), row.get("track", ""), row.get("model", ""), row.get("backend", "")))
    summary = {
        "schema_version": 2,
        "title": "Equal-kernel SAMU vs official-equation RG-LRU, with official Mamba-3 context",
        "tracks": {
            "A_equal_triton": "SAMU and RG-LRU custom Triton inference paths; d_model=128, 16,772 vs 16,768 parameters, 512-byte FP32 state per batch element.",
            "B_official": "Pinned official public implementations. Mamba-3 is same-width best-native and is not parameter/state matched.",
        },
        "equation_audit": audit,
        "environment": env,
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", rows)
    atomic_json(output / "run_manifest.json", {
        "command": " ".join(sys.argv),
        "selected_config_count": len(selected),
        "expected_full_config_count": len(configs()),
        "result_row_count": len(rows),
        "measured": sum(row.get("status") == "measured" for row in rows),
        "unsupported": sum(row.get("status") == "unsupported" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "status": "complete",
    })


if __name__ == "__main__":
    main()
