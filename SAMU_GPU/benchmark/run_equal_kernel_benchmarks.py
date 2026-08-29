"""Equal-maturity SAMU / official-equation RG-LRU Triton benchmarks.

The comparison matches width, parameter count, FP32 recurrent-state bytes,
precision, launch class, and timing method.  RG-LRU equations and numerical
semantics are validated against a pinned RecurrentGemma source checkout.
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch
import triton

from kernels import load_official_rglru, make_samu_parameters
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
EQUAL_PROTOCOL = "equal_d_model+parameters_within_4+equal_fp32_state_bytes"


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
        },
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def measure(operation: Callable[[], Any], *, warmup_ms: int, rep_ms: int,
            inner: int = 1) -> tuple[dict[str, Any], float]:
    torch.cuda.empty_cache()
    compile_start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats()
    # Use Triton's driver-level event implementation rather than torch.cuda.Event.
    # On this pinned Torch 2.1 / CUDA 11.8 stack the latter quantized 7-50 us
    # kernels into 0.16-0.22 ms values and could reverse rankings.  do_bench
    # preallocates driver events, clears L2 before each timed call, synchronizes
    # once, and can return every individual sample.
    requested_warmup_ms = max(1, warmup_ms)
    requested_rep_ms = max(25, rep_ms)
    values = [float(value) for value in triton.testing.do_bench(
        operation, warmup=requested_warmup_ms, rep=requested_rep_ms, return_mode="all"
    )]
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
        "samples": len(values),
        "requested_warmup_ms": requested_warmup_ms,
        "requested_rep_ms": requested_rep_ms,
        "inner_iterations": 1,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "timing": "Triton driver CUDA events; L2 cleared per sample; first-call compile/setup excluded",
    }, compile_seconds


def stabilize_gpu(seconds: float = 2.0) -> dict[str, Any]:
    """Bring clocks to steady state before latency sampling."""

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    started = time.perf_counter()
    launches = 0
    while time.perf_counter() - started < seconds:
        for _ in range(16):
            torch.mm(a, a, out=out)
            launches += 1
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del a, out
    torch.cuda.empty_cache()
    return {
        "method": "repeated 4096x4096 BF16 GEMM followed by synchronization",
        "requested_seconds": seconds,
        "elapsed_seconds": elapsed,
        "launches": launches,
        "post_warmup_clock_power": command_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ]),
    }


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
        "matching_protocol": EQUAL_PROTOCOL,
    }
    value.update(extra)
    return value


def configs() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for length in (128, 512, 2048, 8192, 32768, 65536):
        result.append(base_config("A_equal_triton", "samu", "triton_auto", "prefill", 1, length))
        result.append(base_config("A_equal_triton", "rglru", "triton_auto", "prefill", 1, length))
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
        passes = 1 if config["backend"] == "triton_serial" else 2
        dynamic = b * l * (d * 2 + projected * 2 * (1 + passes) + 2 * m * 2)
        weights = d * math.ceil(projected / 16) * 16 * 2
        auto_chunk = 8 if l <= 256 else 16 if l <= 512 else 32
        chunks = math.ceil(l / (config.get("chunk_size") or auto_chunk)) if passes == 2 else 0
        metadata = b * chunks * 48 * m if passes == 2 else 0
        bounded = config["backend"] == "triton_auto" and l <= 128
        return dynamic + weights + metadata, {
            "exp_or_sigmoid_per_token": passes * (2 if bounded else (m + 3)), "sqrt_per_token": 0,
            "trig_sfu_per_token": 0, "transition_evaluations_per_token": passes,
            "bounded_exp_polynomial": bounded,
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


def make_case(config: dict[str, Any], rglru_root: Path):
    torch.manual_seed(1729)
    b, l, d, m = config["batch"], config["length"], config["d_model"], config["modes"]
    if config["model"] == "samu":
        params = make_samu_parameters(d, m, "cuda", torch.bfloat16)
        packed = pack_samu_parameters(params)
        parameter_count = sum(value.numel() for value in params.__dict__.values())
        if config["workload"] == "decode":
            token = torch.randn(b, d, device="cuda", dtype=torch.bfloat16)
            state = (torch.randn(b, m, device="cuda"), torch.randn(b, m, device="cuda"))
            decode_warps = 8
            decode_block = 64 if b > 16 else 32
            operation = lambda: samu_triton_decode(
                token, state, params, packed,
                block_m=decode_block, num_warps=decode_warps
            )
            launches = 1
        else:
            x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
            if config["backend"] == "triton_auto":
                chunk = 8 if l <= 256 else 16 if l <= 512 else 32
                auto_warps = 4 if l <= 256 else 2
                operation = lambda: samu_triton_chunked(
                    x, params, chunk, packed, num_warps=auto_warps,
                    bounded_poly=packed.bounded_poly_safe if l <= 128 else False,
                    return_cache=True,
                )
                launches = 4
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
                chunk = 8 if l <= 256 else 16 if l <= 1024 else 32
                auto_warps = 4 if l <= 2048 else 2
                operation = lambda: rglru_triton_chunked(
                    x, positions, packed, chunk, num_warps=auto_warps
                )
            elif config["backend"] == "triton_serial":
                operation = lambda: rglru_triton_serial(x, positions, packed)
            else:
                explicit_warps = 4 if l <= 2048 else 2
                operation = lambda: rglru_triton_chunked(
                    x, positions, packed, int(config["chunk_size"]),
                    num_warps=explicit_warps,
                )
            launches = 4 if config["backend"] != "triton_serial" else 2
        return operation, {
            "parameter_count": parameter_count, "state_bytes_per_batch": d * 4,
            "implementation_class": "custom_triton_official_equations",
            "source_commit": digest(Path(__file__).with_name("triton_rglru.py")),
            "official_equation_source_commit": RGLRU_COMMIT,
            "precision_policy": "official eager-BF16 pointwise rounding; FP32 recurrent state/cache",
            "expected_cuda_launches": launches,
        }
    raise ValueError(f"unknown model: {config['model']}")


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
    adversarial_output_error = 0.0
    adversarial_cache_error = 0.0
    for scale in (2.0 ** -8, 2.0 ** -2, 1.0, 4.0, 16.0, 64.0):
        probe = (torch.randn(4, 1, 128, device="cuda") * scale).to(torch.bfloat16)
        probe_cache = torch.randn(4, 128, device="cuda") * scale
        probe_pos = torch.tensor([[0], [1], [7], [0]], device="cuda")
        expected, expected_cache = model(probe, probe_pos, cache=probe_cache, return_cache=True)
        actual, actual_cache = rglru_triton_decode(probe, probe_pos, packed_rg, probe_cache)
        adversarial_output_error = max(adversarial_output_error, float((actual - expected).abs().max()))
        adversarial_cache_error = max(adversarial_cache_error, float((actual_cache - expected_cache).abs().max()))

    samu_params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    packed_samu = pack_samu_parameters(samu_params)
    samu_x = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    samu_reference, samu_reference_cache = samu_packed_reference(
        samu_x, samu_params, packed_samu, return_cache=True
    )
    samu_serial, samu_serial_cache = samu_triton_serial(samu_x, samu_params, packed_samu, return_cache=True)
    samu_chunk, samu_chunk_cache = samu_triton_chunked(samu_x, samu_params, 16, packed_samu, return_cache=True)
    samu_auto_short, samu_auto_short_cache = samu_triton_auto(
        samu_x, samu_params, packed_samu, return_cache=True
    )
    exact_exp_samu = replace(packed_samu, use_bounded_poly=False)
    bounded_samu = pack_samu_parameters(samu_params, enable_bounded_poly=True)
    bounded_out, bounded_cache = samu_triton_chunked(
        samu_x, samu_params, 16, bounded_samu, compressed_p=False, return_cache=True
    )
    exact_exp_out, exact_exp_cache = samu_triton_chunked(
        samu_x, samu_params, 16, exact_exp_samu, compressed_p=False, return_cache=True
    )

    bound = packed_samu.phase_scale
    sample_d = torch.linspace(-bound, bound, 10001, device="cuda")
    d2, d4 = sample_d.square(), sample_d.square().square()
    d6 = d4 * d2
    cos_poly = 1 - .5 * d2 + d4 / 24 - d6 / 720
    sin_poly = sample_d * (1 - d2 / 6 + d4 / 120 - d6 / 5040)
    audit = {
        "rglru_reference": "official recurrentgemma.torch.layers.RGLRU at pinned commit",
        "rglru_reference_commit": RGLRU_COMMIT,
        "rglru_reset_cases": ["nonzero first position with nonzero h0", "mid-sequence reset t=5", "mid-sequence reset t=77", "decode reset/non-reset"],
        "rglru_serial_max_abs": float((serial - official).abs().max()),
        "rglru_serial_mean_abs": float((serial - official).abs().float().mean()),
        "rglru_serial_cache_max_abs": float((serial_cache - official_cache).abs().max()),
        "rglru_chunk16_max_abs": float((chunk - official).abs().max()),
        "rglru_chunk16_mean_abs": float((chunk - official).abs().float().mean()),
        "rglru_chunk16_cache_max_abs": float((chunk_cache - official_cache).abs().max()),
        "rglru_decode_max_abs": float((triton_decode - official_decode).abs().max()),
        "rglru_decode_cache_max_abs": float((triton_decode_cache - official_decode_cache).abs().max()),
        "rglru_adversarial_decode_max_abs": adversarial_output_error,
        "rglru_adversarial_decode_cache_max_abs": adversarial_cache_error,
        "samu_serial_max_abs": float((samu_serial - samu_reference).abs().max()),
        "samu_chunk16_max_abs": float((samu_chunk - samu_reference).abs().max()),
        "samu_serial_cache_max_abs": float((torch.stack(samu_serial_cache, -1) - torch.stack(samu_reference_cache, -1)).abs().max()),
        "samu_chunk16_cache_max_abs": float((torch.stack(samu_chunk_cache, -1) - torch.stack(samu_reference_cache, -1)).abs().max()),
        "samu_bounded_exp_safe": bounded_samu.bounded_poly_safe,
        "samu_bounded_exp_enabled_in_production": "auto prefill L<=128 only; certified parameters required",
        "samu_bounded_exp_max_abs_c": bounded_samu.max_abs_c,
        "samu_bounded_exp_max_decay_exponent": bounded_samu.max_decay_exponent,
        "samu_bounded_vs_exact_output_max_abs": float((bounded_out - exact_exp_out).abs().max()),
        "samu_bounded_vs_exact_cache_max_abs": float((torch.stack(bounded_cache, -1) - torch.stack(exact_exp_cache, -1)).abs().max()),
        "samu_auto_short_max_abs": float((samu_auto_short - samu_reference).abs().max()),
        "samu_auto_short_cache_max_abs": float((torch.stack(samu_auto_short_cache, -1) - torch.stack(samu_reference_cache, -1)).abs().max()),
        "samu_phase_delta_bound_rad": bound,
        "samu_sin_polynomial_max_abs": float((sin_poly - torch.sin(sample_d)).abs().max()),
        "samu_cos_polynomial_max_abs": float((cos_poly - torch.cos(sample_d)).abs().max()),
    }
    limits = {
        "rglru_serial_max_abs": 2e-3,
        "rglru_serial_cache_max_abs": 5e-5,
        "rglru_chunk16_max_abs": 2e-3,
        "rglru_chunk16_cache_max_abs": 5e-5,
        "rglru_decode_max_abs": 2e-3,
        "rglru_decode_cache_max_abs": 5e-5,
        "rglru_adversarial_decode_max_abs": 2e-3,
        "rglru_adversarial_decode_cache_max_abs": 5e-5,
        "samu_serial_max_abs": 1e-2,
        "samu_serial_cache_max_abs": 1e-4,
        "samu_chunk16_max_abs": 1e-2,
        "samu_chunk16_cache_max_abs": 1e-4,
        "samu_auto_short_max_abs": 1e-2,
        "samu_auto_short_cache_max_abs": 1e-4,
        "samu_bounded_vs_exact_output_max_abs": 1e-2,
        "samu_bounded_vs_exact_cache_max_abs": 1e-4,
        "samu_sin_polynomial_max_abs": 1e-6,
        "samu_cos_polynomial_max_abs": 1e-6,
    }
    failures = {
        name: {"measured": audit[name], "limit": limit}
        for name, limit in limits.items()
        if audit[name] > limit
    }
    audit["acceptance_limits"] = limits
    audit["passed"] = not failures
    audit["failures"] = failures
    if failures:
        raise RuntimeError(f"correctness acceptance failed: {json.dumps(failures, sort_keys=True)}")
    return audit


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-model", choices=("samu", "rglru"))
    parser.add_argument("--only-custom-triton", action="store_true")
    parser.add_argument(
        "--model-order", choices=("samu,rglru", "rglru,samu"),
        default="samu,rglru",
        help="reverse this in a second run and pool raw samples to counterbalance order",
    )
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
    stabilization = stabilize_gpu()
    print("stabilization", json.dumps(stabilization), flush=True)
    selected = [config for config in configs()
                if (args.only_model is None or config["model"] == args.only_model)
                and (not args.only_custom_triton or (
                    config["model"] in ("samu", "rglru")
                    and config["backend"] != "official_pytorch_source"
                ))]
    if args.only_model is None and args.model_order == "rglru,samu":
        reordered = []
        for offset in range(0, len(selected), 2):
            pair = selected[offset:offset + 2]
            if len(pair) == 2 and {row["model"] for row in pair} == {"samu", "rglru"}:
                pair.sort(key=lambda row: 0 if row["model"] == "rglru" else 1)
            reordered.extend(pair)
        selected = reordered
    for index, config in enumerate(selected, 1):
        identifier = row_id(config)
        path = raw / f"{identifier}.json"
        if args.resume and path.exists():
            print(f"[{index}/{len(selected)}] cached {identifier}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] run {identifier}", flush=True)
        logical_bytes, sfu = logical_metrics(config)
        try:
            operation, metadata = make_case(config, args.rglru_source)
            is_decode = config["workload"] == "decode"
            # Short 5 ms warmups left the first H800 cases in a lower power
            # state and measurably changed their ranking.  The final run uses
            # sustained per-case warmup after the global clock stabilization.
            warmup_ms = 100
            rep_ms = 200
            inner = 1
            timing, compile_seconds = measure(operation, warmup_ms=warmup_ms, rep_ms=rep_ms,
                                               inner=inner)
            tokens = config["batch"] if is_decode else config["batch"] * config["length"]
            row = {
                "id": identifier, "status": "measured", **config, **metadata, **timing,
                "compile_setup_seconds": compile_seconds,
                "tokens_per_second": tokens * 1000 / timing["median_ms"],
                "logical_bytes_lower_bound": logical_bytes,
                "logical_effective_gbps": (logical_bytes / timing["median_ms"] / 1e6) if logical_bytes else None,
                "sfu_static_counts": sfu,
                "hardware_counter_scope": (
                    "Nsight Compute 2025.3.1 installed, but host kernel has "
                    "RmProfilingAdminOnly=1 and the container lacks CAP_SYS_ADMIN "
                    "(ERR_NVGPUCTRPERM); logical traffic and source-derived SFU counts only"
                ),
            }
        except Exception as error:
            row = {
                "id": identifier, "status": "failed",
                **config, "logical_bytes_lower_bound": logical_bytes,
                "sfu_static_counts": sfu,
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_json(path, row)
        print(json.dumps({key: row.get(key) for key in ("model", "backend", "workload", "batch", "length", "status", "median_ms", "error")}), flush=True)
        torch.cuda.empty_cache()
    selected_ids = {row_id(config) for config in configs()}
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw.glob("*.json"))
            if path.stem in selected_ids]
    rows.sort(key=lambda row: (row.get("workload", ""), row.get("batch", 0), row.get("length", 0), row.get("track", ""), row.get("model", ""), row.get("backend", "")))
    summary = {
        "schema_version": 2,
        "generated_from_raw": True,
        "title": "Equal-kernel SAMU vs official-equation RG-LRU",
        "model_order": args.model_order,
        "gpu_stabilization": stabilization,
        "tracks": {
            "A_equal_triton": "SAMU and RG-LRU custom Triton inference paths; d_model=128, 16,772 vs 16,768 parameters, 512-byte FP32 state per batch element.",
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
        "failed": sum(row.get("status") == "failed" for row in rows),
        "status": "complete",
    })


if __name__ == "__main__":
    main()
