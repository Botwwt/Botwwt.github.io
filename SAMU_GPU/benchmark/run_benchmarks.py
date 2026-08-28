"""Run resumable SAMU / official RG-LRU / official Mamba-3 GPU benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import torch

from kernels import (
    load_official_mamba2,
    load_official_mamba3,
    load_official_rglru,
    make_samu_parameters,
    samu_chunked_compressed,
    samu_controller,
    samu_decode,
    samu_serial,
    samu_tree_materialized,
    samu_write,
)


SAMU_SOURCE = "sha256:164500431103d982813fcff505e2828d4b8714e5891b9051c00150d4febc2477"
MAMBA_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    try:
        import triton
        triton_version = triton.__version__
    except Exception as error:
        triton_version = f"unavailable:{type(error).__name__}"
    packages = {}
    for name in ("jax", "flax", "tilelang", "cutlass", "quack", "causal_conv1d"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception as error:
            packages[name] = f"unavailable:{type(error).__name__}"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_and_board": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "ncu": command_output(["bash", "-lc", "command -v ncu || true"]) or "unavailable",
        "nsys": command_output(["bash", "-lc", "command -v nsys || true"]) or "unavailable",
        "packages": packages,
        "source_commits": {"samu": SAMU_SOURCE, "mamba3": MAMBA_COMMIT, "recurrentgemma": RGLRU_COMMIT},
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def result_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    label = "-".join(str(config.get(key, "na")) for key in ("model", "backend", "workload", "dtype", "batch", "length", "modes"))
    return f"{label}-{digest}".replace("/", "_")


def synchronize_result(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return synchronize_result(item)
            except TypeError:
                pass
    raise TypeError("operation returned no tensor")


def measure(operation: Callable[[], Any], *, warmup: int, samples: int, inner_iterations: int, training: bool) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        value = operation()
        synchronize_result(value)
    torch.cuda.synchronize()
    durations = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = None
        for _inner in range(inner_iterations):
            value = operation()
        synchronize_result(value)
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) / inner_iterations)
    peak = torch.cuda.max_memory_allocated()
    return {
        "raw_samples_ms": durations,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "p10_ms": percentile(durations, .10),
        "p90_ms": percentile(durations, .90),
        "p95_ms": percentile(durations, .95),
        "std_ms": statistics.pstdev(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "peak_allocated_bytes": peak,
        "inner_iterations": inner_iterations,
        "samples": samples,
        "timing": "CUDA events; synchronized; compile/setup outside recorded samples",
    }


def samu_parameter_count(p) -> int:
    return sum(value.numel() for value in p.__dict__.values())


def make_samu_case(config: dict[str, Any], device: str):
    dtype = dtype_from_name(config["dtype"])
    batch, length, d_model, modes = config["batch"], config["length"], config["d_model"], config["modes"]
    torch.manual_seed(1729)
    u = torch.randn(batch, length, d_model, device=device, dtype=dtype, requires_grad=config["workload"] == "forward_backward")
    p = make_samu_parameters(d_model, modes, device, dtype)
    backend = config["backend"]

    if config["workload"] == "decode":
        token = u[:, 0].detach()
        state = (torch.randn(batch, modes, device=device, dtype=dtype), torch.randn(batch, modes, device=device, dtype=dtype))
        precomputed = None
        if backend == "decode_precomputed_factorized":
            wr, wi = samu_write(token, p)
            c, d = samu_controller(token, p)
            precomputed = (wr, wi, c, d)
        factorized = "factorized" in backend
        def forward():
            return samu_decode(token, state, p, factorized=factorized, precomputed=precomputed)
        operation = forward
    else:
        if backend == "serial_direct":
            forward = lambda: samu_serial(u, p, factorized=False)
        elif backend == "serial_factorized":
            forward = lambda: samu_serial(u, p, factorized=True)
        elif backend == "tree_materialized_direct":
            forward = lambda: samu_tree_materialized(u, p, factorized=False)
        elif backend == "tree_materialized_factorized":
            forward = lambda: samu_tree_materialized(u, p, factorized=True)
        elif backend == "chunked_compressed":
            forward = lambda: samu_chunked_compressed(u, p, int(config["chunk_size"]))
        else:
            raise ValueError(f"unknown SAMU backend {backend}")
        if config["workload"] == "forward_backward":
            def operation():
                if u.grad is not None:
                    u.grad = None
                out = forward()
                out.float().square().mean().backward()
                return out
        else:
            operation = lambda: forward()
    metadata = {
        "parameter_count": samu_parameter_count(p),
        "source_commit": SAMU_SOURCE,
        "implementation_class": "audited_pytorch_reference_or_prototype",
        "real_state_scalars": 2 * modes,
        "logical_bytes": batch * length * 2 * modes * torch.tensor([], dtype=dtype).element_size() * 2,
    }
    return operation, metadata


def make_rglru_case(config: dict[str, Any], device: str, source_root: Path):
    dtype = dtype_from_name(config["dtype"])
    width = config["d_model"]
    if config["modes"] * 2 != width:
        raise ValueError("official RG-LRU state-byte protocol requires width=2*modes")
    RGLRU = load_official_rglru(source_root)
    model = RGLRU(width=width, num_heads=int(config.get("num_heads", 2)), device=device, dtype=dtype).to(device).eval()
    batch, length = config["batch"], config["length"]
    u = torch.randn(batch, length, width, device=device, dtype=dtype, requires_grad=config["workload"] == "forward_backward")
    positions = torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
    if config["workload"] == "decode":
        token, position = u[:, :1].detach(), torch.ones(batch, 1, device=device, dtype=torch.long)
        cache = torch.randn(batch, width, device=device, dtype=torch.float32)
        operation = lambda: model(token, position, cache=cache, return_cache=True)
    else:
        forward = lambda: model(u, positions, cache=None, return_cache=True)[0]
        if config["workload"] == "forward_backward":
            def operation():
                model.zero_grad(set_to_none=True)
                if u.grad is not None:
                    u.grad = None
                out = forward()
                out.float().square().mean().backward()
                return out
        else:
            operation = forward
    return operation, {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_commit": RGLRU_COMMIT,
        "implementation_class": "official_recurrentgemma_pytorch_reference",
        "real_state_scalars": width,
        "logical_bytes": None,
    }


def make_mamba3_case(config: dict[str, Any], device: str, source_root: Path):
    if config["dtype"] != "bf16":
        raise RuntimeError("official Mamba-3 performance path is benchmarked in BF16 only")
    dtype = torch.bfloat16
    Mamba3 = load_official_mamba3(source_root)
    model = Mamba3(
        d_model=config["d_model"], d_state=config.get("d_state", 64), headdim=64,
        chunk_size=config.get("chunk_size", 64), is_mimo=False,
        dtype=dtype, device=device,
    ).to(device).eval()
    batch, length, width = config["batch"], config["length"], config["d_model"]
    u = torch.randn(batch, length, width, device=device, dtype=dtype, requires_grad=config["workload"] == "forward_backward")
    if config["workload"] == "decode":
        if getattr(sys.modules.get("mamba_ssm.modules.mamba3"), "mamba3_step_fn", None) is None:
            raise RuntimeError("official Mamba-3 decode uses CuTeDSL step kernel; unavailable in this environment and documented as H100-tested")
        states = model.allocate_inference_cache(batch, 1, device=device, dtype=dtype)
        token = u[:, 0].detach()
        operation = lambda: model.step(token, *states)
    else:
        forward = lambda: model(u)
        if config["workload"] == "forward_backward":
            def operation():
                model.zero_grad(set_to_none=True)
                if u.grad is not None:
                    u.grad = None
                out = forward()
                out.float().square().mean().backward()
                return out
        else:
            operation = forward
    # State is structured [B,nheads,headdim,d_state] plus angle/K/V caches.
    nheads = model.nheads
    state_scalars = nheads * model.headdim * model.d_state + nheads * model.num_rope_angles + nheads * model.d_state + nheads * model.headdim
    return operation, {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_commit": MAMBA_COMMIT,
        "implementation_class": "official_state_spaces_mamba3_siso_triton",
        "real_state_scalars": state_scalars,
        "logical_bytes": None,
    }


def make_mamba2_case(config: dict[str, Any], device: str, source_root: Path):
    if config["dtype"] != "bf16":
        raise RuntimeError("official Mamba-2 performance path is benchmarked in BF16 only")
    Mamba2 = load_official_mamba2(source_root)
    model = Mamba2(
        d_model=config["d_model"], d_state=config.get("d_state", 64),
        headdim=64, chunk_size=config.get("chunk_size", 256),
        dtype=torch.bfloat16, device=device,
    ).to(device).eval()
    u = torch.randn(config["batch"], config["length"], config["d_model"], device=device, dtype=torch.bfloat16)
    if config["workload"] != "forward":
        raise RuntimeError("this Mamba-2 coverage row currently targets official forward only")
    return lambda: model(u), {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_commit": MAMBA_COMMIT,
        "implementation_class": "official_state_spaces_mamba2",
        "real_state_scalars": None,
        "logical_bytes": None,
    }


def base_config(model, backend, workload, dtype, batch, length, modes, d_model, protocol, **extra):
    value = {
        "track": "A_best_available" if model in ("mamba2", "mamba3", "rglru") else "B_apples_to_apples",
        "model": model, "backend": backend, "workload": workload, "dtype": dtype,
        "batch": batch, "length": length, "modes": modes, "d_model": d_model,
        "matching_protocol": protocol, "chunk_size": None, "mode_tile": None,
    }
    value.update(extra)
    return value


def quick_configs() -> list[dict[str, Any]]:
    configs = []
    protocol = "same_width+state_bytes+parameter_approx"
    # Sequence scaling: the primary directly comparable width/state regime.
    for length in (128, 512, 2048, 8192):
        for backend in ("tree_materialized_factorized", "chunked_compressed"):
            configs.append(base_config("samu", backend, "forward", "bf16", 1, length, 64, 128, protocol, chunk_size=64 if backend == "chunked_compressed" else None))
        configs.append(base_config("rglru", "official_pytorch_serial", "forward", "bf16", 1, length, 64, 128, protocol, num_heads=2))
        configs.append(base_config("mamba3", "official_siso_triton", "forward", "bf16", 1, length, 64, 128, "same_width+best_native", d_state=64, chunk_size=64))
    # Forward + backward scaling. Keep the largest first-run shape sane.
    for length in (128, 512, 2048):
        configs.append(base_config("samu", "tree_materialized_factorized", "forward_backward", "bf16", 1, length, 64, 128, protocol))
        configs.append(base_config("samu", "chunked_compressed", "forward_backward", "bf16", 1, length, 64, 128, protocol, chunk_size=64))
        configs.append(base_config("rglru", "official_pytorch_serial", "forward_backward", "bf16", 1, length, 64, 128, protocol, num_heads=2))
        configs.append(base_config("mamba3", "official_siso_triton", "forward_backward", "bf16", 1, length, 64, 128, "same_width+best_native", d_state=64, chunk_size=64))
    # State/width scaling.
    for modes in (64, 128, 256):
        width = modes * 2
        configs.append(base_config("samu", "tree_materialized_factorized", "forward", "bf16", 1, 512, modes, width, protocol))
        configs.append(base_config("samu", "chunked_compressed", "forward", "bf16", 1, 512, modes, width, protocol, chunk_size=64))
        configs.append(base_config("rglru", "official_pytorch_serial", "forward", "bf16", 1, 512, modes, width, protocol, num_heads=2))
        configs.append(base_config("mamba3", "official_siso_triton", "forward", "bf16", 1, 512, modes, width, "same_width+best_native", d_state=modes, chunk_size=64))
    # Chunk sweep and phase implementation.
    for chunk in (16, 32, 64, 128, 256, 512):
        configs.append(base_config("samu", "chunked_compressed", "forward", "bf16", 1, 2048, 64, 128, protocol, chunk_size=chunk))
    for backend in ("serial_direct", "serial_factorized", "tree_materialized_direct", "tree_materialized_factorized"):
        configs.append(base_config("samu", backend, "forward", "bf16", 1, 128, 64, 128, protocol))
    # Decode scaling and isolated control/write preprocessing.
    for batch in (1, 4, 16):
        for backend in ("decode_direct", "decode_factorized", "decode_precomputed_factorized"):
            configs.append(base_config("samu", backend, "decode", "bf16", batch, 1, 64, 128, protocol))
        configs.append(base_config("rglru", "official_pytorch_serial", "decode", "bf16", batch, 1, 64, 128, protocol, num_heads=2))
    configs.append(base_config("mamba3", "official_cute_step", "decode", "bf16", 1, 1, 64, 128, "same_width+best_native", d_state=64))
    configs.append(base_config("mamba2", "official_fused_forward", "forward", "bf16", 1, 512, 64, 128, "same_width+best_native", d_state=64, chunk_size=256))
    # FP32 reference points and larger batches.
    for model, backend in (("samu", "tree_materialized_factorized"), ("samu", "chunked_compressed"), ("rglru", "official_pytorch_serial")):
        configs.append(base_config(model, backend, "forward", "fp32", 1, 512, 64, 128, protocol, chunk_size=64 if backend == "chunked_compressed" else None, num_heads=2))
    for batch in (2, 4, 8, 16):
        configs.append(base_config("samu", "chunked_compressed", "forward", "bf16", batch, 512, 64, 128, protocol, chunk_size=64))
        configs.append(base_config("mamba3", "official_siso_triton", "forward", "bf16", batch, 512, 64, 128, "same_width+best_native", d_state=64, chunk_size=64))
    # Explicit unsupported design rows: not silently omitted.
    configs.append(base_config("samu", "warp_shared_control_custom_kernel", "forward", "bf16", 1, 512, 64, 128, protocol))
    configs.append(base_config("samu", "advanced_scan_custom_kernel", "forward", "bf16", 1, 512, 64, 128, protocol))
    return configs


def full_configs() -> list[dict[str, Any]]:
    configs = quick_configs()
    protocol = "same_width+state_bytes+parameter_approx"
    # High-information boundary slice comes first so --max-configs produces a
    # scientifically useful resumable partial FULL run instead of arbitrary grid order.
    for length in (32768, 65536):
        configs.append(base_config("samu", "tree_materialized_factorized", "forward", "bf16", 1, length, 64, 128, protocol))
        configs.append(base_config("samu", "chunked_compressed", "forward", "bf16", 1, length, 64, 128, protocol, chunk_size=16))
        configs.append(base_config("rglru", "official_pytorch_serial", "forward", "bf16", 1, length, 64, 128, protocol, num_heads=2))
        configs.append(base_config("mamba3", "official_siso_triton", "forward", "bf16", 1, length, 64, 128, "same_width+best_native", d_state=64, chunk_size=64))
    for backend, model in (("tree_materialized_factorized", "samu"), ("chunked_compressed", "samu"), ("official_pytorch_serial", "rglru"), ("official_siso_triton", "mamba3")):
        configs.append(base_config(model, backend, "forward_backward", "bf16", 1, 8192, 64, 128, protocol if model != "mamba3" else "same_width+best_native", chunk_size=16 if backend == "chunked_compressed" else 64 if model == "mamba3" else None, num_heads=2 if model == "rglru" else None, d_state=64 if model == "mamba3" else None))
    for model, backend in (("samu", "tree_materialized_factorized"), ("samu", "chunked_compressed"), ("rglru", "official_pytorch_serial"), ("mamba3", "official_siso_triton")):
        configs.append(base_config(model, backend, "forward", "bf16", 1, 512, 512, 1024, protocol if model != "mamba3" else "same_width+best_native", chunk_size=16 if backend == "chunked_compressed" else 64 if model == "mamba3" else None, num_heads=2 if model == "rglru" else None, d_state=512 if model == "mamba3" else None))
    for batch in (2, 4, 8, 16):
        configs.append(base_config("samu", "tree_materialized_factorized", "forward", "bf16", batch, 512, 64, 128, protocol))
        configs.append(base_config("rglru", "official_pytorch_serial", "forward", "bf16", batch, 512, 64, 128, protocol, num_heads=2))
    for mode_tile in (64, 128, 256, 512):
        configs.append(base_config("samu", "warp_shared_control_custom_kernel", "forward", "bf16", 1, 512, 64, 128, protocol, mode_tile=mode_tile))
    for batch in (1, 2, 4, 8, 16):
        for length in (128, 512, 2048, 8192, 32768):
            for modes in (64, 128, 256, 512):
                width = modes * 2
                for dtype in ("fp32", "bf16"):
                    for backend in ("tree_materialized_factorized", "chunked_compressed"):
                        configs.append(base_config("samu", backend, "forward", dtype, batch, length, modes, width, protocol, chunk_size=64 if backend == "chunked_compressed" else None))
                    configs.append(base_config("rglru", "official_pytorch_serial", "forward", dtype, batch, length, modes, width, protocol, num_heads=2))
                configs.append(base_config("mamba3", "official_siso_triton", "forward", "bf16", batch, length, modes, width, "same_width+best_native", d_state=modes, chunk_size=64))
    # Stable de-duplication preserving order.
    deduplicated = {}
    for config in configs:
        deduplicated[result_id(config)] = config
    return list(deduplicated.values())


def memory_guard(config: dict[str, Any], total_memory: int) -> tuple[bool, int]:
    bytes_per = {"fp32": 4, "bf16": 2, "fp16": 2}[config["dtype"]]
    b, l, d, m = config["batch"], config["length"], config["d_model"], config["modes"]
    # Conservative working estimate including output/intermediates/autograd multiplier.
    estimate = b * l * (d + 12 * m) * bytes_per + 12 * d * max(d, m) * bytes_per
    if config["workload"] == "forward_backward":
        estimate *= 4
    return estimate < total_memory * .55, estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mamba-source", type=Path, default=Path("../mamba-official"))
    parser.add_argument("--rglru-source", type=Path, default=Path("../recurrentgemma-official"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--warmup", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = environment()
    atomic_json(output / "environment.json", env)
    configs = quick_configs() if args.preset == "quick" else full_configs()
    if args.max_configs:
        configs = configs[: args.max_configs]
    manifest = {
        "preset": args.preset, "config_count": len(configs), "resume": args.resume,
        "command": " ".join(sys.argv), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": "../BENCHMARK_METHODOLOGY.md", "status": "running",
    }
    atomic_json(output / "run_manifest.json", manifest)
    samples = args.samples or (7 if args.preset == "quick" else 12)
    warmup = args.warmup or (3 if args.preset == "quick" else 5)
    for index, config in enumerate(configs, start=1):
        identifier = result_id(config)
        path = raw / f"{identifier}.json"
        if args.resume and path.exists():
            print(f"[{index}/{len(configs)}] resume {identifier}", flush=True)
            continue
        base = {**config, "id": identifier, "environment_file": "../environment.json"}
        supported, estimate = memory_guard(config, env["gpu_total_memory_bytes"])
        base["memory_guard_estimate_bytes"] = estimate
        if not supported:
            atomic_json(path, {**base, "status": "skipped_memory_guard", "error": "conservative estimate exceeds 55% of GPU memory"})
            print(f"[{index}/{len(configs)}] memory-guard {identifier}", flush=True)
            continue
        if config["backend"] in ("warp_shared_control_custom_kernel", "advanced_scan_custom_kernel"):
            atomic_json(path, {**base, "status": "unsupported", "error_type": "NotImplemented", "error": "No audited custom kernel exists; row retained to prevent silent omission", "source_commit": SAMU_SOURCE})
            print(f"[{index}/{len(configs)}] unsupported {identifier}", flush=True)
            continue
        try:
            if config["model"] == "samu":
                operation, metadata = make_samu_case(config, "cuda")
            elif config["model"] == "rglru":
                operation, metadata = make_rglru_case(config, "cuda", args.rglru_source)
            elif config["model"] == "mamba2":
                operation, metadata = make_mamba2_case(config, "cuda", args.mamba_source)
            else:
                operation, metadata = make_mamba3_case(config, "cuda", args.mamba_source)
            # First invocation compiles/autotunes outside recorded timing.
            compile_start = time.perf_counter()
            first = operation()
            synchronize_result(first)
            torch.cuda.synchronize()
            compile_setup_seconds = time.perf_counter() - compile_start
            is_decode = config["workload"] == "decode"
            inner = 100 if is_decode else 1
            timing = measure(operation, warmup=warmup, samples=samples, inner_iterations=inner, training=config["workload"] == "forward_backward")
            tokens = config["batch"] * max(config["length"], 1)
            median_seconds = timing["median_ms"] / 1000
            record = {
                **base, **metadata, **timing, "status": "measured",
                "compile_setup_seconds": compile_setup_seconds,
                "tokens_per_second": tokens / median_seconds if median_seconds else None,
                "profiler_measured_dram_bytes": None,
                "profiler_measured_l2_bytes": None,
                "occupancy": None, "registers_per_thread": None, "tensor_core_utilization": None,
                "sfu_utilization": None, "sm_utilization": None,
            }
            atomic_json(path, record)
            print(f"[{index}/{len(configs)}] {identifier} median={timing['median_ms']:.4f} ms", flush=True)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            atomic_json(path, {**base, "status": "oom", "error_type": type(error).__name__, "error": str(error)[:2000]})
            print(f"[{index}/{len(configs)}] OOM {identifier}", flush=True)
        except Exception as error:
            unsupported = config["model"] == "mamba2" or (config["model"] == "mamba3" and config["workload"] == "decode")
            atomic_json(path, {**base, "status": "unsupported" if unsupported else "failed", "error_type": type(error).__name__, "error": str(error)[:3000], "traceback": traceback.format_exc()[-6000:]})
            print(f"[{index}/{len(configs)}] failed {identifier}: {type(error).__name__}: {error}", flush=True)
        finally:
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(output / "run_manifest.json", manifest)
    subprocess.check_call([sys.executable, str(Path(__file__).with_name("aggregate.py")), str(output)])


if __name__ == "__main__":
    main()
