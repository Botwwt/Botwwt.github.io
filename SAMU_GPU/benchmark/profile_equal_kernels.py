"""Profile representative SAMU and official-equation RG-LRU inference paths."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from triton.runtime.jit import JITFunction

from kernels import load_official_rglru, make_samu_parameters
import triton_rglru as rg
import triton_samu as samu


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def synchronize(value) -> None:
    del value
    torch.cuda.synchronize()


def cuda_device_attribute(attribute: int, fallback: int) -> int:
    """Read a numeric CUDA device limit without relying on Torch's short repr."""

    candidates = [ctypes.util.find_library("cudart"), "libcudart.so.12",
                  "libcudart.so.11.0", "libcudart.so"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            runtime = ctypes.CDLL(candidate)
            value = ctypes.c_int()
            if runtime.cudaDeviceGetAttribute(ctypes.byref(value), attribute, 0) == 0:
                return value.value
        except OSError:
            continue
    return fallback


def device_limits(props) -> dict[str, int]:
    # cudaDevAttrMaxSharedMemoryPerMultiprocessor=81,
    # cudaDevAttrMaxRegistersPerMultiprocessor=82,
    # cudaDevAttrMaxBlocksPerMultiprocessor=106.
    return {
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "max_warps_per_sm": props.max_threads_per_multi_processor // 32,
        "registers_per_sm": cuda_device_attribute(82, 65536),
        "shared_memory_per_sm": cuda_device_attribute(81, 102400),
        "max_blocks_per_sm": cuda_device_attribute(106, 16),
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=15
        ).strip()
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def counter_permission_evidence() -> dict[str, object]:
    params = Path("/proc/driver/nvidia/params")
    admin_only = None
    if params.exists():
        for line in params.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("RmProfilingAdminOnly:"):
                admin_only = line.split(":", 1)[1].strip()
                break
    return {
        "available": False,
        "ncu_version": command_output([
            "/opt/nvidia/nsight-compute/2025.3.1/ncu", "--version"
        ]),
        "host_RmProfilingAdminOnly": admin_only,
        "container_capabilities": command_output(["capsh", "--print"]),
        "collection_error_observed": "ERR_NVGPUCTRPERM",
        "policy": "hardware DRAM/L2/SFU/achieved-occupancy fields remain null",
    }


def compiled_kernels(functions: list[JITFunction], limits: dict[str, int]) -> list[dict]:
    result, seen = [], set()
    max_threads = limits["max_threads_per_sm"]
    max_warps = max_threads // 32
    regs_per_sm = limits["registers_per_sm"]
    shared_per_sm = limits["shared_memory_per_sm"]
    max_blocks = limits["max_blocks_per_sm"]
    for function in functions:
        for device_cache in function.device_caches.values():
            compiled = device_cache[0]
            for kernel in compiled.values():
                if kernel.hash in seen:
                    continue
                seen.add(kernel.hash)
                warps = kernel.metadata.num_warps
                threads = warps * 32
                registers = kernel.n_regs
                shared = kernel.metadata.shared
                by_threads = max_threads // threads
                by_registers = regs_per_sm // max(registers * threads, 1)
                by_shared = shared_per_sm // shared if shared else max_blocks
                resident_blocks = min(max_blocks, by_threads, by_registers, by_shared)
                occupancy = resident_blocks * warps / max_warps
                resource_limits = {
                    "blocks_by_threads": by_threads,
                    "blocks_by_registers_unrounded": by_registers,
                    "blocks_by_shared_memory": by_shared,
                    "hardware_max_blocks": max_blocks,
                }
                result.append({
                    "name": kernel.name,
                    "hash": kernel.hash,
                    "registers_per_thread": registers,
                    "spills_per_thread": kernel.n_spills,
                    "shared_memory_bytes_per_block": shared,
                    "warps_per_block": warps,
                    "threads_per_block": threads,
                    "derived_resident_blocks_per_sm": resident_blocks,
                    "derived_occupancy": occupancy,
                    "derived_resource_limits": resource_limits,
                    "occupancy_method": (
                        "resource-limit upper bound from cudaDeviceGetAttribute and "
                        "Triton compiler metadata; register allocation granularity and "
                        "scheduler/barrier limits are not available without Nsight counters"
                    ),
                })
    return result


def profile_operation(name: str, operation, triton_functions: list[JITFunction],
                      limits: dict[str, int]) -> dict:
    for _ in range(3):
        synchronize(operation())
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        value = operation()
        synchronize(value)
    # Raw DeviceType.CUDA events are actual kernels/copies.  key_averages()
    # also attributes child CUDA time to CPU operators (e.g. aten::mm), which
    # would double-count launches.
    aggregated = {}
    raw_cuda_events = []
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        raw_cuda_events.append(event)
        row = aggregated.setdefault(event.name, {
            "name": event.name, "count": 0, "self_cuda_time_us": 0.0,
            "cuda_time_us": 0.0, "flops_reported": 0,
        })
        row["count"] += 1
        row["self_cuda_time_us"] += float(event.self_cuda_time_total or 0.0)
        row["cuda_time_us"] += float(event.cuda_time_total or 0.0)
    events = list(aggregated.values())
    events.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    total = sum(row["self_cuda_time_us"] for row in events)
    for event in events:
        event["self_cuda_share"] = event["self_cuda_time_us"] / total if total else None
    return {
        "name": name,
        "scope": "one eager operation after three warmups",
        "profiler": "torch.profiler",
        "self_cuda_total_us": total,
        "cuda_kernel_launch_count": sum(
            1 for event in raw_cuda_events if not event.name.startswith("Memcpy")
        ),
        "cuda_memcpy_count": sum(
            1 for event in raw_cuda_events if event.name.startswith("Memcpy")
        ),
        "events": events,
        "triton_compiled_kernels": compiled_kernels(triton_functions, limits),
        "hardware_counters": {
            "dram_bytes": None,
            "dram_bandwidth": None,
            "l2_bytes": None,
            "sfu_utilization": None,
            "sm_utilization": None,
            "reason": (
                "Nsight Compute 2025.3.1 is installed, but the host kernel sets "
                "RmProfilingAdminOnly=1 and this container lacks CAP_SYS_ADMIN "
                "(ERR_NVGPUCTRPERM); no hardware counter is fabricated."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(1729)
    props = torch.cuda.get_device_properties(0)
    limits = device_limits(props)
    b, l, d, modes = 1, 2048, 128, 64

    samu_params = make_samu_parameters(d, modes, "cuda", torch.bfloat16)
    packed_samu = samu.pack_samu_parameters(samu_params)
    samu_x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
    samu_prefill = lambda: samu.samu_triton_chunked(
        samu_x, samu_params, 32, packed_samu, return_cache=True
    )
    samu_token = torch.randn(b, d, device="cuda", dtype=torch.bfloat16)
    samu_state = (torch.randn(b, modes, device="cuda"), torch.randn(b, modes, device="cuda"))
    samu_decode = lambda: samu.samu_triton_decode(samu_token, samu_state, samu_params, packed_samu)

    RGLRU = load_official_rglru(args.rglru_source)
    rg_model = RGLRU(width=d, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    packed_rg = rg.pack_rglru(rg_model)
    rg_x = torch.randn(b, l, d, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(l, device="cuda").unsqueeze(0)
    rg_prefill = lambda: rg.rglru_triton_chunked(rg_x, positions, packed_rg, 32)
    rg_token = torch.randn(b, 1, d, device="cuda", dtype=torch.bfloat16)
    rg_position = torch.ones(b, 1, device="cuda", dtype=torch.long)
    rg_state = torch.randn(b, d, device="cuda")
    rg_decode = lambda: rg.rglru_triton_decode(rg_token, rg_position, packed_rg, rg_state)

    profiles = []
    profiles.append(profile_operation("samu_c32_prefill_B1_L2048", samu_prefill, [
        samu._chunk_summary_kernel, samu._chunk_prefix_kernel, samu._chunk_replay_kernel,
    ], limits))
    profiles.append(profile_operation("rglru_c32_prefill_B1_L2048", rg_prefill, [
        rg._chunk_summary_kernel, rg._chunk_prefix_kernel, rg._chunk_replay_kernel,
    ], limits))
    profiles.append(profile_operation("samu_fused_decode_B1", samu_decode, [samu._decode_kernel], limits))
    profiles.append(profile_operation("rglru_fused_decode_B1", rg_decode, [rg._decode_kernel], limits))
    atomic_json(args.output, {
        "schema_version": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": props.name,
        "device_limits": limits,
        "counter_permission": counter_permission_evidence(),
        "warning": "Launch timings/counts and Triton compiler register/spill/shared metadata are measured. Occupancy is a resource upper bound. Hardware bandwidth/SFU counters are permission-blocked and remain null.",
        "profiles": profiles,
    })


if __name__ == "__main__":
    main()
