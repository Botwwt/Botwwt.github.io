"""H800 audit of the SAMU-specific GPU adaptation candidates.

The timed unit is the pair that actually appears in a decoder layer:
RMSNorm followed by one recurrent decode update.  RG-LRU uses the official
16-block gate layout from Griffin.  SAMU variants are:

* grouped8_dense: equal temporal parameter count to RG-LRU-16 (+4 scalars);
* grouped16_dense: natural 16-way write sharding, dense shared controllers;
* grouped16_direct: the two shared controls are carried in two normalized
  channels.  This changes the architecture and therefore requires retraining
  and a quality study; it is never presented as a trained-model result.

Calibration and final AB/BA runs are kept separate.  BF16 activations and FP32
recurrent state are used throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from kernels import load_official_rglru, make_samu_parameters
from profile_equal_kernels import compiled_kernels, device_limits
from run_equal_kernel_benchmarks import measure, stabilize_gpu
import triton_rglru as rg_impl
from triton_rglru import pack_rglru, rglru_triton_decode
import triton_samu_grouped as samu_impl
from triton_samu_grouped import (
    grouped_samu_reference,
    grouped_samu_rmsnorm_decode,
    pack_grouped_samu_parameters,
    triton_rmsnorm,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def packed_bytes_samu(packed) -> int:
    tensors = [
        packed.wr, packed.wi, packed.phase_direction, packed.radial_direction,
        packed.base.nu, packed.base.cos_theta, packed.base.sin_theta, packed.base.gamma,
    ]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def packed_bytes_rg(packed) -> int:
    tensors = [packed.gate_weight, packed.gate_bias, packed.softplus_a]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def timing(operation, warmup_ms: int, rep_ms: int) -> dict:
    measured, compile_seconds = measure(operation, warmup_ms=warmup_ms, rep_ms=rep_ms)
    return {**measured, "compile_seconds": compile_seconds}


def correctness(width: int, norm_weight, packed, direct: bool) -> dict:
    maxima = {"output_max_abs": 0.0, "cache_real_max_abs": 0.0, "cache_imag_max_abs": 0.0}
    cases = []
    for batch, scale in ((1, 2.0**-8), (4, 1.0), (17, 16.0)):
        torch.manual_seed(7400 + width + batch)
        x = (torch.randn(batch, width, device="cuda") * scale).to(torch.bfloat16)
        state = (
            torch.randn(batch, width // 2, device="cuda") * scale,
            torch.randn(batch, width // 2, device="cuda") * scale,
        )
        # The separate Triton RMSNorm is the precision-policy reference.  The
        # candidate may change reduction scheduling when the selectors fuse,
        # so one BF16 output ULP is allowed while the FP32 cache remains tight.
        normalized = triton_rmsnorm(x, norm_weight)
        expected, expected_cache = grouped_samu_reference(
            normalized, state, packed, direct_control=direct
        )
        actual, actual_cache = grouped_samu_rmsnorm_decode(
            x, norm_weight, state, packed, block_m=32, num_warps=2,
            direct_control=direct,
        )
        torch.cuda.synchronize()
        row = {
            "batch": batch,
            "input_scale": scale,
            "output_max_abs": float((actual - expected).abs().max()),
            "cache_real_max_abs": float((actual_cache[0] - expected_cache[0]).abs().max()),
            "cache_imag_max_abs": float((actual_cache[1] - expected_cache[1]).abs().max()),
        }
        cases.append(row)
        for key in maxima:
            maxima[key] = max(maxima[key], row[key])
    limits = {"output_max_abs": 0.125, "cache_real_max_abs": 0.02, "cache_imag_max_abs": 0.02}
    return {
        "reference": "separate Triton RMSNorm plus PyTorch grouped-SAMU equation",
        "cases": cases,
        "maxima": maxima,
        "limits": limits,
        "passed": all(maxima[key] <= limit for key, limit in limits.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--calibration-rep-ms", type=int, default=50)
    parser.add_argument("--final-rep-ms", type=int, default=250)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    source_commit = command_output(["git", "-C", str(args.rglru_source), "rev-parse", "HEAD"])
    if source_commit != RGLRU_COMMIT:
        raise RuntimeError(f"official RG-LRU commit mismatch: {source_commit}")
    stabilization = stabilize_gpu()
    width, modes = 2048, 1024
    parameters = make_samu_parameters(width, modes, "cuda", torch.bfloat16, seed=5701)
    samu8 = pack_grouped_samu_parameters(parameters, groups=8)
    samu16 = pack_grouped_samu_parameters(parameters, groups=16)
    RGLRU = load_official_rglru(args.rglru_source)
    torch.manual_seed(5701)
    rg_module = RGLRU(width=width, num_heads=16, device="cuda", dtype=torch.bfloat16).eval()
    rg = pack_rglru(rg_module)
    norm_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    rg_parameter_count = sum(tensor.numel() for tensor in (rg.gate_weight, rg.gate_bias, rg.softplus_a))
    counts = {
        "grouped8_dense": {
            "learned_temporal_parameters": samu8.learned_parameter_count,
            "projection_macs_per_token": samu8.projection_macs_per_token,
            "packed_parameter_bytes": packed_bytes_samu(samu8),
        },
        "grouped16_dense": {
            "learned_temporal_parameters": samu16.learned_parameter_count,
            "projection_macs_per_token": samu16.projection_macs_per_token,
            "packed_parameter_bytes": packed_bytes_samu(samu16),
        },
        "grouped16_direct": {
            "learned_temporal_parameters": samu16.learned_parameter_count - 2 * width,
            "projection_macs_per_token": samu16.projection_macs_per_token - 2 * width,
            "packed_parameter_bytes": packed_bytes_samu(samu16) - 2 * width * 2,
        },
        "rglru_official16": {
            "learned_temporal_parameters": rg_parameter_count,
            "projection_macs_per_token": rg.gate_weight.numel(),
            "packed_parameter_bytes": packed_bytes_rg(rg),
        },
    }
    for value in counts.values():
        value["state_bytes_per_sequence"] = width * 4
        value["norm_weight_bytes"] = width * 2
        value["expected_launches"] = 2

    audits = {
        "grouped8_dense": correctness(width, norm_weight, samu8, False),
        "grouped16_dense": correctness(width, norm_weight, samu16, False),
        "grouped16_direct": correctness(width, norm_weight, samu16, True),
    }
    calibration, final_rows, comparisons = [], [], []
    variants = {
        "grouped8_dense": (samu8, False),
        "grouped16_dense": (samu16, False),
        "grouped16_direct": (samu16, True),
    }
    candidate_configs = ((8, 2), (16, 2), (16, 4), (16, 8),
                         (32, 1), (32, 2), (32, 4), (64, 2))
    for batch in (1, 4, 16, 64, 128):
        torch.manual_seed(9100 + batch)
        token = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
        samu_state = (
            torch.randn(batch, modes, device="cuda"),
            torch.randn(batch, modes, device="cuda"),
        )
        rg_state = torch.randn(batch, width, device="cuda")
        positions = torch.ones(batch, 1, device="cuda", dtype=torch.long)
        candidates = []
        for name, (packed, direct) in variants.items():
            for block_m, warps in candidate_configs:
                if block_m > packed.group_modes:
                    continue
                row = {
                    "model": name, "batch": batch,
                    "block_m": block_m, "num_warps": warps, **counts[name],
                }
                try:
                    result = timing(
                        lambda packed=packed, direct=direct, block_m=block_m, warps=warps:
                        grouped_samu_rmsnorm_decode(
                            token, norm_weight, samu_state, packed,
                            block_m=block_m, num_warps=warps,
                            direct_control=direct,
                        ),
                        10, args.calibration_rep_ms,
                    )
                    row.update(status="measured", **result)
                    candidates.append(row)
                except Exception as error:
                    row.update(status="unsupported", error=repr(error))
                calibration.append(row)
        for warps in (1, 2, 4, 8):
            row = {
                "model": "rglru_official16", "batch": batch,
                "block_m": None, "num_warps": warps, **counts["rglru_official16"],
            }
            try:
                result = timing(
                    lambda warps=warps: rglru_triton_decode(
                        triton_rmsnorm(token, norm_weight)[:, None], positions,
                        rg, rg_state, num_warps=warps,
                    ),
                    10, args.calibration_rep_ms,
                )
                row.update(status="measured", **result)
                candidates.append(row)
            except Exception as error:
                row.update(status="unsupported", error=repr(error))
            calibration.append(row)

        selected = {}
        for name in (*variants.keys(), "rglru_official16"):
            selected[name] = min(
                (row for row in candidates if row["model"] == name),
                key=lambda row: row["median_ms"],
            )
        operations = {}
        for name, (packed, direct) in variants.items():
            config = selected[name]
            operations[name] = (
                lambda packed=packed, direct=direct, config=config:
                grouped_samu_rmsnorm_decode(
                    token, norm_weight, samu_state, packed,
                    block_m=config["block_m"], num_warps=config["num_warps"],
                    direct_control=direct,
                )
            )
        rg_config = selected["rglru_official16"]
        operations["rglru_official16"] = lambda: rglru_triton_decode(
            triton_rmsnorm(token, norm_weight)[:, None], positions,
            rg, rg_state, num_warps=rg_config["num_warps"],
        )
        medians = {name: [] for name in operations}
        orders = (
            ("forward", tuple(operations.keys())),
            ("reverse", tuple(reversed(tuple(operations.keys())))),
        )
        for order_name, order in orders:
            stabilize_gpu(seconds=0.5)
            for name in order:
                result = timing(operations[name], 30, args.final_rep_ms)
                medians[name].append(result["median_ms"])
                final_rows.append({
                    "model": name, "batch": batch, "measurement_order": order_name,
                    "selected_block_m": selected[name]["block_m"],
                    "selected_num_warps": selected[name]["num_warps"],
                    **counts[name], **result,
                })
        rg_ms = statistics.mean(medians["rglru_official16"])
        for name in variants:
            samu_ms = statistics.mean(medians[name])
            comparisons.append({
                "model": name, "batch": batch,
                "samu_order_balanced_median_ms": samu_ms,
                "rglru_order_balanced_median_ms": rg_ms,
                "rglru_over_samu": rg_ms / samu_ms,
                "winner": name if samu_ms < rg_ms else "rglru_official16",
                **counts[name],
            })

    props = torch.cuda.get_device_properties(0)
    limits = device_limits(props)
    compiler_resources = compiled_kernels([
        samu_impl._rmsnorm_controller_kernel,
        samu_impl._grouped_decode_from_control_kernel,
        samu_impl._rmsnorm_kernel,
        rg_impl._decode_kernel,
    ], limits)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "hostname": platform.node(), "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "rglru_commit": source_commit,
        },
        "scope": "RMSNorm plus one recurrent decode update at D=2048",
        "status": {
            "grouped8_dense": "parameter-matched architecture ablation; retraining required",
            "grouped16_dense": "natural sharding architecture ablation; retraining required",
            "grouped16_direct": "GPU adaptation candidate; retraining and quality validation required",
            "rglru_official16": "official equation and official 16-block gate structure",
        },
        "counts": counts,
        "source_derived_special_functions_per_token": {
            "grouped16_direct": {
                "shared_tanh": 2,
                "mode_exp": modes,
                "sqrt_or_rsqrt": 1,
                "note": "two shared controls; exp(c) is scalar but repeated once per mode tile in the current kernel",
            },
            "rglru_official16": {
                "sigmoid": 2 * width,
                "exp": 2 * width,
                "sqrt": width,
                "rsqrt": 1,
                "note": "counts follow the pinned official BF16 equation implemented in triton_rglru.py",
            },
        },
        "gpu_stabilization": stabilization,
        "correctness": audits,
        "calibration_rows": calibration,
        "final_rows": final_rows,
        "comparisons": comparisons,
        "compiler_resources": compiler_resources,
        "hardware_counter_policy": {
            "dram_l2_sfu_achieved_occupancy": None,
            "reason": "Nsight counters are blocked by host RmProfilingAdminOnly=1 (ERR_NVGPUCTRPERM); no values are inferred or fabricated",
        },
    }
    if not all(audit["passed"] for audit in audits.values()):
        atomic_json(args.output, result)
        raise RuntimeError("correctness failed; timing rows quarantined")
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
