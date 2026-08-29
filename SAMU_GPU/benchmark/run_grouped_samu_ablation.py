"""Calibrate and independently time Grouped SAMU-8 against official RG-LRU-16.

This is an architecture ablation, not a no-retraining optimization.  The
comparison mirrors the 16-block gate configuration used by every Griffin
training experiment.  Both sides use the same BF16 projection boundary and
FP32 recurrent cache.  Calibration samples never become reported final rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from kernels import load_official_rglru, make_samu_parameters
from run_equal_kernel_benchmarks import measure, stabilize_gpu
from triton_rglru import pack_rglru, rglru_triton_decode
from triton_samu_grouped import (
    grouped_samu_reference,
    grouped_samu_triton_decode,
    pack_grouped_samu_parameters,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correctness(width: int, packed, seed: int) -> dict:
    torch.manual_seed(seed)
    worst_output = worst_real = worst_imag = 0.0
    cases = []
    for batch, scale in ((1, 2.0**-8), (2, 1.0), (5, 16.0)):
        token = (torch.randn(batch, width, device="cuda") * scale).to(torch.bfloat16)
        state = (
            torch.randn(batch, width // 2, device="cuda") * scale,
            torch.randn(batch, width // 2, device="cuda") * scale,
        )
        expected, expected_cache = grouped_samu_reference(token, state, packed)
        actual, actual_cache = grouped_samu_triton_decode(token, state, packed)
        torch.cuda.synchronize()
        errors = {
            "output_max_abs": float((actual - expected).abs().max()),
            "cache_real_max_abs": float((actual_cache[0] - expected_cache[0]).abs().max()),
            "cache_imag_max_abs": float((actual_cache[1] - expected_cache[1]).abs().max()),
        }
        cases.append({"batch": batch, "input_scale": scale, **errors})
        worst_output = max(worst_output, errors["output_max_abs"])
        worst_real = max(worst_real, errors["cache_real_max_abs"])
        worst_imag = max(worst_imag, errors["cache_imag_max_abs"])
    limits = {"output_max_abs": 2e-3, "cache_real_max_abs": 2e-5, "cache_imag_max_abs": 2e-5}
    maxima = {
        "output_max_abs": worst_output,
        "cache_real_max_abs": worst_real,
        "cache_imag_max_abs": worst_imag,
    }
    return {
        "cases": cases,
        "maxima": maxima,
        "limits": limits,
        "passed": all(maxima[key] <= limit for key, limit in limits.items()),
    }


def timed(operation, warmup_ms: int, rep_ms: int) -> dict:
    timing, compile_seconds = measure(operation, warmup_ms=warmup_ms, rep_ms=rep_ms)
    return {**timing, "compile_seconds": compile_seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--calibration-rep-ms", type=int, default=50)
    parser.add_argument("--final-rep-ms", type=int, default=200)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    source_commit = os.popen(f"git -C {args.rglru_source} rev-parse HEAD").read().strip()
    if source_commit != RGLRU_COMMIT:
        raise RuntimeError(f"official RG-LRU source commit mismatch: {source_commit}")
    stabilization = stabilize_gpu()
    RGLRU = load_official_rglru(args.rglru_source)
    calibration, final_rows, audits, comparisons = [], [], {}, []

    for width in (128, 2048):
        modes = width // 2
        samu_parameters = make_samu_parameters(width, modes, "cuda", torch.bfloat16, seed=5701)
        samu = pack_grouped_samu_parameters(samu_parameters, groups=8)
        torch.manual_seed(5701)
        rg_module = RGLRU(width=width, num_heads=16, device="cuda", dtype=torch.bfloat16).eval()
        rg = pack_rglru(rg_module)
        audits[str(width)] = correctness(width, samu, 8100 + width)
        counts = {
            "samu_learned_parameters": samu.learned_parameter_count,
            "rglru_learned_parameters": sum(
                tensor.numel() for tensor in (rg.gate_weight, rg.gate_bias, rg.softplus_a)
            ),
            "parameter_difference": samu.learned_parameter_count - sum(
                tensor.numel() for tensor in (rg.gate_weight, rg.gate_bias, rg.softplus_a)
            ),
            "samu_projection_macs_per_token": samu.projection_macs_per_token,
            "rglru_projection_macs_per_token": rg.gate_weight.numel(),
            "samu_state_bytes_per_sequence": 2 * modes * 4,
            "rglru_state_bytes_per_sequence": width * 4,
        }
        blocks = (1, 2, 4, 8) if width == 128 else (16, 32, 64, 128)
        for batch in (1, 4, 16, 64, 128):
            torch.manual_seed(9000 + width + batch)
            token = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
            samu_state = (
                torch.randn(batch, modes, device="cuda"),
                torch.randn(batch, modes, device="cuda"),
            )
            rg_state = torch.randn(batch, width, device="cuda")
            positions = torch.ones(batch, 1, device="cuda", dtype=torch.long)
            candidates = []
            for block_m in blocks:
                for warps in (1, 2, 4, 8):
                    row = {
                        "model": "samu_grouped8", "width": width, "batch": batch,
                        "block_m": block_m, "num_warps": warps, **counts,
                    }
                    try:
                        timing = timed(
                            lambda block_m=block_m, warps=warps: grouped_samu_triton_decode(
                                token, samu_state, samu, block_m=block_m, num_warps=warps
                            ),
                            10, args.calibration_rep_ms,
                        )
                        row.update(status="measured", **timing)
                        candidates.append(row)
                    except Exception as error:
                        row.update(status="unsupported", error=repr(error))
                    calibration.append(row)
            for warps in (1, 2, 4, 8):
                row = {
                    "model": "rglru_official16", "width": width, "batch": batch,
                    "block_m": None, "num_warps": warps, **counts,
                }
                try:
                    timing = timed(
                        lambda warps=warps: rglru_triton_decode(
                            token[:, None], positions, rg, rg_state, num_warps=warps
                        ),
                        10, args.calibration_rep_ms,
                    )
                    row.update(status="measured", **timing)
                    candidates.append(row)
                except Exception as error:
                    row.update(status="unsupported", error=repr(error))
                calibration.append(row)

            selected = {}
            for model in ("samu_grouped8", "rglru_official16"):
                measured = [row for row in candidates if row["model"] == model]
                selected[model] = min(measured, key=lambda row: row["median_ms"])
            operations = {
                "samu_grouped8": lambda: grouped_samu_triton_decode(
                    token, samu_state, samu,
                    block_m=selected["samu_grouped8"]["block_m"],
                    num_warps=selected["samu_grouped8"]["num_warps"],
                ),
                "rglru_official16": lambda: rglru_triton_decode(
                    token[:, None], positions, rg, rg_state,
                    num_warps=selected["rglru_official16"]["num_warps"],
                ),
            }
            measured_by_model = {name: [] for name in operations}
            for order_name, order in (("AB", ("samu_grouped8", "rglru_official16")),
                                      ("BA", ("rglru_official16", "samu_grouped8"))):
                stabilize_gpu(seconds=0.5)
                for model in order:
                    timing = timed(operations[model], 30, args.final_rep_ms)
                    final_rows.append({
                        "model": model, "width": width, "batch": batch,
                        "measurement_order": order_name,
                        "selected_block_m": selected[model]["block_m"],
                        "selected_num_warps": selected[model]["num_warps"],
                        **counts, **timing,
                    })
                    measured_by_model[model].append(timing["median_ms"])
            samu_ms = statistics.mean(measured_by_model["samu_grouped8"])
            rg_ms = statistics.mean(measured_by_model["rglru_official16"])
            comparisons.append({
                "width": width, "batch": batch,
                "samu_order_balanced_median_ms": samu_ms,
                "rglru_order_balanced_median_ms": rg_ms,
                "rglru_over_samu": rg_ms / samu_ms,
                "winner": "samu_grouped8" if samu_ms < rg_ms else "rglru_official16",
                **counts,
            })
        del rg_module

    props = torch.cuda.get_device_properties(0)
    script_dir = Path(__file__).parent
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": "architecture_ablation_requires_retraining_and_quality_validation",
        "fairness": {
            "official_rglru_configuration": "16 block-diagonal gate groups, as stated in Griffin Section 4.1",
            "samu_configuration": "8 block-diagonal complex-write groups plus two token-shared dense scalar selectors",
            "precision": "BF16 projection boundary and returned activation; FP32 recurrent cache",
            "state_bytes": "equal at fixed width",
            "calibration_separation": "all candidates retained; final AB/BA timings are independent",
        },
        "environment": {
            "hostname": platform.node(), "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "rglru_commit": source_commit,
            "source_sha256": {
                "grouped_samu": digest(script_dir / "triton_samu_grouped.py"),
                "rglru": digest(script_dir / "triton_rglru.py"),
            },
        },
        "gpu_stabilization": stabilization,
        "correctness": audits,
        "calibration_rows": calibration,
        "final_rows": final_rows,
        "comparisons": comparisons,
    }
    if not all(audit["passed"] for audit in audits.values()):
        atomic_json(args.output, result)
        raise RuntimeError("grouped SAMU correctness failed; timings are quarantined")
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
