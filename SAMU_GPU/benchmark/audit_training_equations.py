"""Numerical audit for the training recurrences used by the full model.

RG-LRU is compared directly with the pinned, unmodified
``recurrentgemma.torch.layers.RGLRU`` implementation.  SAMU and both custom
reverse scans are checked by ``run_small_model_study.correctness_checks``.
No performance measurement is accepted unless this audit passes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from kernels import load_official_rglru
from run_small_model_study import RGLRUMixer, correctness_checks


OFFICIAL_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def official_training_audit(source: Path) -> dict:
    torch.manual_seed(7391)
    width, heads, batch, length = 384, 16, 3, 37
    OfficialRGLRU = load_official_rglru(source)
    official = OfficialRGLRU(
        width=width, num_heads=heads, device="cuda", dtype=torch.float32
    ).train()
    candidate = RGLRUMixer(width, heads).cuda().train()
    candidate.load_state_dict(official.state_dict(), strict=True)

    source_x = torch.randn(
        batch, length, width, device="cuda", dtype=torch.bfloat16
    )
    official_x = source_x.detach().clone()
    candidate_x = source_x.detach().clone()
    positions = torch.arange(length, device="cuda").expand(batch, -1)
    # Directly compare the complete gate, normalization, reset and recurrence
    # forward path.  Reverse-scan gradients are audited independently against
    # explicit eager recurrences in ``correctness_checks`` below.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        official_output, official_cache = official(
            official_x, positions, return_cache=True
        )
        candidate_output, candidate_cache = candidate(
            candidate_x, return_cache=True
        )
    report = {
        "reference": "unmodified recurrentgemma.torch.layers.RGLRU",
        "reference_commit": OFFICIAL_COMMIT,
        "shape": {"batch": batch, "length": length, "width": width, "heads": heads},
        "precision": "BF16 input/output and gate matmuls; FP32 recurrence state",
        "output_max_abs": float(
            (candidate_output.float() - official_output.float()).abs().max()
        ),
        "cache_max_abs": float(
            (candidate_cache - official_cache).abs().max()
        ),
        "gradient_audit": (
            "custom real and complex reverse scans are compared with explicit "
            "eager autograd recurrences in custom_scan_reference_checks"
        ),
    }
    limits = {
        "output_max_abs": 0.02,
        "cache_max_abs": 0.02,
    }
    failures = {
        name: {"measured": report[name], "limit": limit}
        for name, limit in limits.items()
        if report[name] > limit
    }
    report["acceptance_limits"] = limits
    report["passed"] = not failures
    report["failures"] = failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = official_training_audit(args.rglru_source)
    custom_scans = correctness_checks()
    custom_limits = {
        key: (
            0.12 if "samu_chunk" in key and "bfloat16" in key
            else 0.04 if "bfloat16" in key else 3e-4
        )
        for key in custom_scans
    }
    custom_failures = {
        f"{group}.{name}": {"measured": value, "limit": custom_limits[group]}
        for group, values in custom_scans.items()
        for name, value in values.items()
        if isinstance(value, (int, float))
        and "relative" not in name
        and value > custom_limits[group]
    }
    for group, values in custom_scans.items():
        if "samu_chunk" in group and values.get("gradient_max_relative", 0.0) > 0.003:
            custom_failures[f"{group}.gradient_max_relative"] = {
                "measured": values["gradient_max_relative"],
                "limit": 0.003,
            }
    result = {
        "official_rglru_training": official,
        "custom_scan_reference_checks": custom_scans,
        "custom_scan_acceptance_limits": custom_limits,
        "custom_scan_failures": custom_failures,
        "passed": official["passed"] and not custom_failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("training equation audit failed")


if __name__ == "__main__":
    main()
