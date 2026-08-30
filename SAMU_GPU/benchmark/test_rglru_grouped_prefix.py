"""Correctness gate for strengthened RG-LRU grouped affine prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from canonical_rg_lru_reference import canonical_rg_lru_from_logits
from triton_fused_rglru_training import fused_rglru_scan


def metric(reference, actual):
    reference, actual = reference.float(), actual.float()
    difference = actual - reference
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        ).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            reference.flatten(), actual.flatten(), dim=0, eps=1.0e-12
        ).item(),
    }


def evaluate(values, positions, backend, group_size):
    x, gx, ga, a_param, h0 = [
        value.detach().clone().requires_grad_(True) for value in values
    ]
    if backend == "reference":
        output, last = canonical_rg_lru_from_logits(
            x, gx, ga, a_param, segment_pos=positions,
            initial_state=h0,
            production_dtype=x.dtype if x.dtype == torch.bfloat16 else None,
        )
    else:
        output, last = fused_rglru_scan(
            x, gx, ga, a_param, segment_pos=positions, h0=h0,
            chunk_size=16, block_size=64, reset_first=False,
            hierarchical_prefix=True,
            prefix_group_size=group_size if backend == "grouped" else 0,
        )
    # Reuse a deterministic cotangent independent of backend evaluation.
    weight = torch.sin(torch.arange(output.numel(), device="cuda").reshape_as(output))
    last_weight = torch.cos(torch.arange(last.numel(), device="cuda").reshape_as(last))
    ((output.float() * weight.float()).sum() +
     (last.float() * last_weight.float()).sum()).backward()
    return output.detach(), last.detach(), [
        tensor.grad.detach() for tensor in (x, gx, ga, a_param, h0)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(76501)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for chunks, group_size in ((3, 32), (33, 32), (65, 64)):
            # Leave the final chunk partial so padding cannot enter a summary.
            batch, width, length = 2, 129, chunks * 16 - 7
            values = (
                torch.randn(batch, length, width, device="cuda", dtype=dtype) * 0.2,
                torch.randn(batch, length, width, device="cuda", dtype=dtype) * 0.5,
                torch.randn(batch, length, width, device="cuda", dtype=dtype) * 0.5,
                torch.randn(width, device="cuda", dtype=torch.float32) * 0.2 + 1.0,
                torch.randn(batch, width, device="cuda", dtype=torch.float32) * 0.1,
            )
            positions = torch.arange(length, device="cuda", dtype=torch.int32)[None].repeat(batch, 1)
            positions[0, 0] = 3
            positions[0, 5] = 0
            positions[1, length // 2] = 0
            positions[0, 16] = 0
            positions[1, 17] = 0
            expected = evaluate(values, positions, "reference", 0)
            hillis = evaluate(values, positions, "hillis_steele", 0)
            actual = evaluate(values, positions, "grouped", group_size)
            rows.append({
                "dtype": str(dtype), "chunks": chunks,
                "group_size": group_size,
                "output": metric(expected[0], actual[0]),
                "last": metric(expected[1], actual[1]),
                "gradients": [metric(a, b) for a, b in zip(expected[2], actual[2])],
                "hillis_steele": {
                    "output": metric(expected[0], hillis[0]),
                    "last": metric(expected[1], hillis[1]),
                    "gradients": [
                        metric(a, b) for a, b in zip(expected[2], hillis[2])
                    ],
                },
                "comparison_to_hillis_steele": {
                    "output": metric(hillis[0], actual[0]),
                    "last": metric(hillis[1], actual[1]),
                    "gradients": [
                        metric(a, b) for a, b in zip(hillis[2], actual[2])
                    ],
                },
            })
    result = {"schema_version": 1, "rows": rows}
    result["sequential_reference_passed"] = all(
        row["output"]["relative_l2"] <= (2e-4 if "bfloat16" in row["dtype"] else 2e-5)
        and row["last"]["relative_l2"] <= (2e-4 if "bfloat16" in row["dtype"] else 2e-5)
        and max(value["relative_l2"] for value in row["gradients"])
        <= (2e-2 if "bfloat16" in row["dtype"] else 2e-4)
        and min(value["cosine_similarity"] for value in row["gradients"]) >= 0.9999
        for row in rows
    )
    result["regression_to_hillis_steele_passed"] = all(
        row["comparison_to_hillis_steele"]["output"]["relative_l2"] <= 2e-5
        and row["comparison_to_hillis_steele"]["last"]["relative_l2"] <= 2e-5
        and max(value["relative_l2"] for value in
                row["comparison_to_hillis_steele"]["gradients"]) <= 2e-4
        and min(value["cosine_similarity"] for value in
                row["comparison_to_hillis_steele"]["gradients"]) >= 0.99999
        for row in rows
    )
    result["passed"] = (
        result["sequential_reference_passed"]
        and result["regression_to_hillis_steele_passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
