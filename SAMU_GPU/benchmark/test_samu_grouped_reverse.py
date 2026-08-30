"""Correctness gate for two-chunk reverse replay and spectral accumulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_training_scan import samu_chunk_scan


def metric(a, b):
    a, b = a.float(), b.float()
    difference = b - a
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) /
                        torch.linalg.vector_norm(a).clamp_min(1.0e-12)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0, eps=1.0e-12
        ).item(),
    }


def evaluate(sources, cotangents, *, group, fused_write):
    values = [value.detach().clone().requires_grad_(True) for value in sources]
    eta, delta, nu_log, theta_log = values[:4]
    if fused_write:
        raw_x = values[4]
        empty = raw_x.new_empty((0,))
        output = samu_chunk_scan(
            eta, delta, nu_log, theta_log, empty, empty,
            chunk_size=16, hierarchical_prefix=True,
            mode_block_size=256, raw_x=raw_x, fused_write=True,
            backward_chunk_group=group,
        )
    else:
        br, bi = values[4:]
        output = samu_chunk_scan(
            eta, delta, nu_log, theta_log, br, bi,
            chunk_size=16, hierarchical_prefix=True,
            mode_block_size=256, backward_chunk_group=group,
        )
    torch.autograd.backward(output, cotangents)
    return output, [value.grad for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(76411)
    rows = []
    batch, length, modes = 2, 64, 129
    for dtype in (torch.float32, torch.bfloat16):
        common = [
            0.08 * torch.randn(batch, length, device="cuda"),
            0.08 * torch.randn(batch, length, device="cuda"),
            torch.linspace(-2.0, 0.3, modes, device="cuda"),
            torch.linspace(-1.5, 1.0, modes, device="cuda"),
        ]
        cotangents = tuple(
            torch.randn(batch, length, modes, device="cuda", dtype=dtype)
            for _ in range(2)
        )
        for fused_write in (False, True):
            if fused_write:
                tail = [0.2 * torch.randn(
                    batch, length, 2 * modes, device="cuda", dtype=dtype
                )]
            else:
                tail = [
                    0.2 * torch.randn(
                        batch, length, modes, device="cuda", dtype=dtype
                    ) for _ in range(2)
                ]
            sources = common + tail
            reference = evaluate(
                sources, cotangents, group=1, fused_write=fused_write
            )
            candidate = evaluate(
                sources, cotangents, group=2, fused_write=fused_write
            )
            rows.append({
                "dtype": str(dtype), "fused_write": fused_write,
                "outputs": [metric(a, b) for a, b in zip(reference[0], candidate[0])],
                "gradients": [metric(a, b) for a, b in zip(reference[1], candidate[1])],
            })
    result = {"schema_version": 1, "rows": rows}
    result["passed"] = all(
        max(value["relative_l2"] for value in row["outputs"]) == 0.0
        and max(value["relative_l2"] for value in row["gradients"]) <= 2.0e-5
        and min(value["cosine_similarity"] for value in row["gradients"]) >= 0.99999
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
