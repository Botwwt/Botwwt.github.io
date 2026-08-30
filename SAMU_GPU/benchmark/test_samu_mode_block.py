"""Correctness of SAMU complex mode-tile width candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from triton_training_scan import samu_chunk_scan


def metric(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    d = b - a
    return {"max_abs": float(d.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(d) /
                                 torch.linalg.vector_norm(a).clamp_min(1e-30)),
            "cosine_similarity": float(F.cosine_similarity(a, b, dim=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(91531)
    batch, length, modes = 2, 64, 257
    sources = (
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(modes, device="cuda") - 1,
        torch.randn(modes, device="cuda"),
        torch.randn(batch, length, modes, device="cuda", dtype=torch.bfloat16) * 0.1,
        torch.randn(batch, length, modes, device="cuda", dtype=torch.bfloat16) * 0.1,
    )
    cotangents = (torch.randn_like(sources[4]), torch.randn_like(sources[5]))

    def evaluate(block):
        values = [value.detach().clone().requires_grad_(True) for value in sources]
        output = samu_chunk_scan(
            *values, 16, False, True, False, False, False, False, False, block
        )
        torch.autograd.backward(output, cotangents)
        return output, [value.grad for value in values]

    reference_output, reference_gradients = evaluate(128)
    rows = []
    for block in (32, 64, 256):
        output, gradients = evaluate(block)
        rows.append({"mode_block": block,
                     "outputs": [metric(a, b) for a, b in zip(reference_output, output)],
                     "gradients": [metric(a, b) for a, b in zip(reference_gradients, gradients)]})
    result = {"schema_version": 1, "reference_mode_block": 128, "rows": rows}
    result["passed"] = all(
        max(x["relative_l2"] for x in row["outputs"]) <= 1e-5
        and max(x["relative_l2"] for x in row["gradients"]) <= 1e-5
        and min(x["cosine_similarity"] for x in row["gradients"]) >= 0.99999
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
