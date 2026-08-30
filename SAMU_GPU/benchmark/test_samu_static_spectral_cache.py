"""Correctness of the O(M) static spectrum cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from triton_training_scan import samu_chunk_scan


def metric(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    difference = b - a
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) /
                             torch.linalg.vector_norm(a).clamp_min(1e-30)),
        "cosine_similarity": float(F.cosine_similarity(a, b, dim=0)),
    }


def run(dtype):
    torch.manual_seed(91541)
    batch, length, modes = 2, 64, 257
    sources = (
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(modes, device="cuda") - 1.0,
        torch.randn(modes, device="cuda"),
        torch.randn(batch, length, modes, device="cuda", dtype=dtype) * 0.1,
        torch.randn(batch, length, modes, device="cuda", dtype=dtype) * 0.1,
    )
    cotangents = (torch.randn_like(sources[4]), torch.randn_like(sources[5]))

    def evaluate(cached):
        values = [value.detach().clone().requires_grad_(True) for value in sources]
        output = samu_chunk_scan(
            *values, 16, True, False, False, False, False, False, cached, 128
        )
        torch.autograd.backward(output, cotangents)
        return output, [value.grad for value in values]

    reference_output, reference_gradients = evaluate(False)
    candidate_output, candidate_gradients = evaluate(True)
    return {
        "dtype": str(dtype),
        "outputs": [metric(a, b) for a, b in zip(reference_output, candidate_output)],
        "gradients": [metric(a, b) for a, b in zip(reference_gradients, candidate_gradients)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [run(torch.float32), run(torch.bfloat16)]
    result = {"schema_version": 1,
              "reference": "spectrum evaluated independently in every chunk program",
              "candidate": "one O(M) nu/theta/cos/sin cache launch",
              "rows": rows}
    result["passed"] = all(
        max(value["relative_l2"] for value in row["outputs"]) <= 1e-6
        and max(value["relative_l2"] for value in row["gradients"]) <= 1e-5
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
