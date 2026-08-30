"""Correctness of per-chunk two-stage nu/theta gradient reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from triton_training_scan import samu_chunk_scan


def metric(reference, candidate):
    reference, candidate = reference.detach().float().flatten(), candidate.detach().float().flatten()
    difference = candidate - reference
    norm = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / norm),
        "cosine_similarity": float(F.cosine_similarity(reference, candidate, dim=0)),
    }


def run(dtype):
    torch.manual_seed(91511)
    batch, length, modes = 2, 64, 257
    sources = (
        torch.randn(batch, length, device="cuda", dtype=torch.float32) * 0.1,
        torch.randn(batch, length, device="cuda", dtype=torch.float32) * 0.1,
        torch.randn(modes, device="cuda", dtype=torch.float32) - 1.0,
        torch.randn(modes, device="cuda", dtype=torch.float32),
        torch.randn(batch, length, modes, device="cuda", dtype=dtype) * 0.1,
        torch.randn(batch, length, modes, device="cuda", dtype=dtype) * 0.1,
    )
    cotangent_r = torch.randn_like(sources[4])
    cotangent_i = torch.randn_like(sources[5])

    def evaluate(two_stage):
        values = [source.detach().clone().requires_grad_(True) for source in sources]
        out_r, out_i = samu_chunk_scan(
            *values, 16, False, True, False, False, two_stage
        )
        loss = ((out_r.float() * cotangent_r.float()).sum()
                + (out_i.float() * cotangent_i.float()).sum())
        loss.backward()
        return (out_r, out_i), [value.grad for value in values]

    reference_output, reference_gradients = evaluate(False)
    candidate_output, candidate_gradients = evaluate(True)
    output_metrics = [metric(a, b) for a, b in zip(reference_output, candidate_output)]
    gradient_metrics = [metric(a, b) for a, b in zip(reference_gradients, candidate_gradients)]
    return {
        "dtype": str(dtype),
        "shape": {"batch": batch, "length": length, "modes": modes},
        "output": output_metrics,
        "gradients": gradient_metrics,
        "max_gradient_relative_l2": max(row["relative_l2"] for row in gradient_metrics),
        "min_gradient_cosine": min(row["cosine_similarity"] for row in gradient_metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [run(torch.float32), run(torch.bfloat16)]
    result = {
        "schema_version": 1,
        "reference": "existing atomic nu/theta accumulation",
        "candidate": "per-chunk partial plus deterministic-order torch FP32 reduction",
        "rows": rows,
    }
    result["passed"] = all(
        row["max_gradient_relative_l2"] <= 5e-5 and row["min_gradient_cosine"] >= 0.99999
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
