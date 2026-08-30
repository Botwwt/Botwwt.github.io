"""Correctness and repeatability of atomic shared-control reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_training_scan import samu_chunk_scan


def metric(reference, actual):
    a, b = reference.float(), actual.float()
    difference = b - a
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(a).clamp_min(1.0e-12)
        ).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0, eps=1.0e-12
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(76529)
    batch, length, modes = 2, 96, 257
    sources = (
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(batch, length, device="cuda") * 0.1,
        torch.randn(modes, device="cuda") - 1.0,
        torch.randn(modes, device="cuda"),
        torch.randn(batch, length, modes, device="cuda",
                    dtype=torch.bfloat16) * 0.1,
        torch.randn(batch, length, modes, device="cuda",
                    dtype=torch.bfloat16) * 0.1,
    )
    cotangents = (torch.randn_like(sources[4]), torch.randn_like(sources[5]))

    def evaluate(atomic):
        values = [value.detach().clone().requires_grad_(True)
                  for value in sources]
        outputs = samu_chunk_scan(
            *values, chunk_size=32, hierarchical_prefix=True,
            precompute_shared=True, atomic_shared=atomic,
            mode_block_size=128, prefix_group_size=32,
        )
        torch.autograd.backward(outputs, cotangents)
        return tuple(value.detach() for value in outputs), tuple(
            value.grad.detach().clone() for value in values
        )

    reference = evaluate(False)
    repeats = [evaluate(True) for _ in range(4)]
    comparisons = [{
        "outputs": [metric(a, b) for a, b in zip(reference[0], actual[0])],
        "gradients": [metric(a, b) for a, b in zip(reference[1], actual[1])],
    } for actual in repeats]
    run_to_run = [{
        "outputs": [metric(a, b) for a, b in zip(repeats[0][0], actual[0])],
        "gradients": [metric(a, b) for a, b in zip(repeats[0][1], actual[1])],
    } for actual in repeats[1:]]
    result = {
        "schema_version": 1,
        "shape": {"batch": batch, "length": length, "modes": modes,
                  "mode_tiles": 3},
        "reference": "two-stage FP32 shared-control reduction",
        "candidate": "FP32 atomicAdd shared-control reduction",
        "comparisons": comparisons,
        "run_to_run": run_to_run,
    }
    result["passed"] = all(
        max(value["relative_l2"] for value in row["outputs"]) == 0.0
        and max(value["relative_l2"] for value in row["gradients"]) <= 5e-5
        and min(value["cosine_similarity"] for value in row["gradients"]) >= 0.99999
        for row in comparisons
    )
    result["run_to_run_bitwise"] = all(
        max(value["max_abs"] for value in row["outputs"]) == 0.0
        and max(value["max_abs"] for value in row["gradients"]) == 0.0
        for row in run_to_run
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
