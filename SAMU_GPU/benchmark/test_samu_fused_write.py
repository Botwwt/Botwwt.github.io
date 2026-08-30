"""Correctness gate for SAMU's on-the-fly normalized-write chunk path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_training_scan import samu_chunk_scan


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a, b = reference.float(), candidate.float()
    difference = b - a
    denominator = torch.linalg.vector_norm(a).clamp_min(1.0e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / denominator).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0, eps=1.0e-12
        ).item(),
    }


def evaluate(sources, cotangents, *, fused: bool, hierarchical: bool,
             precompute_shared: bool):
    eta, delta, nu_log, theta_log, raw_x = [
        value.detach().clone().requires_grad_(True) for value in sources
    ]
    modes = nu_log.numel()
    if fused:
        empty = raw_x.new_empty((0,))
        outputs = samu_chunk_scan(
            eta, delta, nu_log, theta_log, empty, empty,
            chunk_size=16,
            hierarchical_prefix=hierarchical,
            precompute_shared=precompute_shared,
            mode_block_size=256,
            raw_x=raw_x,
            fused_write=True,
        )
    else:
        nu = torch.exp(nu_log)
        gamma = torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8
        write_r = (raw_x[..., :modes].float() * gamma).to(raw_x.dtype)
        write_i = (raw_x[..., modes:].float() * gamma).to(raw_x.dtype)
        outputs = samu_chunk_scan(
            eta, delta, nu_log, theta_log, write_r, write_i,
            chunk_size=16,
            hierarchical_prefix=hierarchical,
            precompute_shared=precompute_shared,
            mode_block_size=256,
        )
    torch.autograd.backward(outputs, cotangents)
    return outputs, [eta.grad, delta.grad, nu_log.grad, theta_log.grad, raw_x.grad]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(41103)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        batch, length, modes = 2, 64, 129
        sources = (
            (0.08 * torch.randn(batch, length, device="cuda")).requires_grad_(),
            (0.08 * torch.randn(batch, length, device="cuda")).requires_grad_(),
            torch.linspace(-2.0, 0.3, modes, device="cuda").requires_grad_(),
            torch.linspace(-1.5, 1.0, modes, device="cuda").requires_grad_(),
            (0.2 * torch.randn(batch, length, 2 * modes, device="cuda", dtype=dtype)).requires_grad_(),
        )
        cotangents = tuple(
            torch.randn(batch, length, modes, device="cuda", dtype=dtype)
            for _ in range(2)
        )
        for hierarchical, shared in ((False, True), (True, False)):
            reference_output, reference_gradients = evaluate(
                sources, cotangents, fused=False,
                hierarchical=hierarchical, precompute_shared=shared,
            )
            candidate_output, candidate_gradients = evaluate(
                sources, cotangents, fused=True,
                hierarchical=hierarchical, precompute_shared=shared,
            )
            rows.append({
                "dtype": str(dtype),
                "hierarchical_prefix": hierarchical,
                "precompute_shared": shared,
                "outputs": [
                    metric(a, b) for a, b in zip(reference_output, candidate_output)
                ],
                "gradients": {
                    name: metric(a, b)
                    for name, a, b in zip(
                        ("eta", "delta", "nu_log", "theta_log", "raw_x"),
                        reference_gradients, candidate_gradients,
                    )
                },
            })
    result = {
        "schema_version": 1,
        "reference": "materialized canonical gamma*x write plus exact chunk scan",
        "candidate": "register-generated gamma*x with fused raw-x and nu backward",
        "rows": rows,
    }
    result["passed"] = all(
        max(value["relative_l2"] for value in row["outputs"]) <= 2.0e-4
        and max(value["relative_l2"] for value in row["gradients"].values()) <= 2.0e-4
        and min(value["cosine_similarity"] for value in row["gradients"].values()) >= 0.99999
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
