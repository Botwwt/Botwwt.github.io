"""Long-sequence numerical stability of raw-D compressed chunk summaries."""

from __future__ import annotations

import argparse
import json
import math
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


def run(length, modes, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    eta = torch.randn(1, length, generator=generator, device="cuda") * 0.02
    delta = torch.randn(1, length, generator=generator, device="cuda") * 0.02
    radius_squared = torch.rand(modes, generator=generator, device="cuda").clamp_min(1e-6)
    nu = -0.5 * torch.log(radius_squared)
    theta = (2 * math.pi * torch.rand(modes, generator=generator, device="cuda")).clamp_min(1e-6)
    nu_log, theta_log = torch.log(nu), torch.log(theta)
    br = torch.randn(1, length, modes, generator=generator, device="cuda",
                     dtype=torch.bfloat16) * 0.05
    bi = torch.randn(1, length, modes, generator=generator, device="cuda",
                     dtype=torch.bfloat16) * 0.05
    sources = (eta, delta, nu_log, theta_log, br, bi)
    cotangent_r = torch.randn_like(br)
    cotangent_i = torch.randn_like(bi)

    def evaluate(compressed):
        values = [value.detach().clone().requires_grad_(True) for value in sources]
        output = samu_chunk_scan(
            *values, 16, True, False, compressed, False, False, False
        )
        loss = ((output[0].float() * cotangent_r.float()).sum()
                + (output[1].float() * cotangent_i.float()).sum())
        loss.backward()
        return output, [value.grad for value in values]

    reference_output, reference_gradients = evaluate(False)
    candidate_output, candidate_gradients = evaluate(True)
    chunks = delta.reshape(1, length // 16, 16)
    chunk_d = chunks.sum(dim=-1)
    phase = theta[None, None, :] * 16 + chunk_d[..., None]
    return {
        "shape": {"batch": 1, "length": length, "modes": modes},
        "phase_diagnostics": {
            "max_abs_token_prefix_D": float(delta.cumsum(dim=1).abs().max()),
            "max_abs_chunk_D": float(chunk_d.abs().max()),
            "max_abs_reconstructed_chunk_phase": float(phase.abs().max()),
            "phase_wrapping": "none; raw D is used",
        },
        "outputs": [metric(a, b) for a, b in zip(reference_output, candidate_output)],
        "gradients": [metric(a, b) for a, b in zip(reference_gradients, candidate_gradients)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [run(65536, 128, 93501), run(131072, 64, 93502)]
    result = {"schema_version": 1,
              "reference": "materialized-P hierarchical prefix",
              "candidate": "compressed G/D then transient-P hierarchical prefix",
              "rows": rows}
    result["passed"] = all(
        max(value["relative_l2"] for value in row["outputs"]) <= 2e-4
        and max(value["relative_l2"] for value in row["gradients"]) <= 2e-4
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
