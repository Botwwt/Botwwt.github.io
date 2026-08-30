"""End-to-end controller -> recurrence gradient gate for fused SAMU writes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from run_small_model_study import SAMUMixer, set_seed


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


def evaluate(module, x, cotangent):
    module.zero_grad(set_to_none=True)
    value = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = module(value)
    (output.float() * cotangent.float()).sum().backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in module.named_parameters()}
    gradients["input"] = value.grad.detach().clone()
    return output.detach(), gradients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_seed(41109)
    rows = []
    for reference_backend, candidate_backend in (
        ("shared_sfu_chunk16", "fused_write_shared_sfu_chunk16"),
        ("hierarchical_chunk16", "fused_write_hierarchical_chunk16"),
    ):
        width, batch, length = 258, 2, 64
        reference = SAMUMixer(width).cuda().train()
        candidate = copy.deepcopy(reference)
        for module in (reference, candidate):
            module.set_tiled_training_block_size(256)
            module.set_controller_projection_dtype("fp32_fused_coords")
            with torch.no_grad():
                module.phase_amplitude.fill_(0.7)
                module.radial_amplitude.fill_(0.5)
        reference.set_scan_backend(reference_backend)
        candidate.set_scan_backend(candidate_backend)
        x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        cotangent = torch.randn_like(x)
        reference_output, reference_gradients = evaluate(reference, x, cotangent)
        candidate_output, candidate_gradients = evaluate(candidate, x, cotangent)
        rows.append({
            "reference_backend": reference_backend,
            "candidate_backend": candidate_backend,
            "output": metric(reference_output, candidate_output),
            "gradients": {
                name: metric(reference_gradients[name], candidate_gradients[name])
                for name in reference_gradients
            },
        })
    result = {"schema_version": 1, "rows": rows}
    result["passed"] = all(
        row["output"]["relative_l2"] <= 2.0e-4
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
