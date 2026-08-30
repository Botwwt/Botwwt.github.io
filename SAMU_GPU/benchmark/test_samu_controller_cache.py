"""Correctness gates for save-p/r and full-controller recompute caches."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from run_small_model_study import SAMUMixer


def metric(reference, candidate):
    reference, candidate = reference.detach().float().flatten(), candidate.detach().float().flatten()
    difference = candidate - reference
    norm = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).tiny)
    if float(torch.linalg.vector_norm(reference)) == 0.0 and float(torch.linalg.vector_norm(candidate)) == 0.0:
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(reference, candidate, dim=0))
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / norm),
        "cosine_similarity": cosine,
    }


def run(dtype):
    torch.manual_seed(66521)
    batch, length, width = 2, 64, 512
    base = SAMUMixer(width).cuda().train()
    base.set_scan_backend("shared_sfu_chunk16")
    with torch.no_grad():
        base.phase_amplitude.fill_(0.7)
        base.radial_amplitude.fill_(0.5)
    x_source = torch.randn(batch, length, width, device="cuda", dtype=dtype)
    cotangent = torch.randn_like(x_source)

    def evaluate(controller):
        module = copy.deepcopy(base)
        module.set_controller_projection_dtype(controller)
        x = x_source.detach().clone().requires_grad_(True)
        output = module(x)[0]
        (output.float() * cotangent.float()).sum().backward()
        return {
            "output": output.detach(),
            "grad_x": x.grad.detach(),
            **{f"parameter::{name}": parameter.grad.detach()
               for name, parameter in module.named_parameters()},
        }

    reference = evaluate("fp32")
    candidates = []
    for name in ("fp32_fused_coords", "fp32_cache_save", "fp32_cache_recompute"):
        value = evaluate(name)
        comparisons = {key: metric(reference[key], value[key]) for key in reference}
        candidates.append({
            "candidate": name,
            "output": comparisons.pop("output"),
            "grad_x": comparisons.pop("grad_x"),
            "parameter_gradients": comparisons,
            "parameter_gradient_max_relative_l2": max(
                row["relative_l2"] for row in comparisons.values()
            ),
            "parameter_gradient_min_cosine": min(
                row["cosine_similarity"] for row in comparisons.values()
            ),
        })
    return {"dtype": str(dtype), "candidates": candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [run(torch.float32), run(torch.bfloat16)]
    result = {"schema_version": 1, "rows": rows}
    result["passed"] = all(
        candidate["output"]["max_abs"] <= 1e-6
        and candidate["grad_x"]["relative_l2"] <= 1e-3
        and candidate["parameter_gradient_max_relative_l2"] <= 3e-3
        and candidate["parameter_gradient_min_cosine"] >= 0.99999
        for row in rows for candidate in row["candidates"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
