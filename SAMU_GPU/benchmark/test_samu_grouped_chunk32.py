"""Interaction gate for SAMU shared-SFU, grouped prefix and K=32 replay."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import torch

from run_small_model_study import SAMUMixer, set_seed


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


def evaluate(module, source, cotangent):
    module.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_(True)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if source.dtype == torch.bfloat16 else contextlib.nullcontext()
    )
    with autocast:
        output = module(x)[0]
    (output.float() * cotangent.float()).sum().backward()
    return output.detach(), x.grad.detach(), {
        name: parameter.grad.detach()
        for name, parameter in module.named_parameters()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_seed(76517)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        # 33 chunks crosses a group boundary and exercises a partial final group.
        batch, length, width = 1, 33 * 32, 258
        reference = SAMUMixer(width).cuda().train()
        candidate = copy.deepcopy(reference)
        reference.set_scan_backend(
            "fused_output_fused_write_shared_sfu_chunk32"
        )
        candidate.set_scan_backend(
            "fused_output_fused_write_shared_sfu_"
            "grouped_prefix64_hierarchical_chunk32"
        )
        for module in (reference, candidate):
            module.set_tiled_training_block_size(256)
            module.set_controller_projection_dtype("fp32_fused_coords")
            with torch.no_grad():
                module.phase_amplitude.fill_(0.7)
                module.radial_amplitude.fill_(0.5)
        source = torch.randn(
            batch, length, width, device="cuda", dtype=dtype
        )
        cotangent = torch.randn_like(source)
        expected = evaluate(reference, source, cotangent)
        actual = evaluate(candidate, source, cotangent)
        rows.append({
            "dtype": str(dtype),
            "length": length,
            "output": metric(expected[0], actual[0]),
            "input_gradient": metric(expected[1], actual[1]),
            "parameter_gradients": {
                name: metric(expected[2][name], actual[2][name])
                for name in expected[2]
            },
        })
        del reference, candidate, source, cotangent
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "reference": "exact serial chunk-prefix K=32 with identical fused paths",
        "candidate": "shared-SFU + grouped64 affine prefix + K=32 replay",
        "rows": rows,
    }
    result["passed"] = all(
        row["output"]["relative_l2"] <= 5.0e-6
        and row["input_gradient"]["relative_l2"] <= 5.0e-6
        and max(value["relative_l2"] for value in
                row["parameter_gradients"].values()) <= 5.0e-5
        and row["output"]["cosine_similarity"] >= 0.99999
        and row["input_gradient"]["cosine_similarity"] >= 0.99999
        and min(value["cosine_similarity"] for value in
                row["parameter_gradients"].values()) >= 0.99999
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
