"""Interaction gate for compressed G/D with the selected grouped K32 path."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import torch

from run_small_model_study import SAMUMixer, set_seed


BASE = (
    "fused_output_fused_write_shared_sfu_"
    "grouped_prefix64_hierarchical_chunk32"
)
COMPRESSED = (
    "fused_output_fused_write_shared_sfu_compressed_"
    "grouped_prefix64_hierarchical_chunk32"
)


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


def evaluate(module, source, cotangent):
    module.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_(True)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if source.dtype == torch.bfloat16 else contextlib.nullcontext())
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
    set_seed(76543)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        batch, length, width = 1, 96, 512
        reference = SAMUMixer(width).cuda().train()
        candidate = copy.deepcopy(reference)
        for module, backend in ((reference, BASE), (candidate, COMPRESSED)):
            module.set_scan_backend(backend)
            module.set_tiled_training_block_size(256)
            module.set_controller_projection_dtype("fp32_fused_coords")
            with torch.no_grad():
                module.phase_amplitude.fill_(0.7)
                module.radial_amplitude.fill_(0.5)
        source = torch.randn(batch, length, width, device="cuda", dtype=dtype)
        cotangent = torch.randn_like(source)
        expected = evaluate(reference, source, cotangent)
        actual = evaluate(candidate, source, cotangent)
        rows.append({
            "dtype": str(dtype),
            "output": metric(expected[0], actual[0]),
            "input_gradient": metric(expected[1], actual[1]),
            "parameter_gradients": {
                name: metric(expected[2][name], actual[2][name])
                for name in expected[2]
            },
        })
    result = {
        "schema_version": 1,
        "reference": "materialized transient P grouped K32",
        "candidate": "compressed G/D plus transient-P reconstruction grouped K32",
        "rows": rows,
    }
    result["passed"] = all(
        row["output"]["relative_l2"] <= 5e-6
        and row["input_gradient"]["relative_l2"] <= 5e-5
        and max(value["relative_l2"] for value in
                row["parameter_gradients"].values()) <= 1e-3
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
