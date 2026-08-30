"""Measure recurrence amplification after controller-only numerical checks."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from run_small_model_study import SAMUMixer


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().flatten()
    candidate = candidate.detach().float().flatten()
    difference = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    denominator = reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    if float(reference_norm) == 0.0 and float(candidate_norm) == 0.0:
        cosine = 1.0
    elif float(reference_norm) == 0.0 or float(candidate_norm) == 0.0:
        cosine = 0.0
    else:
        cosine = float(F.cosine_similarity(reference, candidate, dim=0))
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
        "cosine_similarity": cosine,
    }


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def run(module, x_source, cotangent, projection_dtype):
    candidate = copy.deepcopy(module)
    candidate.set_scan_backend("tiled_serial")
    candidate.set_tiled_training_block_size(128)
    candidate.set_controller_projection_dtype(projection_dtype)
    x = x_source.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = candidate(x)[0]
    (output.float() * cotangent.float()).sum().backward()
    return {
        "output": output.detach(),
        "grad_x": x.grad.detach(),
        **{f"parameter::{name}": parameter.grad.detach()
           for name, parameter in candidate.named_parameters()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=77031)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    shapes = ((2, 127, 512), (1, 511, 1024))
    result = {
        "schema_version": 1,
        "protocol": {
            "scope": "formal controller through exact SAMU tiled recurrence",
            "reference": "FP32 cuBLAS projection",
            "input_dtype": "BF16",
            "active_amplitudes": {"phase": 0.7, "radial": 0.5},
            "fp32_repeat": "measures the run-to-run numerical floor of the scan",
        },
        "rows": [],
    }
    for batch, length, width in shapes:
        base = SAMUMixer(width).cuda().train()
        with torch.no_grad():
            base.phase_amplitude.fill_(0.7)
            base.radial_amplitude.fill_(0.5)
        x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        cotangent = torch.randn_like(x)
        reference = run(base, x, cotangent, "fp32")
        comparisons = []
        for name, dtype in (
            ("fp32_repeat", "fp32"),
            ("triton_tree_fp32", "triton_fp32"),
            ("cublas_bf16_fp32_accum", "bf16"),
            ("fp32_forward_recompute_backward", "fp32_recompute"),
            ("fp32_forward_triton_backward", "fp32_triton_backward"),
        ):
            candidate = run(base, x, cotangent, dtype)
            values = {key: metric(reference[key], candidate[key]) for key in reference}
            comparisons.append({
                "candidate": name,
                "output": values.pop("output"),
                "grad_x": values.pop("grad_x"),
                "parameter_gradients": values,
                "parameter_gradient_max_relative_l2": max(
                    value["relative_l2"] for value in values.values()
                ),
                "parameter_gradient_min_cosine": min(
                    value["cosine_similarity"] for value in values.values()
                ),
            })
        result["rows"].append({
            "shape": {"batch": batch, "length": length, "width": width},
            "comparisons": comparisons,
        })
        atomic_json(args.output, result)
        del base, x, cotangent, reference
        torch.cuda.empty_cache()
    print(f"complete {args.output}")


if __name__ == "__main__":
    main()
