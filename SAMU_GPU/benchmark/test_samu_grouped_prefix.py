"""Correctness gate for the O(C) grouped complex affine prefix hierarchy."""

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
    reference_norm = torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / reference_norm).item(),
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
    set_seed(76431)
    rows = []
    configurations = (
        (3, 32),
        (33, 32),
        (65, 64),
    )
    for dtype in (torch.float32, torch.bfloat16):
        for chunks, group_size in configurations:
            batch, width, length = 1, 258, chunks * 16
            sequential = SAMUMixer(width).cuda().train()
            hillis = copy.deepcopy(sequential)
            grouped = copy.deepcopy(sequential)
            sequential.set_scan_backend("tiled_serial")
            hillis.set_scan_backend("fused_output_hierarchical_chunk16")
            grouped.set_scan_backend(
                f"fused_output_grouped_prefix{group_size}_hierarchical_chunk16"
            )
            for module in (sequential, hillis, grouped):
                module.set_tiled_training_block_size(256)
                module.set_controller_projection_dtype("fp32_fused_coords")
                with torch.no_grad():
                    module.phase_amplitude.fill_(0.7)
                    module.radial_amplitude.fill_(0.5)
            source = torch.randn(
                batch, length, width, device="cuda", dtype=dtype
            )
            cotangent = torch.randn_like(source)
            expected = evaluate(sequential, source, cotangent)
            evaluated = {}
            for name, module in (("hillis_steele", hillis), ("grouped", grouped)):
                actual = evaluate(module, source, cotangent)
                evaluated[name] = actual
                rows.append({
                    "dtype": str(dtype),
                    "chunks": chunks,
                    "group_size": group_size,
                    "candidate": name,
                    "output": metric(expected[0], actual[0]),
                    "input_gradient": metric(expected[1], actual[1]),
                    "parameter_gradients": {
                        parameter_name: metric(
                            expected[2][parameter_name],
                            actual[2][parameter_name],
                        )
                        for parameter_name in expected[2]
                    },
                })
            grouped_row = rows[-1]
            grouped_row["comparison_to_hillis_steele"] = {
                "output": metric(evaluated["hillis_steele"][0],
                                 evaluated["grouped"][0]),
                "input_gradient": metric(evaluated["hillis_steele"][1],
                                         evaluated["grouped"][1]),
                "parameter_gradients": {
                    parameter_name: metric(
                        evaluated["hillis_steele"][2][parameter_name],
                        evaluated["grouped"][2][parameter_name],
                    )
                    for parameter_name in evaluated["hillis_steele"][2]
                },
            }
            del sequential, hillis, grouped, source, cotangent
            torch.cuda.empty_cache()
    result = {"schema_version": 1, "reference": "tiled serial", "rows": rows}
    result["sequential_reference_passed"] = all(
        row["output"]["relative_l2"] <= 5.0e-6
        and row["input_gradient"]["relative_l2"] <= 5.0e-5
        and max(value["relative_l2"]
                for value in row["parameter_gradients"].values()) <= 1.0e-3
        and row["output"]["cosine_similarity"] >= 0.99999
        and row["input_gradient"]["cosine_similarity"] >= 0.99999
        and min(value["cosine_similarity"]
                for value in row["parameter_gradients"].values()) >= 0.99999
        for row in rows
    )
    grouped_rows = [row for row in rows if row["candidate"] == "grouped"]
    result["regression_to_hillis_steele_passed"] = all(
        row["comparison_to_hillis_steele"]["output"]["relative_l2"] <= 5.0e-6
        and row["comparison_to_hillis_steele"]["input_gradient"]["relative_l2"] <= 5.0e-6
        and max(value["relative_l2"] for value in
                row["comparison_to_hillis_steele"]["parameter_gradients"].values())
        <= 5.0e-5
        and min(value["cosine_similarity"] for value in
                row["comparison_to_hillis_steele"]["parameter_gradients"].values())
        >= 0.99999
        for row in grouped_rows
    )
    result["passed"] = (
        result["sequential_reference_passed"]
        and result["regression_to_hillis_steele_passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
