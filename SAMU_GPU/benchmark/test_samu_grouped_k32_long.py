"""Long-sequence regression gate for the selected grouped K=32 SAMU path.

The grouped prefix changes only the parenthesization of exact complex-affine
chunk composition.  This gate compares it with the serial chunk-prefix path at
the lengths used for the final scaling study, including all input and parameter
gradients.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import torch

from run_small_model_study import SAMUMixer, set_seed


def metric(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float()
    actual = actual.detach().float()
    difference = actual - reference
    norm = torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / norm).item(),
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
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }


def parse_case(value: str) -> tuple[int, torch.dtype]:
    length_text, dtype_text = value.split(":", 1)
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype_text not in dtypes:
        raise argparse.ArgumentTypeError("dtype must be fp32 or bf16")
    return int(length_text), dtypes[dtype_text]


def run_case(length: int, dtype: torch.dtype, seed: int, width: int,
             hybrid: bool, group_size: int, candidate_serial: bool) -> dict:
    # Long-prefix stability is independent of the number of mode tiles.  Keep
    # this gate narrow so it does not turn into another width/memory benchmark;
    # partial and multi-tile widths are covered by the regular correctness gate.
    batch = 1
    set_seed(seed)
    reference = SAMUMixer(width).cuda().train()
    candidate = copy.deepcopy(reference)
    reference.set_scan_backend(
        "fused_output_fused_write_shared_sfu_chunk32"
    )
    candidate_backend = "fused_output_fused_write_shared_sfu_chunk32"
    if not candidate_serial:
        candidate_backend = (
            "fused_output_fused_write_shared_sfu_"
            + ("serial_forward_prefix_" if hybrid else "")
            + f"grouped_prefix{group_size}_hierarchical_chunk32"
        )
    candidate.set_scan_backend(candidate_backend)
    for module in (reference, candidate):
        module.set_tiled_training_block_size(256)
        module.set_controller_projection_dtype("fp32_fused_coords")
        with torch.no_grad():
            module.phase_amplitude.fill_(0.7)
            module.radial_amplitude.fill_(0.5)
    source = torch.randn(batch, length, width, device="cuda", dtype=dtype)
    cotangent = torch.randn_like(source)
    expected = evaluate(reference, source, cotangent)
    actual = evaluate(candidate, source, cotangent)
    row = {
        "shape": {"batch": batch, "length": length, "width": width},
        "dtype": str(dtype),
        "chunks": length // 32,
        "prefix_group_size": group_size,
        "prefix_groups": (length // 32 + group_size - 1) // group_size,
        "hybrid_serial_forward": hybrid,
        "candidate_serial_control": candidate_serial,
        "output": metric(expected[0], actual[0]),
        "input_gradient": metric(expected[1], actual[1]),
        "parameter_gradients": {
            name: metric(expected[2][name], actual[2][name])
            for name in expected[2]
        },
    }
    del reference, candidate, source, cotangent, expected, actual
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", type=parse_case, default=[])
    parser.add_argument("--width", type=int, default=18)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128),
                        default=64)
    parser.add_argument("--candidate-serial", action="store_true")
    args = parser.parse_args()
    cases = args.case or [
        (65536, torch.float32),
        (65536, torch.bfloat16),
        (131072, torch.bfloat16),
    ]
    rows = [
        run_case(length, dtype, 76821 + index, args.width, args.hybrid,
                 args.group_size, args.candidate_serial)
        for index, (length, dtype) in enumerate(cases)
    ]
    result = {
        "schema_version": 1,
        "reference": "serial exact chunk-prefix K=32 with identical fused paths",
        "candidate": (
            "independent serial K=32 control"
            if args.candidate_serial else (
                f"serial forward plus grouped{args.group_size} reverse "
                "complex-affine prefix K=32"
                if args.hybrid else
                f"grouped{args.group_size} complex-affine prefix K=32"
            )
        ),
        "thresholds_predeclared": {
            "output_relative_l2": 5.0e-6,
            "input_gradient_relative_l2": 5.0e-5,
            "parameter_gradient_relative_l2": 1.0e-3,
            "minimum_cosine_similarity": 0.99999,
        },
        "rows": rows,
    }
    result["passed"] = all(
        row["output"]["relative_l2"] <= 5.0e-6
        and row["input_gradient"]["relative_l2"] <= 5.0e-5
        and max(value["relative_l2"] for value in
                row["parameter_gradients"].values()) <= 1.0e-3
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
