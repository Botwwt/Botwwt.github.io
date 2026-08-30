"""Regression gate for the measured BF16 SAMU auto-dispatch table."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from run_small_model_study import SAMUMixer, set_seed


CASES = (
    (4, 2048, 258, "fused_output_fused_write_shared_sfu_chunk32"),
    (1, 4096, 258,
     "fused_output_fused_write_shared_sfu_chunk32"),
    (1, 8192, 2048,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
    (1, 16384, 258,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
    (1, 65536, 18,
     "fused_output_fused_write_shared_sfu_serial_forward_prefix_"
     "grouped_prefix64_hierarchical_chunk32"),
)


def metric(a, b):
    a, b = a.float(), b.float()
    difference = b - a
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) /
                        torch.linalg.vector_norm(a).clamp_min(1.0e-12)).item(),
    }


def evaluate(module, source, cotangent):
    module.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(x)[0]
    (output.float() * cotangent.float()).sum().backward()
    return output.detach(), x.grad.detach(), {
        name: parameter.grad.detach() for name, parameter in module.named_parameters()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_seed(76319)
    rows = []
    for batch, length, width, expected_backend in CASES:
        automatic = SAMUMixer(width).cuda().train()
        explicit = copy.deepcopy(automatic)
        automatic.set_scan_backend("auto")
        explicit.set_scan_backend(expected_backend)
        for module in (automatic, explicit):
            module.set_tiled_training_block_size(256)
            with torch.no_grad():
                module.phase_amplitude.fill_(0.7)
                module.radial_amplitude.fill_(0.5)
        source = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        cotangent = torch.randn_like(source)
        auto = evaluate(automatic, source, cotangent)
        reference = evaluate(explicit, source, cotangent)
        rows.append({
            "shape": {"batch": batch, "length": length, "width": width},
            "expected_backend": expected_backend,
            "output": metric(reference[0], auto[0]),
            "input_gradient": metric(reference[1], auto[1]),
            "parameter_gradients": {
                name: metric(reference[2][name], auto[2][name])
                for name in reference[2]
            },
        })
    result = {"schema_version": 1, "rows": rows}
    result["passed"] = all(
        row["output"]["max_abs"] == 0.0
        and row["input_gradient"]["max_abs"] == 0.0
        # Spectral parameters use FP32 atomics across chunks; two otherwise
        # identical launches may differ by a few ULPs in accumulation order.
        and max(value["relative_l2"]
                for value in row["parameter_gradients"].values()) <= 2.0e-6
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
