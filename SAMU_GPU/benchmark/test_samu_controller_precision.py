"""Numerical A/B for removing SAMU's logical full-size FP32 input copy."""

from __future__ import annotations

import copy
import json

import torch

from run_small_model_study import SAMUMixer


def main():
    torch.manual_seed(73001)
    batch, length, width = 2, 127, 512
    reference = SAMUMixer(width).cuda().train()
    with torch.no_grad():
        reference.phase_amplitude.fill_(0.7)
        reference.radial_amplitude.fill_(0.5)
    candidate = copy.deepcopy(reference)
    reference.set_scan_backend("tiled_serial")
    candidate.set_scan_backend("tiled_serial")
    reference.set_controller_projection_dtype("fp32")
    candidate.set_controller_projection_dtype("bf16")
    x_reference = torch.randn(
        batch, length, width, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    x_candidate = x_reference.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_reference, _ = reference(x_reference)
        output_candidate, _ = candidate(x_candidate)
    cotangent = torch.randn_like(output_reference)
    (output_reference.float() * cotangent.float()).sum().backward()
    (output_candidate.float() * cotangent.float()).sum().backward()
    parameter_rows = []
    for (name_reference, parameter_reference), (name_candidate, parameter_candidate) in zip(
        reference.named_parameters(), candidate.named_parameters()
    ):
        assert name_reference == name_candidate
        absolute = float(
            (parameter_candidate.grad.float() - parameter_reference.grad.float()).abs().max()
        )
        relative = absolute / max(1.0, float(parameter_reference.grad.float().abs().max()))
        parameter_rows.append({"name": name_reference, "max_abs": absolute, "max_relative": relative})
    input_absolute = float((x_candidate.grad.float() - x_reference.grad.float()).abs().max())
    input_relative = input_absolute / max(1.0, float(x_reference.grad.float().abs().max()))
    output_absolute = float((output_candidate.float() - output_reference.float()).abs().max())
    report = {
        "shape": {"batch": batch, "length": length, "width": width},
        "reference": "explicit full-size FP32 controller input",
        "candidate": "BF16 input/directions, FP32 two-value controller nonlinearities",
        "output_max_abs": output_absolute,
        "input_gradient_max_abs": input_absolute,
        "input_gradient_max_relative": input_relative,
        "parameter_gradients": parameter_rows,
        "parameter_gradient_max_relative": max(row["max_relative"] for row in parameter_rows),
    }
    report["passed_production_tolerance"] = (
        output_absolute <= 0.02
        and input_relative <= 0.05
        and report["parameter_gradient_max_relative"] <= 0.05
    )
    print(json.dumps(report, indent=2))
    if not report["passed_production_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
