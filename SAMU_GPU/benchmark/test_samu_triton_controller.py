"""Correctness of the FP32-accumulating Triton SAMU controller."""

from __future__ import annotations

import copy
import json

import torch

from run_small_model_study import SAMUMixer


def main():
    torch.manual_seed(77001)
    batch, length, width = 2, 127, 512
    reference = SAMUMixer(width).cuda().train()
    with torch.no_grad():
        reference.phase_amplitude.fill_(0.7)
        reference.radial_amplitude.fill_(0.5)
    candidate = copy.deepcopy(reference)
    for module in (reference, candidate):
        module.set_scan_backend("tiled_serial")
    reference.set_controller_projection_dtype("fp32")
    candidate.set_controller_projection_dtype("triton_fp32")
    x_reference = torch.randn(batch, length, width, device="cuda",
                              dtype=torch.bfloat16, requires_grad=True)
    x_candidate = x_reference.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_reference, _ = reference(x_reference)
        output_candidate, _ = candidate(x_candidate)
    cotangent = torch.randn_like(output_reference)
    (output_reference.float() * cotangent.float()).sum().backward()
    (output_candidate.float() * cotangent.float()).sum().backward()
    parameters = []
    for (reference_name, reference_parameter), (candidate_name, candidate_parameter) in zip(
            reference.named_parameters(), candidate.named_parameters()):
        assert reference_name == candidate_name
        absolute = float((candidate_parameter.grad.float() -
                          reference_parameter.grad.float()).abs().max())
        relative = absolute / max(1.0, float(reference_parameter.grad.float().abs().max()))
        parameters.append({"name": reference_name, "max_abs": absolute,
                           "max_relative": relative})
    input_absolute = float((x_candidate.grad.float() -
                            x_reference.grad.float()).abs().max())
    input_relative = input_absolute / max(1.0, float(x_reference.grad.float().abs().max()))
    output_absolute = float((output_candidate.float() -
                             output_reference.float()).abs().max())
    report = {
        "shape": {"batch": batch, "length": length, "width": width},
        "reference": "x.float() plus torch F.linear, FP32 weights/accumulation",
        "candidate": "direct BF16 reads, tiled Triton dot, FP32 weights/accumulation",
        "output_max_abs": output_absolute,
        "input_gradient_max_abs": input_absolute,
        "input_gradient_max_relative": input_relative,
        "parameter_gradients": parameters,
        "parameter_gradient_max_relative": max(row["max_relative"] for row in parameters),
    }
    report["passed_production_tolerance"] = (
        output_absolute <= 0.005 and input_relative <= 0.01
        and report["parameter_gradient_max_relative"] <= 0.01
    )
    print(json.dumps(report, indent=2))
    if not report["passed_production_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
