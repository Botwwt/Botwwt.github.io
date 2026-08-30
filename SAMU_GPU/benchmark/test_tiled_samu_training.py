"""Correctness matrix for wide tiled, reset-aware SAMU serial training."""

from __future__ import annotations

import json

import torch

from triton_training_scan import samu_tiled_serial_scan


def eager(eta, delta, nu_log, theta_log, br, bi, positions, h0_r, h0_i):
    nu = torch.exp(nu_log.float())
    theta = torch.exp(theta_log.float())
    state_r, state_i = h0_r.float(), h0_i.float()
    outputs_r, outputs_i = [], []
    for time_index in range(eta.shape[1]):
        radius = torch.exp(-nu * torch.exp(eta[:, time_index].float()).unsqueeze(-1))
        phase = theta + delta[:, time_index].float().unsqueeze(-1)
        ar, ai = radius * torch.cos(phase), radius * torch.sin(phase)
        reset = positions[:, time_index].eq(0).unsqueeze(-1)
        ar = torch.where(reset, torch.zeros_like(ar), ar)
        ai = torch.where(reset, torch.zeros_like(ai), ai)
        next_r = ar * state_r - ai * state_i + br[:, time_index].float()
        next_i = ai * state_r + ar * state_i + bi[:, time_index].float()
        state_r, state_i = next_r, next_i
        outputs_r.append(state_r.to(br.dtype))
        outputs_i.append(state_i.to(bi.dtype))
    return torch.stack(outputs_r, 1), torch.stack(outputs_i, 1), state_r, state_i


def clones(tensors):
    return [value.detach().clone().requires_grad_(True) for value in tensors]


def run_case(dtype, batch, length, modes, block_size):
    torch.manual_seed(7000 + batch * 1000 + length * 10 + modes + block_size)
    eta = torch.randn(batch, length, device="cuda") * 0.05
    delta = torch.randn(batch, length, device="cuda") * 0.05
    nu_log = torch.randn(modes, device="cuda") * 0.1 - 1.0
    theta_log = torch.randn(modes, device="cuda") * 0.1
    br = (torch.randn(batch, length, modes, device="cuda") * 0.1).to(dtype)
    bi = (torch.randn(batch, length, modes, device="cuda") * 0.1).to(dtype)
    h0_r = torch.randn(batch, modes, device="cuda") * 0.1
    h0_i = torch.randn(batch, modes, device="cuda") * 0.1
    positions = torch.arange(length, device="cuda", dtype=torch.int32)[None].repeat(batch, 1)
    positions[:, 0] = 5
    if length > 5:
        positions[0, 5] = 0
    if batch > 1 and length > 16:
        positions[1, 16] = 0
    if length > 17:
        positions[0, 17] = 0
    source = (eta, delta, nu_log, theta_log, br, bi, h0_r, h0_i)
    observed_inputs = clones(source)
    expected_inputs = clones(source)
    observed = samu_tiled_serial_scan(
        *observed_inputs[:6], segment_pos=positions,
        initial_state=(observed_inputs[6], observed_inputs[7]),
        block_size=block_size,
    )
    expected = eager(*expected_inputs[:6], positions, expected_inputs[6], expected_inputs[7])
    cotangents = [torch.randn_like(value) for value in observed]
    sum((value.float() * cotangent.float()).sum()
        for value, cotangent in zip(observed, cotangents)).backward()
    sum((value.float() * cotangent.float()).sum()
        for value, cotangent in zip(expected, cotangents)).backward()
    output_error = max(
        float((actual.float() - target.float()).abs().max())
        for actual, target in zip(observed, expected)
    )
    gradient_errors = {
        name: float((actual.grad.float() - target.grad.float()).abs().max())
        for name, actual, target in zip(
            ("eta", "delta", "nu_log", "theta_log", "br", "bi", "h0_r", "h0_i"),
            observed_inputs, expected_inputs,
        )
    }
    gradient_relative = {
        name: error / max(1.0, float(target.grad.float().abs().max()))
        for (name, error), target in zip(gradient_errors.items(), expected_inputs)
    }
    return {
        "dtype": str(dtype), "batch": batch, "length": length,
        "modes": modes, "block_size": block_size,
        "output_max_abs": output_error,
        "gradient_max_abs": max(gradient_errors.values()),
        "gradient_max_relative": max(gradient_relative.values()),
        "gradient_errors": gradient_errors,
        "gradient_relative_errors": gradient_relative,
    }


def main():
    cases = (
        (1, 1, 33, 32),
        (2, 17, 257, 64),
        (2, 33, 640, 128),
        (1, 65, 1024, 128),
        (1, 129, 1280, 256),
    )
    rows = [
        run_case(dtype, *case)
        for dtype in (torch.float32, torch.bfloat16)
        for case in cases
    ]
    failures = [
        row for row in rows
        if row["output_max_abs"] > (0.02 if row["dtype"] == "torch.bfloat16" else 2e-5)
        or row["gradient_max_relative"] > (0.03 if row["dtype"] == "torch.bfloat16" else 3e-4)
    ]
    report = {"passed": not failures, "failures": failures, "rows": rows}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
