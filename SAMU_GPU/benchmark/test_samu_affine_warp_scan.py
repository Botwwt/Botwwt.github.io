"""Correctness gates for ordered SAMU affine tile scan and backward."""

from __future__ import annotations

import json

import torch

from canonical_samu_affine_reference import samu_sequential_from_shared
from triton_samu_affine_warp_scan import samu_affine_tile_scan


def clone_grad(values):
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def run_case(length, modes, dtype, steps, mode_block):
    torch.manual_seed(83000 + length + modes + steps + mode_block)
    batch = 2
    eta = torch.randn(batch, length, device="cuda", dtype=torch.float32) * 0.04
    delta = torch.randn(batch, length, device="cuda", dtype=torch.float32) * 0.04
    nu_log = torch.randn(modes, device="cuda", dtype=torch.float32) * 0.1
    theta_log = torch.randn(modes, device="cuda", dtype=torch.float32) * 0.1
    br = torch.randn(batch, length, modes, device="cuda", dtype=dtype) * 0.1
    bi = torch.randn_like(br) * 0.1
    h0r = torch.randn(batch, modes, device="cuda", dtype=torch.float32) * 0.1
    h0i = torch.randn_like(h0r) * 0.1
    positions = torch.arange(length, device="cuda", dtype=torch.int32)[None].repeat(batch, 1)
    positions[0, 0] = 7
    if length > 2:
        positions[1, 2] = 0
    if length > 7:
        positions[0, 7] = 0
    if length > steps:
        positions[1, steps] = 0
    source = (eta, delta, nu_log, theta_log, br, bi, h0r, h0i)
    observed_inputs = clone_grad(source)
    expected_inputs = clone_grad(source)
    observed_r, observed_i, observed_last_r, observed_last_i = samu_affine_tile_scan(
        *observed_inputs[:6], segment_pos=positions,
        initial_state=observed_inputs[6:], steps=steps, mode_block=mode_block,
    )
    ee, ed, en, et, ebr, ebi, eh0r, eh0i = expected_inputs
    spectral_nu, spectral_theta = en.exp(), et.exp()
    expected, expected_last = samu_sequential_from_shared(
        ee.exp(), ed.cos(), ed.sin(), spectral_nu,
        spectral_theta.cos(), spectral_theta.sin(),
        torch.complex(ebr.float(), ebi.float()),
        segment_pos=positions,
        initial_state=torch.complex(eh0r, eh0i),
    )
    # Production kernels expose BF16/FP32 real tensors rather than complex64.
    expected_r, expected_i = expected.real.to(dtype), expected.imag.to(dtype)
    weights = (torch.randn_like(observed_r), torch.randn_like(observed_i),
               torch.randn_like(observed_last_r), torch.randn_like(observed_last_i))
    observed_loss = sum((value.float() * weight.float()).sum()
                        for value, weight in zip(
                            (observed_r, observed_i, observed_last_r, observed_last_i), weights))
    expected_loss = sum((value.float() * weight.float()).sum()
                        for value, weight in zip(
                            (expected_r, expected_i, expected_last.real,
                             expected_last.imag), weights))
    observed_loss.backward(); expected_loss.backward()
    output_pairs = ((observed_r, expected_r), (observed_i, expected_i),
                    (observed_last_r, expected_last.real),
                    (observed_last_i, expected_last.imag))
    output_max = max(float((a.float() - b.float()).abs().max()) for a, b in output_pairs)
    output_l2 = max(float(torch.linalg.vector_norm((a-b).float()) /
                          torch.linalg.vector_norm(b.float()).clamp_min(1e-12))
                    for a, b in output_pairs)
    gradient_rows = []
    for name, actual, target in zip(
            ("eta", "delta", "nu_log", "theta_log", "br", "bi", "h0r", "h0i"),
            observed_inputs, expected_inputs):
        a, b = actual.grad.float().flatten(), target.grad.float().flatten()
        absolute = float((a - b).abs().max())
        relative_l2 = float(torch.linalg.vector_norm(a-b) /
                            torch.linalg.vector_norm(b).clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        gradient_rows.append({"name": name, "max_abs": absolute,
                              "relative_l2": relative_l2, "cosine": cosine})
    return {"dtype": str(dtype), "batch": batch, "length": length,
            "modes": modes, "steps": steps, "mode_block": mode_block,
            "output_max_abs": output_max, "output_relative_l2": output_l2,
            "gradient_max_abs": max(row["max_abs"] for row in gradient_rows),
            "gradient_relative_l2": max(row["relative_l2"] for row in gradient_rows),
            "gradient_min_cosine": min(row["cosine"] for row in gradient_rows),
            "gradients": gradient_rows}


def main():
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for length in (1, 2, 3, 7, 31, 32, 33):
            rows.append(run_case(length, 19, dtype, 16, 32))
    failures = [row for row in rows if
                row["output_max_abs"] > (0.03 if "bfloat16" in row["dtype"] else 5e-4)
                or row["gradient_relative_l2"] > (0.03 if "bfloat16" in row["dtype"] else 5e-3)
                or row["gradient_min_cosine"] < 0.999]
    report = {"passed": not failures, "failures": failures, "rows": rows}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
