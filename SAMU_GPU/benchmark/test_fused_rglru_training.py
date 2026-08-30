"""Correctness matrix for fused RG-LRU training preparation and scans."""

from __future__ import annotations

import json

import torch

from canonical_rg_lru_reference import canonical_rg_lru_from_logits
from triton_fused_rglru_training import fused_rglru_scan


def reference(x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0):
    return canonical_rg_lru_from_logits(
        x, gate_x_logit, gate_a_logit, a_param,
        segment_pos=segment_pos, initial_state=h0,
        production_dtype=x.dtype if x.dtype == torch.bfloat16 else None,
    )


def clone_grad(*tensors):
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def run_case(dtype, length, width, chunk_size, hierarchical_prefix=False):
    torch.manual_seed(1000 + length + width + chunk_size)
    batch = 2
    x = torch.randn(batch, length, width, device="cuda", dtype=dtype) * 0.2
    gx = torch.randn_like(x) * 0.5
    ga = torch.randn_like(x) * 0.5
    a_param = torch.randn(width, device="cuda", dtype=torch.float32) * 0.2 + 1.0
    h0 = torch.randn(batch, width, device="cuda", dtype=torch.float32) * 0.1
    positions = torch.arange(length, device="cuda", dtype=torch.int32)[None].repeat(batch, 1)
    positions[0, 0] = 3  # exercise a non-reset initial state
    if length > 5:
        positions[0, 5] = 0
    if length > 8:
        positions[1, length // 2] = 0
    if length > 17:
        positions[0, 16] = 0  # exact chunk boundary for C16
        positions[1, 17] = 0  # inside the following chunk

    observed_inputs = clone_grad(x, gx, ga, a_param, h0)
    expected_inputs = clone_grad(x, gx, ga, a_param, h0)
    observed, observed_last = fused_rglru_scan(
        *observed_inputs[:4], segment_pos=positions, h0=observed_inputs[4],
        chunk_size=chunk_size, block_size=64, reset_first=False,
        hierarchical_prefix=hierarchical_prefix,
    )
    expected, expected_last = reference(
        *expected_inputs[:4], positions, expected_inputs[4]
    )
    weight = torch.randn_like(observed)
    last_weight = torch.randn_like(observed_last)
    ((observed.float() * weight.float()).sum() +
     (observed_last * last_weight).sum()).backward()
    ((expected.float() * weight.float()).sum() +
     (expected_last * last_weight).sum()).backward()
    grad_errors = {
        name: float((actual.grad.float() - target.grad.float()).abs().max())
        for name, actual, target in zip(
            ("x", "gate_x_logit", "gate_a_logit", "a_param", "h0"),
            observed_inputs,
            expected_inputs,
        )
    }
    grad_relative = {
        name: error / max(1.0, float(target.grad.float().abs().max()))
        for (name, error), target in zip(grad_errors.items(), expected_inputs)
    }
    return {
        "dtype": str(dtype),
        "batch": batch,
        "length": length,
        "width": width,
        "chunk_size": chunk_size,
        "hierarchical_prefix": hierarchical_prefix,
        "output_max_abs": float((observed.float() - expected.float()).abs().max()),
        "last_max_abs": float((observed_last - expected_last).abs().max()),
        "gradient_max_abs": max(grad_errors.values()),
        "gradient_max_relative": max(grad_relative.values()),
        "gradient_errors": grad_errors,
        "gradient_relative_errors": grad_relative,
    }


def main():
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for length, width in ((1, 33), (17, 65), (33, 129), (65, 256)):
            for chunk_size in (0, 8, 16, 32):
                rows.append(run_case(dtype, length, width, chunk_size))
            rows.append(run_case(dtype, length, width, 16, True))
    failures = []
    for row in rows:
        bf16 = row["dtype"] == "torch.bfloat16"
        output_limit = 0.02 if bf16 else 2e-5
        last_limit = 0.02 if bf16 else 2e-5
        relative_limit = 0.02 if bf16 else 2e-4
        if (row["output_max_abs"] > output_limit or
                row["last_max_abs"] > last_limit or
                row["gradient_max_relative"] > relative_limit):
            failures.append(row)
    report = {"passed": not failures, "failures": failures, "rows": rows}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
