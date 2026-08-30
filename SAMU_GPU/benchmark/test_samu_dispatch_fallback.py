"""Regression test for partial/reset/h0 fallback from compressed chunk scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_training_scan import samu_tiled_serial_scan, samu_training_scan_dispatch


def scenario(name, length, segment_pos=None, initial_state=None):
    torch.manual_seed(88101 + length)
    batch, modes = 2, 129
    eta = (torch.randn(batch, length, device="cuda") * 0.1).requires_grad_(True)
    delta = (torch.randn(batch, length, device="cuda") * 0.1).requires_grad_(True)
    nu_log = (torch.randn(modes, device="cuda") - 1).requires_grad_(True)
    theta_log = torch.randn(modes, device="cuda").requires_grad_(True)
    br = (torch.randn(batch, length, modes, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    bi = (torch.randn(batch, length, modes, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    tensors = (eta, delta, nu_log, theta_log, br, bi)

    def evaluate(dispatch):
        values = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        state = None if initial_state is None else tuple(value.detach().clone().requires_grad_(True)
                                                         for value in initial_state)
        if dispatch:
            out = samu_training_scan_dispatch(
                *values, chunk_size=16, hierarchical_prefix=True,
                precompute_shared=True, compressed_transition=True,
                segment_pos=segment_pos, initial_state=state,
                reset_first=True, block_size=128,
            )
        else:
            out_r, out_i, _, _ = samu_tiled_serial_scan(
                *values, segment_pos=segment_pos, initial_state=state,
                block_size=128, reset_first=True,
            )
            out = (out_r, out_i)
        cotangent = tuple(torch.ones_like(value) for value in out)
        torch.autograd.backward(out, cotangent)
        gradients = [value.grad for value in values]
        if state is not None:
            gradients.extend(value.grad for value in state)
        return out, gradients

    reference, reference_gradients = evaluate(False)
    candidate, candidate_gradients = evaluate(True)
    output_max = max(float((a.float() - b.float()).abs().max())
                     for a, b in zip(reference, candidate))
    gradient_max = max(float((a.float() - b.float()).abs().max())
                       for a, b in zip(reference_gradients, candidate_gradients))
    return {"scenario": name, "length": length,
            "output_max_abs": output_max, "gradient_max_abs": gradient_max}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch, length = 2, 32
    packed = torch.arange(length, device="cuda", dtype=torch.int32).repeat(batch, 1)
    packed[:, 7:] -= 7
    packed[:, 21:] -= 14
    h0 = (torch.randn(batch, 129, device="cuda"), torch.randn(batch, 129, device="cuda"))
    rows = [
        scenario("partial_chunk", 33),
        scenario("packed_resets", 32, segment_pos=packed),
        scenario("nonzero_h0", 32, initial_state=h0),
    ]
    result = {"schema_version": 1, "rows": rows,
              "passed": all(row["output_max_abs"] == 0.0 and row["gradient_max_abs"] == 0.0
                            for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
