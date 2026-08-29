"""Audit and tune SAMU's parameter-certified bounded-math inference path."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch
import triton

from kernels import make_samu_parameters
from triton_samu import (
    pack_samu_parameters,
    samu_triton_auto,
    samu_triton_decode,
)


def bench(operation, warmup: int = 25, repeat: int = 100) -> float:
    return float(triton.testing.do_bench(operation, warmup=warmup, rep=repeat))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192, 32768, 65536])
    args = parser.parse_args()
    torch.manual_seed(1729)
    params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    bounded = pack_samu_parameters(params, enable_bounded_poly=True)
    exact = replace(bounded, use_bounded_poly=False)
    report = {
        "bounds": {
            "enabled": bounded.use_bounded_poly,
            "max_abs_c": bounded.max_abs_c,
            "max_decay_exponent": bounded.max_decay_exponent,
        },
        "prefill": [],
        "decode": [],
    }
    for length in args.lengths:
        x = torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16)
        bounded_out, bounded_cache = samu_triton_auto(x, params, bounded, compressed_p=False, return_cache=True)
        exact_out, exact_cache = samu_triton_auto(x, params, exact, compressed_p=False, return_cache=True)
        report["prefill"].append({
            "length": length,
            "bounded_ms": bench(lambda: samu_triton_auto(x, params, bounded, compressed_p=False, return_cache=True)),
            "exact_exp_ms": bench(lambda: samu_triton_auto(x, params, exact, compressed_p=False, return_cache=True)),
            "output_max_abs": float((bounded_out - exact_out).abs().max()),
            "output_mean_abs": float((bounded_out - exact_out).abs().float().mean()),
            "cache_max_abs": float((torch.stack(bounded_cache, -1) - torch.stack(exact_cache, -1)).abs().max()),
        })
        del x, bounded_out, exact_out, bounded_cache, exact_cache

    token = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    state = (torch.randn(1, 64, device="cuda"), torch.randn(1, 64, device="cuda"))
    exact_decode = samu_triton_decode(token, state, params, exact, block_m=32, num_warps=4)
    for block_m in (16, 32, 64, 128):
        for num_warps in (2, 4, 8):
            try:
                output = samu_triton_decode(token, state, params, bounded, block_m=block_m, num_warps=num_warps)
                report["decode"].append({
                    "block_m": block_m,
                    "num_warps": num_warps,
                    "bounded_ms": bench(lambda bm=block_m, nw=num_warps: samu_triton_decode(token, state, params, bounded, block_m=bm, num_warps=nw)),
                    "exact_exp_ms": bench(lambda bm=block_m, nw=num_warps: samu_triton_decode(token, state, params, exact, block_m=bm, num_warps=nw)),
                    "max_abs_vs_exact_exp": float((torch.stack(output, -1) - torch.stack(exact_decode, -1)).abs().max()),
                })
            except Exception as error:
                report["decode"].append({"block_m": block_m, "num_warps": num_warps, "error": str(error)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
