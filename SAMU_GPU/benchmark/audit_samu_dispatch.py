"""Sweep SAMU serial/chunk launch geometry for short and medium prefill."""

from __future__ import annotations

import argparse
import json

import torch
import triton

from kernels import make_samu_parameters
from triton_samu import pack_samu_parameters, samu_triton_chunked, samu_triton_serial


def bench(operation) -> float:
    return float(triton.testing.do_bench(operation, warmup=25, rep=100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 2048])
    args = parser.parse_args()
    torch.manual_seed(1729)
    params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    packed = pack_samu_parameters(params)
    rows = []
    for length in args.lengths:
        x = torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16)
        rows.append({"length": length, "backend": "serial", "ms": bench(lambda: samu_triton_serial(x, params, packed, return_cache=True))})
        for chunk in (4, 8, 16, 32):
            if length % chunk:
                continue
            for warps in (1, 2, 4, 8):
                rows.append({
                    "length": length,
                    "backend": f"c{chunk}",
                    "warps": warps,
                    "ms": bench(lambda c=chunk, w=warps: samu_triton_chunked(x, params, c, packed, num_warps=w, compressed_p=False, return_cache=True)),
                })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
