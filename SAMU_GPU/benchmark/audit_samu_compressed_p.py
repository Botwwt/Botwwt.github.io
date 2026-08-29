"""Compare SAMU chunk summaries with vector-P recurrence vs (G,D) reconstruction."""

from __future__ import annotations

import argparse
import json

import torch
import triton

from kernels import make_samu_parameters
from triton_samu import pack_samu_parameters, samu_triton_chunked


def bench(operation, warmup: int = 25, repeat: int = 100) -> float:
    return float(triton.testing.do_bench(operation, warmup=warmup, rep=repeat))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192, 32768, 65536])
    parser.add_argument("--chunks", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()
    torch.manual_seed(1729)
    params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    packed = pack_samu_parameters(params)
    rows = []
    for length in args.lengths:
        x = torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16)
        for chunk in args.chunks:
            if length % chunk:
                continue
            compressed_out, compressed_cache = samu_triton_chunked(
                x, params, chunk, packed, compressed_p=True, return_cache=True
            )
            vector_out, vector_cache = samu_triton_chunked(
                x, params, chunk, packed, compressed_p=False, return_cache=True
            )
            rows.append({
                "length": length,
                "chunk": chunk,
                "compressed_ms": bench(lambda: samu_triton_chunked(x, params, chunk, packed, compressed_p=True, return_cache=True)),
                "vector_p_ms": bench(lambda: samu_triton_chunked(x, params, chunk, packed, compressed_p=False, return_cache=True)),
                "output_max_abs": float((compressed_out - vector_out).abs().max()),
                "output_mean_abs": float((compressed_out - vector_out).abs().float().mean()),
                "cache_max_abs": float((torch.stack(compressed_cache, -1) - torch.stack(vector_cache, -1)).abs().max()),
            })
        del x
    print(json.dumps({"method": "on-chip (G,D) sufficient statistics; materialized q and P remain mode-sized", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
