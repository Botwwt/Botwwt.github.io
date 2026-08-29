"""Low-overhead equal-track SAMU/RG-LRU latency cross-check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from kernels import load_official_rglru, make_samu_parameters
from triton_rglru import pack_rglru, rglru_triton_auto, rglru_triton_decode
from triton_samu import pack_samu_parameters, samu_triton_auto, samu_triton_decode


def bench(operation) -> float:
    return float(triton.testing.do_bench(operation, warmup=25, rep=100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192, 32768, 65536])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(1729)
    params = make_samu_parameters(128, 64, "cuda", torch.bfloat16)
    samu = pack_samu_parameters(params)
    RGLRU = load_official_rglru(args.rglru_source)
    rg_model = RGLRU(width=128, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    rg = pack_rglru(rg_model)
    rows = []
    for length in args.lengths:
        x = torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        samu_ms = bench(lambda: samu_triton_auto(x, params, samu, compressed_p=False, return_cache=True))
        rg_ms = bench(lambda: rglru_triton_auto(x, positions, rg))
        rows.append({"length": length, "samu_ms": samu_ms, "rglru_ms": rg_ms, "samu_over_rglru_speedup": rg_ms / samu_ms})
    for batch in (1, 4, 16, 64):
        token = torch.randn(batch, 128, device="cuda", dtype=torch.bfloat16)
        samu_state = (torch.randn(batch, 64, device="cuda"), torch.randn(batch, 64, device="cuda"))
        rg_state = torch.randn(batch, 128, device="cuda")
        positions = torch.ones(batch, device="cuda", dtype=torch.long)
        for warps in (2, 4, 8):
            samu_ms = bench(lambda w=warps: samu_triton_decode(token, samu_state, params, samu, block_m=32, num_warps=w))
            rows.append({"workload": "decode", "batch": batch, "model": "samu", "warps": warps, "ms": samu_ms})
        rg_ms = bench(lambda: rglru_triton_decode(token, positions, rg, rg_state))
        rows.append({"workload": "decode", "batch": batch, "model": "rglru", "warps": 4, "ms": rg_ms})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
