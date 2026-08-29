"""Recorded architecture-specific dispatch sweep for SAMU and RG-LRU.

This is a calibration run, never the final reported timing run.  It searches
the same serial/chunk/warp space for both models, writes every candidate, and
lets a later independent run use the selected configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from kernels import load_official_rglru, make_samu_parameters
from run_equal_kernel_benchmarks import measure, stabilize_gpu
from triton_rglru import (
    pack_rglru,
    rglru_triton_chunked,
    rglru_triton_decode,
    rglru_triton_serial,
)
from triton_samu import (
    pack_samu_parameters,
    samu_triton_chunked,
    samu_triton_decode,
    samu_triton_decode_split,
    samu_triton_serial,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def timed(rows: list[dict], base: dict, operation, warmup_ms: int, rep_ms: int) -> None:
    try:
        timing, compile_seconds = measure(
            operation, warmup_ms=warmup_ms, rep_ms=rep_ms
        )
        rows.append({**base, "status": "measured", "compile_seconds": compile_seconds, **timing})
        print(base, timing["median_ms"], flush=True)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        rows.append({**base, "status": "unsupported", "error": str(error)})
        torch.cuda.empty_cache()
        print(base, "unsupported", repr(error), flush=True)


def prefill_sweep(rows: list[dict], rglru_source: Path, warmup_ms: int, rep_ms: int) -> None:
    d, modes = 128, 64
    samu_params = make_samu_parameters(d, modes, "cuda", torch.bfloat16)
    samu_packed = pack_samu_parameters(samu_params)
    RGLRU = load_official_rglru(rglru_source)
    torch.manual_seed(1729)
    rg_model = RGLRU(width=d, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    rg_packed = pack_rglru(rg_model)
    for length in (128, 512, 2048, 8192, 32768, 65536):
        x = torch.randn(1, length, d, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        for warps in (2, 4, 8):
            timed(rows, {"scope": "micro_prefill", "model": "samu", "length": length,
                         "variant": "serial", "num_warps": warps},
                  lambda w=warps: samu_triton_serial(
                      x, samu_params, samu_packed, num_warps=w, return_cache=True
                  ), warmup_ms, rep_ms)
            timed(rows, {"scope": "micro_prefill", "model": "rglru", "length": length,
                         "variant": "serial", "num_warps": warps},
                  lambda w=warps: rglru_triton_serial(
                      x, positions, rg_packed, num_warps=w
                  ), warmup_ms, rep_ms)
            for chunk in (8, 16, 32):
                timed(rows, {"scope": "micro_prefill", "model": "samu", "length": length,
                             "variant": "chunk", "chunk_size": chunk, "num_warps": warps},
                      lambda c=chunk, w=warps: samu_triton_chunked(
                          x, samu_params, c, samu_packed, num_warps=w,
                          bounded_poly=samu_packed.bounded_poly_safe if length <= 128 else False,
                          return_cache=True,
                      ), warmup_ms, rep_ms)
                timed(rows, {"scope": "micro_prefill", "model": "rglru", "length": length,
                             "variant": "chunk", "chunk_size": chunk, "num_warps": warps},
                      lambda c=chunk, w=warps: rglru_triton_chunked(
                          x, positions, rg_packed, c, num_warps=w
                      ), warmup_ms, rep_ms)


def fused_decode_sweep(rows: list[dict], rglru_source: Path,
                       warmup_ms: int, rep_ms: int) -> None:
    d, modes = 128, 64
    samu_params = make_samu_parameters(d, modes, "cuda", torch.bfloat16)
    samu_packed = pack_samu_parameters(samu_params)
    RGLRU = load_official_rglru(rglru_source)
    torch.manual_seed(1729)
    rg_model = RGLRU(width=d, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    rg_packed = pack_rglru(rg_model)
    for batch in (1, 4, 16, 64):
        token = torch.randn(batch, d, device="cuda", dtype=torch.bfloat16)
        samu_state = (torch.randn(batch, modes, device="cuda"),
                      torch.randn(batch, modes, device="cuda"))
        positions = torch.ones(batch, 1, device="cuda", dtype=torch.long)
        rg_state = torch.randn(batch, d, device="cuda")
        for warps in (2, 4, 8):
            for block in (16, 32, 64):
                timed(rows, {"scope": "micro_fused_decode", "model": "samu",
                             "batch": batch, "block_m": block, "num_warps": warps},
                      lambda b=block, w=warps: samu_triton_decode(
                          token, samu_state, samu_params, samu_packed,
                          block_m=b, num_warps=w,
                      ), warmup_ms, rep_ms)
            timed(rows, {"scope": "micro_fused_decode", "model": "rglru",
                         "batch": batch, "num_warps": warps},
                  lambda w=warps: rglru_triton_decode(
                      token[:, None], positions, rg_packed, rg_state, num_warps=w
                  ), warmup_ms, rep_ms)


def split_decode_sweep(rows: list[dict], rglru_source: Path,
                       warmup_ms: int, rep_ms: int) -> None:
    d, modes = 2048, 1024
    samu_params = make_samu_parameters(d, modes, "cuda", torch.bfloat16, seed=5701)
    samu_packed = pack_samu_parameters(samu_params)
    RGLRU = load_official_rglru(rglru_source)
    torch.manual_seed(5701)
    rg_model = RGLRU(width=d, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
    rg_packed = pack_rglru(rg_model)
    for batch in (1, 16, 128):
        token = torch.randn(batch, d, device="cuda", dtype=torch.bfloat16)
        samu_state = (torch.randn(batch, modes, device="cuda"),
                      torch.randn(batch, modes, device="cuda"))
        positions = torch.ones(batch, 1, device="cuda", dtype=torch.long)
        rg_state = torch.randn(batch, d, device="cuda")
        for warps in (2, 4, 8):
            for block in (64, 128, 256):
                timed(rows, {"scope": "system_split_decode", "model": "samu",
                             "batch": batch, "block_m": block, "num_warps": warps},
                      lambda b=block, w=warps: samu_triton_decode_split(
                          token, samu_state, samu_packed, block_m=b, num_warps=w
                      ), warmup_ms, rep_ms)
            timed(rows, {"scope": "system_split_decode", "model": "rglru",
                         "batch": batch, "num_warps": warps},
                  lambda w=warps: rglru_triton_serial(
                      token[:, None], positions, rg_packed, rg_state, num_warps=w
                  ), warmup_ms, rep_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=50)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    stabilization = stabilize_gpu()
    print("stabilization", stabilization, flush=True)
    rows: list[dict] = []
    prefill_sweep(rows, args.rglru_source, args.warmup_ms, args.rep_ms)
    fused_decode_sweep(rows, args.rglru_source, args.warmup_ms, args.rep_ms)
    split_decode_sweep(rows, args.rglru_source, args.warmup_ms, args.rep_ms)
    props = torch.cuda.get_device_properties(0)
    atomic_json(args.output, {
        "schema_version": 1,
        "role": "calibration_only_not_final_reported_timings",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "hostname": platform.node(), "gpu": props.name,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        },
        "gpu_stabilization": stabilization,
        "rows": rows,
    })


if __name__ == "__main__":
    main()
