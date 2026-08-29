"""GPU analogue of Griffin Section 4.2 / Appendix D scan experiments.

Griffin Figure 8(a) fixes B=8 and token width=1024, then varies sequence
length.  This script keeps that shape but measures an H800, not a TPU-v3.  The
timed scope begins after the dense projection so it isolates the low-arithmetic
intensity scan/gate stage discussed in the paper.  SAMU receives D write
scalars plus two shared controls; RG-LRU receives D token scalars plus 2D
projected gates, following the pinned official equation.

The custom linear and chunk paths are production-like Triton kernels.  The
materialized associative paths are transparent PyTorch references and are
labelled as such; they are not claimed to be optimized GPU baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import triton
import torch.nn.functional as F

from kernels import _affine_scan_real2, load_official_rglru, make_samu_parameters
from run_equal_kernel_benchmarks import measure, stabilize_gpu
import triton_rglru as rg_impl
import triton_samu as samu_impl


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def timed(operation, warmup_ms: int, rep_ms: int) -> dict:
    result, compile_seconds = measure(operation, warmup_ms=warmup_ms, rep_ms=rep_ms)
    return {**result, "compile_seconds": compile_seconds}


def affine_scan_real(a: torch.Tensor, b: torch.Tensor):
    """Inclusive work-efficient tree scan for h=a*h_prev+b."""

    length = a.shape[-2]
    if length == 1:
        return a, b
    left_a, right_a = a[..., 0::2, :], a[..., 1::2, :]
    left_b, right_b = b[..., 0::2, :], b[..., 1::2, :]
    pairs = right_a.shape[-2]
    left_a, left_b = left_a[..., :pairs, :], left_b[..., :pairs, :]
    pair_a = right_a * left_a
    pair_b = right_a * left_b + right_b
    outer_a, outer_b = affine_scan_real(pair_a, pair_b)
    even_a = a[..., 0::2, :]
    even_b = b[..., 0::2, :]
    if even_a.shape[-2] > 1:
        prior_a, prior_b = outer_a[..., :-1, :], outer_b[..., :-1, :]
        next_a, next_b = a[..., 2::2, :], b[..., 2::2, :]
        tail_a = next_a * prior_a
        tail_b = next_a * prior_b + next_b
        even_a = torch.cat((a[..., :1, :], tail_a), dim=-2)
        even_b = torch.cat((b[..., :1, :], tail_b), dim=-2)
    inter_a = torch.stack((even_a[..., :pairs, :], outer_a), dim=-2).flatten(-3, -2)
    inter_b = torch.stack((even_b[..., :pairs, :], outer_b), dim=-2).flatten(-3, -2)
    if length % 2:
        inter_a = torch.cat((inter_a, even_a[..., -1:, :]), dim=-2)
        inter_b = torch.cat((inter_b, even_b[..., -1:, :]), dim=-2)
    return inter_a, inter_b


class SamuInputs:
    def __init__(self, batch: int, length: int, width: int):
        self.width = width
        self.modes = width // 2
        self.parameters = make_samu_parameters(width, self.modes, "cuda", torch.bfloat16, seed=5701)
        self.packed = samu_impl.pack_samu_parameters(self.parameters)
        torch.manual_seed(10000 + length)
        self.projected = torch.randn(
            batch, length, self.packed.padded_width,
            device="cuda", dtype=torch.bfloat16,
        )
        self.batch, self.length = batch, length

    def serial(self, warps: int = 4):
        out = torch.empty(
            self.batch, self.length, self.modes, 2,
            device="cuda", dtype=torch.bfloat16,
        )
        last_r = torch.empty(self.batch, self.modes, device="cuda")
        last_i = torch.empty_like(last_r)
        block = min(128, triton.next_power_of_2(self.modes))

        def operation():
            samu_impl._serial_prefill_kernel[(triton.cdiv(self.modes, block), self.batch)](
                self.projected, self.packed.nu, self.packed.cos_theta,
                self.packed.sin_theta, self.packed.gamma,
                out, last_r, last_i, self.length, modes=self.modes,
                packed_width=self.packed.padded_width, RETURN_CACHE=True,
                BLOCK_M=block, num_warps=warps,
                **samu_impl._launch_meta(self.packed),
            )
            return out, last_r, last_i

        return operation

    def chunk(self, chunk_size: int, warps: int, *, compressed_p: bool = False,
              rho_taylor: bool = False):
        chunks = self.length // chunk_size
        shape = (self.batch, chunks, self.modes)
        pr, pi, qr, qi = [torch.empty(shape, device="cuda") for _ in range(4)]
        in_r, in_i = torch.empty(shape, device="cuda"), torch.empty(shape, device="cuda")
        out = torch.empty(
            self.batch, self.length, self.modes, 2,
            device="cuda", dtype=torch.bfloat16,
        )
        last_r = torch.empty(self.batch, self.modes, device="cuda")
        last_i = torch.empty_like(last_r)
        block = min(128, triton.next_power_of_2(self.modes))
        grid = (triton.cdiv(self.modes, block), self.batch * chunks)
        use_compressed = compressed_p and not rho_taylor
        chunk_ct = self.packed.chunk_cos_theta[chunk_size] if use_compressed else self.packed.cos_theta
        chunk_st = self.packed.chunk_sin_theta[chunk_size] if use_compressed else self.packed.sin_theta
        meta = dict(
            length=self.length, modes=self.modes, chunks=chunks,
            packed_width=self.packed.padded_width, CHUNK=chunk_size,
            BLOCK_M=block, USE_COMPRESSED_P=use_compressed,
            num_warps=warps,
            **samu_impl._launch_meta(self.packed, False, rho_taylor),
        )

        def operation():
            samu_impl._chunk_summary_kernel[grid](
                self.projected, self.packed.nu, self.packed.cos_theta,
                self.packed.sin_theta, chunk_ct, chunk_st, self.packed.gamma,
                pr, pi, qr, qi, **meta,
            )
            samu_impl._chunk_prefix_kernel[(triton.cdiv(self.modes, block), self.batch)](
                pr, pi, qr, qi, in_r, in_i, modes=self.modes, chunks=chunks,
                last_r=last_r, last_i=last_i, RETURN_CACHE=True,
                BLOCK_M=block, num_warps=warps,
            )
            replay_meta = {key: value for key, value in meta.items() if key != "USE_COMPRESSED_P"}
            samu_impl._chunk_replay_kernel[grid](
                self.projected, self.packed.nu, self.packed.cos_theta,
                self.packed.sin_theta, self.packed.gamma, in_r, in_i, out,
                **replay_meta,
            )
            return out, last_r, last_i

        return operation

    def associative(self, dtype: torch.dtype):
        modes = self.modes

        def operation():
            projected = self.projected.to(dtype)
            raw_phase = projected[..., 2 * modes].float()
            raw_radial = projected[..., 2 * modes + 1].float()
            sp = torch.tanh(raw_phase + self.packed.phase_bias)
            sr = torch.tanh(raw_radial + self.packed.radial_bias)
            d = self.packed.phase_scale * sp
            c = self.packed.radial_scale * sr / (1.0 + sr.square())
            rho = torch.exp(-self.packed.nu * torch.exp(c).unsqueeze(-1)).to(dtype)
            cd, sd = torch.cos(d).unsqueeze(-1), torch.sin(d).unsqueeze(-1)
            cp = (self.packed.cos_theta * cd - self.packed.sin_theta * sd).to(dtype)
            si = (self.packed.sin_theta * cd + self.packed.cos_theta * sd).to(dtype)
            wr = projected[..., 0:2 * modes:2] * self.packed.gamma.to(dtype)
            wi = projected[..., 1:2 * modes:2] * self.packed.gamma.to(dtype)
            _, _, xr, xi = _affine_scan_real2(rho * cp, rho * si, wr, wi)
            return xr, xi

        return operation


class RGLRUInputs:
    def __init__(self, batch: int, length: int, width: int, RGLRU):
        self.width, self.batch, self.length = width, batch, length
        torch.manual_seed(5701)
        module = RGLRU(width=width, num_heads=16, device="cuda", dtype=torch.bfloat16).eval()
        self.packed = rg_impl.pack_rglru(module)
        torch.manual_seed(11000 + length)
        self.x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        self.projected = torch.randn(
            self.packed.num_heads, batch * length, 2 * self.packed.head_dim,
            device="cuda", dtype=torch.bfloat16,
        )
        self.positions = torch.ones(batch, length, device="cuda", dtype=torch.long)

    def serial(self, warps: int = 4):
        out = torch.empty_like(self.x)
        last = torch.empty(self.batch, self.width, device="cuda")
        block = triton.next_power_of_2(self.packed.head_dim)

        def operation():
            rg_impl._serial_scan_kernel[(self.packed.num_heads, self.batch)](
                self.x, self.projected, self.packed.gate_bias,
                self.packed.softplus_a, self.positions, last, out, last,
                self.length, self.batch * self.length,
                WIDTH=self.width, HEAD_DIM=self.packed.head_dim,
                HAS_H0=False, BLOCK=block, num_warps=warps,
            )
            return out, last

        return operation

    def chunk(self, chunk_size: int, warps: int):
        chunks = self.length // chunk_size
        shape = (self.batch, chunks, self.width)
        summary_a, summary_b = torch.empty(shape, device="cuda"), torch.empty(shape, device="cuda")
        chunk_input = torch.empty(shape, device="cuda")
        out = torch.empty_like(self.x)
        last = torch.empty(self.batch, self.width, device="cuda")
        block = triton.next_power_of_2(self.packed.head_dim)
        grid = (self.packed.num_heads, self.batch * chunks)
        common = dict(
            length=self.length, rows=self.batch * self.length, chunks=chunks,
            WIDTH=self.width, HEAD_DIM=self.packed.head_dim,
            CHUNK=chunk_size, BLOCK=block, num_warps=warps,
        )

        def operation():
            rg_impl._chunk_summary_kernel[grid](
                self.x, self.projected, self.packed.gate_bias,
                self.packed.softplus_a, self.positions,
                summary_a, summary_b, **common,
            )
            rg_impl._chunk_prefix_kernel[(self.packed.num_heads, self.batch)](
                summary_a, summary_b, last, chunk_input, last, chunks,
                WIDTH=self.width, HEAD_DIM=self.packed.head_dim,
                HAS_H0=False, BLOCK=block, num_warps=warps,
            )
            rg_impl._chunk_replay_kernel[grid](
                self.x, self.projected, self.packed.gate_bias,
                self.packed.softplus_a, self.positions,
                chunk_input, out, **common,
            )
            return out, last

        return operation

    def associative(self, dtype: torch.dtype):
        batch, length, width = self.batch, self.length, self.width
        heads, head_dim = self.packed.num_heads, self.packed.head_dim

        def operation():
            px = self.projected[..., :head_dim].permute(1, 0, 2).reshape(batch, length, width).to(dtype)
            pa = self.projected[..., head_dim:].permute(1, 0, 2).reshape(batch, length, width).to(dtype)
            bx = self.packed.gate_bias[..., :head_dim].reshape(width).to(dtype)
            ba = self.packed.gate_bias[..., head_dim:].reshape(width).to(dtype)
            softplus_a = self.packed.softplus_a.to(dtype)
            gate_x = torch.sigmoid(px + bx)
            gate_a = torch.sigmoid(pa + ba)
            log_a = -8.0 * gate_a * softplus_a
            a = torch.exp(log_a)
            multiplier = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_a), min=0.0))
            write = self.x.to(dtype) * gate_x * multiplier
            return affine_scan_real(a, write)[1]

        return operation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    RGLRU = load_official_rglru(args.rglru_source)
    stabilization = stabilize_gpu()
    rows = []
    for length in (2048, 4096, 8192, 16384):
        samu = SamuInputs(8, length, 1024)
        rg = RGLRUInputs(8, length, 1024, RGLRU)
        operations = {
            "samu_linear": samu.serial(4),
            "samu_chunk32": samu.chunk(32, 2),
            "samu_chunk32_spectral_taylor": samu.chunk(32, 2, rho_taylor=True),
            "samu_chunk32_compressed_transition": samu.chunk(32, 2, compressed_p=True),
            "rglru_linear": rg.serial(4),
            "rglru_chunk32": rg.chunk(32, 4),
        }
        # Associative references materialize O(BLD) transitions and keep every
        # tree level live.  Limit them to 8K to avoid turning OOM into a timing.
        if length <= 8192:
            operations.update({
                "samu_associative_bf16_reference": samu.associative(torch.bfloat16),
                "samu_associative_fp32_reference": samu.associative(torch.float32),
                "rglru_associative_bf16_reference": rg.associative(torch.bfloat16),
                "rglru_associative_fp32_reference": rg.associative(torch.float32),
            })
        for order_name, names in (("forward", tuple(operations)),
                                  ("reverse", tuple(reversed(tuple(operations))))):
            stabilize_gpu(seconds=0.5)
            for name in names:
                base = {
                    "method": name, "batch": 8, "length": length,
                    "real_state_scalars": 1024, "measurement_order": order_name,
                    "scope": "post-projection forward scan/gate stage",
                }
                try:
                    result = timed(operations[name], 20, args.rep_ms)
                    rows.append({**base, "status": "measured", **result})
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    rows.append({**base, "status": "unsupported", "error": repr(error)})
                    torch.cuda.empty_cache()
        del samu, rg, operations
        torch.cuda.empty_cache()

    summary = []
    keys = sorted({(row["method"], row["length"]) for row in rows if row["status"] == "measured"})
    for method, length in keys:
        values = [row["median_ms"] for row in rows
                  if row["status"] == "measured" and row["method"] == method and row["length"] == length]
        summary.append({
            "method": method, "length": length,
            "order_balanced_median_ms": statistics.mean(values),
            "samples_from_orders": len(values),
        })
    props = torch.cuda.get_device_properties(0)
    atomic_json(args.output, {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "figure": "Griffin Figure 8(a), Section 4.2 and Appendix D.2",
            "matched_shape": "batch 8, 1024 real state scalars, sequence length sweep",
            "hardware_difference": "H800 PCIe GPU versus the paper's two-chip TPU-v3 pod",
            "timed_scope": "post-projection forward scan/gate stage",
            "not_measured": [
                "multi-device model-parallel all-reduce (only one H800 is available)",
                "ZeRO optimizer sharding (only one H800 is available)",
                "full backward/training step; current custom Triton paths are inference-forward kernels",
            ],
        },
        "environment": {
            "hostname": platform.node(), "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "rglru_commit": RGLRU_COMMIT,
        },
        "static_input_scalars_per_token": {
            "samu": 1026,
            "rglru": 3072,
            "explanation": "SAMU consumes 1024 write scalars plus two shared controls; RG-LRU consumes the 1024 token values plus 2048 projected gate values",
        },
        "gpu_stabilization": stabilization,
        "rows": rows,
        "summary": summary,
    })


if __name__ == "__main__":
    main()
