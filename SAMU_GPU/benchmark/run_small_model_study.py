"""Train and benchmark matched small Hawk-style SAMU and RG-LRU models.

This is a scaled, fully trained companion to the Griffin Section 4/5 study.
It keeps the official recurrent-block layout (two input branches, causal
depthwise Conv1D, recurrent mixer, multiplicative join and gated MLP).  The
only architecture switch is the recurrent mixer.  SAMU uses the existing
``linear_x`` output as its complex write, instead of adding the extra dense
D_RNN x D_RNN projection used by the isolated canonical microbenchmark.

Training uses the same data windows, initialization of shared tensors,
optimizer, schedule, token budget and validation batches for both methods.
The complete-step timing axis fixes tokens/step while varying sequence length,
as in Griffin Figure 3.  Decode measures complete autoregressive trajectories
with CUDA Graph replay, as in Griffin Section 5.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from triton_training_scan import (
    complex_scan,
    complex_scan_with_last,
    real_chunk_scan,
    real_scan,
    real_scan_with_last,
    samu_scan,
    samu_tiled_serial_scan,
    samu_scan_precomputed_decay,
    samu_chunk_scan,
    samu_training_scan_dispatch,
    samu_scan_with_last,
    samu_decode,
    samu_decode_blocked,
)
from triton_fused_rglru_training import fused_rglru_scan
from triton_samu_controller import (
    formal_samu_controller_recompute,
    formal_samu_coordinates,
    formal_samu_coordinates_forward_cache,
    reference_controller_projection_recompute,
    reference_forward_triton_backward,
    shared_controller_projection,
)
from triton_samu_affine_warp_scan import samu_affine_tile_scan
from triton_rglru import pack_rglru, rglru_triton_decode, rglru_triton_serial


OFFICIAL_RECURRENTGEMMA_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


@dataclass(frozen=True)
class StudyConfig:
    vocab_size: int
    width: int = 256
    rnn_width: int = 384
    depth: int = 6
    mlp_expansion: int = 3
    num_gate_blocks: int = 16
    conv_width: int = 4
    train_sequence_length: int = 256
    train_batch_size: int = 32
    # Griffin/Hawk paper presets use no sqrt(D) embedding scaling and apply
    # the 2/N variance factor to every residual branch's final projection.
    embedding_scale_by_sqrt_dim: bool = False
    final_w_init_variance_scale: float | None = None

    @property
    def mlp_width(self) -> int:
        return self.width * self.mlp_expansion

    @property
    def resolved_final_w_init_variance_scale(self) -> float:
        return (
            2.0 / self.depth
            if self.final_w_init_variance_scale is None
            else self.final_w_init_variance_scale
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        normalized = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * (1.0 + self.scale)


class SqrtBoundDerivative(torch.autograd.Function):
    """Official RG-LRU square root with its BF16-safe clipped derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        clipped = torch.clamp(4.0 * x, min=1.0 / 1_000_000.0)
        return grad_output / torch.sqrt(clipped)


class CausalDepthwiseConv1D(nn.Module):
    def __init__(self, width: int, temporal_width: int):
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w = nn.Parameter(torch.empty(temporal_width, width))
        self.b = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.w, std=math.sqrt(0.01 / temporal_width))

    def forward(self, x, return_cache: bool = False):
        channels = x.transpose(1, 2)
        padded = F.pad(channels, (self.temporal_width - 1, 0))
        output = F.conv1d(
            padded, self.w.transpose(0, 1).unsqueeze(1), self.b,
            groups=self.width,
        ).transpose(1, 2)
        cache = x[:, 1 - self.temporal_width:].contiguous() if return_cache else None
        return output, cache

    def step(self, x, cache):
        combined = torch.cat((cache, x[:, None]), dim=1)
        output = (combined * self.w[None]).sum(dim=1) + self.b
        # Elementwise reduction is promoted to FP32 by CUDA autocast, whereas
        # the full-sequence grouped conv returns BF16.  Restore the official
        # BF16 activation boundary so prefill and one-token decode use the
        # same numerical policy; recurrence caches remain explicitly FP32.
        return output.to(x.dtype), combined[:, 1:].contiguous()


class BlockDiagonalLinear(nn.Module):
    """Official RecurrentGemma block-diagonal orientation and initialization."""

    def __init__(self, width: int, num_blocks: int):
        super().__init__()
        if width % num_blocks:
            raise ValueError("rnn_width must be divisible by num_gate_blocks")
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.w = nn.Parameter(torch.empty(num_blocks, self.block_width, self.block_width))
        self.b = nn.Parameter(torch.zeros(num_blocks, self.block_width))
        nn.init.normal_(self.w, std=math.sqrt(1.0 / self.block_width))

    def forward(self, x):
        shape = x.shape
        blocks = x.reshape(*shape[:-1], self.num_blocks, self.block_width)
        output = torch.einsum("...hi,hij->...hj", blocks, self.w) + self.b
        return output.reshape(shape)


def _official_a_parameter(width: int) -> torch.Tensor:
    radius_squared = torch.empty(width).uniform_(0.9**2 + 1e-8, 0.999**2 + 1e-8)
    log_radius = 0.5 * torch.log(radius_squared)
    return torch.log(torch.exp(-log_radius) - 1.0)


class RGLRUMixer(nn.Module):
    """Official RG-LRU equations with a CUDA custom-VJP scan."""

    def __init__(self, width: int, num_blocks: int):
        super().__init__()
        self.width = width
        self.input_gate = BlockDiagonalLinear(width, num_blocks)
        self.a_gate = BlockDiagonalLinear(width, num_blocks)
        self.a_param = nn.Parameter(_official_a_parameter(width))
        self._inference_packed = None
        self._decode_backend = "fused"
        self._decode_num_warps = 2
        self._fused_training_block_size = 64
        self.scan_backend = "auto"

    def _terms(self, x):
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        multiplier = SqrtBoundDerivative.apply((1.0 - torch.exp(2.0 * log_a)).clamp_min(0.0))
        write = x * gate_x * multiplier.to(x.dtype)
        # Every sampled training sequence is an independent document.  This is
        # the official segment_pos==0 behavior: no previous state and no gamma
        # normalization on the first token.
        a = torch.cat((torch.zeros_like(a[:, :1]), a[:, 1:]), dim=1)
        write = torch.cat((x[:, :1] * gate_x[:, :1], write[:, 1:]), dim=1)
        return a.to(x.dtype), write.to(x.dtype)

    def forward(self, x, return_cache: bool = False):
        backend = self.scan_backend
        fused_block_size = self._fused_training_block_size
        if backend == "auto":
            batch, length, width = x.shape
            # The O(C) grouped affine hierarchy is profitable once there are
            # enough K=16 chunks.  Keep the lower-launch chunk path for small
            # or non-production shapes.  Block choices are H800 Gate-3 results.
            if x.dtype == torch.bfloat16 and length >= 2048:
                backend = "fused_grouped_prefix32_hierarchical_chunk16"
                if length >= 16384:
                    fused_block_size = 128 if width >= 1536 else 256
                else:
                    fused_block_size = 256 if width > 2048 else 128
            else:
                backend = "fused_serial" if length < 16 else "fused_chunk16"
        hierarchical_prefix = (
            backend.startswith("fused_hierarchical_chunk")
            or (backend.startswith("fused_grouped_prefix")
                and "hierarchical_chunk" in backend)
        )
        if (backend == "fused_serial" or backend.startswith("fused_chunk")
                or hierarchical_prefix):
            gate_x_logit = self.input_gate(x)
            gate_a_logit = self.a_gate(x)
            chunk_size = 0 if backend == "fused_serial" else int(
                backend.rsplit("chunk", 1)[1]
            )
            prefix_group_size = next(
                (size for size in (32, 64, 128)
                 if f"grouped_prefix{size}" in backend),
                0,
            )
            output, last = fused_rglru_scan(
                x,
                gate_x_logit,
                gate_a_logit,
                self.a_param,
                chunk_size=chunk_size,
                block_size=fused_block_size,
                reset_first=True,
                hierarchical_prefix=hierarchical_prefix,
                prefix_group_size=prefix_group_size,
            )
            return output, last if return_cache else None
        a, write = self._terms(x)
        if return_cache:
            output, cache = real_scan_with_last(a, write)
        elif backend == "triton":
            if not return_cache:
                output, cache = real_scan(a, write), None
        elif backend == "framework_eager":
            output = eager_real_scan(a, write)
            cache = output[:, -1].float() if return_cache else None
        elif backend.startswith("chunk"):
            output = real_chunk_scan(
                a, write, int(backend.removeprefix("chunk"))
            )
            cache = None
        elif backend in ("associative_bf16", "associative_fp32"):
            scan_dtype = (
                torch.bfloat16
                if backend == "associative_bf16"
                else torch.float32
            )
            output = associative_real_scan(
                a.to(scan_dtype), write.to(scan_dtype)
            ).to(x.dtype)
            cache = output[:, -1].float() if return_cache else None
        else:
            raise ValueError(f"unknown scan backend: {backend}")
        return output, cache

    def set_scan_backend(self, backend: str) -> None:
        self.scan_backend = backend

    def set_fused_training_block_size(self, block_size: int) -> None:
        if block_size not in (32, 64, 128, 256):
            raise ValueError(block_size)
        self._fused_training_block_size = block_size

    def prepare_inference(self):
        self._inference_packed = pack_rglru(self)

    def set_decode_num_warps(self, num_warps: int) -> None:
        if num_warps not in (1, 2, 4, 8):
            raise ValueError(num_warps)
        self._decode_num_warps = num_warps

    def set_decode_backend(self, backend: str) -> None:
        if backend not in ("fused", "bmm"):
            raise ValueError(backend)
        self._decode_backend = backend

    def step(self, x, state, segment_pos):
        if self._inference_packed is not None:
            if self._decode_backend == "bmm":
                output, new_state = rglru_triton_serial(
                    x[:, None], segment_pos[:, None], self._inference_packed,
                    state, num_warps=self._decode_num_warps,
                )
            else:
                output, new_state = rglru_triton_decode(
                    x, segment_pos, self._inference_packed, state,
                    num_warps=self._decode_num_warps,
                )
            return output[:, 0], new_state
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        multiplier = torch.sqrt((1.0 - torch.exp(2.0 * log_a)).clamp_min(0.0))
        write = x * gate_x * multiplier.to(x.dtype)
        reset = segment_pos.eq(0).unsqueeze(-1)
        a = torch.where(reset, torch.zeros_like(a), a)
        write = torch.where(reset, x * gate_x, write)
        new_state = a.float() * state + write.float()
        return new_state.to(x.dtype), new_state


class SAMUMixer(nn.Module):
    """Canonical G=1 SAMU whose write is supplied by the outer linear_x."""

    def __init__(self, width: int):
        super().__init__()
        if width % 2:
            raise ValueError("SAMU rnn_width must be even")
        self.width = width
        self.modes = width // 2
        # Canonical RTU/SAMU initialization.  The recurrent radius is
        # exp(-nu), so drawing radius_squared uniformly gives
        # nu=-0.5*log(U).  Phases cover the full circle.  This must not be
        # replaced by the long-memory distribution used in inference-only
        # timing fixtures.
        radius_squared = torch.rand(self.modes).clamp_min(torch.finfo(torch.float32).tiny)
        nu = -0.5 * torch.log(radius_squared)
        theta = (2.0 * math.pi * torch.rand(self.modes)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        self.nu_log = nn.Parameter(torch.log(nu))
        self.theta_log = nn.Parameter(torch.log(theta))
        self.phase_direction = nn.Parameter(torch.randn(width + 1))
        self.radial_direction = nn.Parameter(torch.randn(width + 1))
        # Zero amplitude is the canonical RTU working null.  Ordinary BPTT
        # still gives the amplitudes a first-order gradient while the two
        # directions remain inactive until their amplitude becomes nonzero.
        self.phase_amplitude = nn.Parameter(torch.zeros(()))
        self.radial_amplitude = nn.Parameter(torch.zeros(()))
        self._inference_fused = False
        self._decode_backend = "auto"
        self._inference_pack = None
        self._tiled_training_block_size = 128
        # Exact FP32 projection with fused low-rank coordinate backward is the
        # latency default. The reference and cache-saving variants remain
        # explicit debugging / memory-pressure choices.
        self._controller_projection_dtype = "fp32_fused_coords"
        # H800 dispatch measured on complete steps: the state-stationary scan
        # wins for short, batch-rich sequences; the exact 16-token chunk scan
        # wins once the time axis is long enough to supply the missing
        # parallelism. Cache-producing prefill always uses the serial kernel.
        self.scan_backend = "auto"

    def _controls(self, x):
        # Both shared coordinates are one two-column projection.  This is
        # exactly equivalent to appending a constant 1 and applying the two
        # normalized direction vectors separately, while avoiding the
        # augmented [B,L,D+1] tensor and a second projection launch.
        if self._controller_projection_dtype == "fp32_cache_recompute":
            return formal_samu_controller_recompute(
                x, self.phase_direction, self.radial_direction,
                self.phase_amplitude, self.radial_amplitude, self.modes,
            )
        directions = torch.stack((self.phase_direction, self.radial_direction))
        directions = F.normalize(directions.float(), dim=1)
        if self._controller_projection_dtype in (
                "fp32_triton_backward", "fp32_cache_save"):
            projected = reference_forward_triton_backward(
                x,
                directions[:, :-1].contiguous(),
                directions[:, -1].contiguous(),
            )
        elif self._controller_projection_dtype == "fp32_recompute":
            projected = reference_controller_projection_recompute(
                x,
                directions[:, :-1].contiguous(),
                directions[:, -1].contiguous(),
            )
        elif self._controller_projection_dtype in (
                "triton_fp32", "triton_fp32_rounded"):
            projected = shared_controller_projection(
                x,
                directions[:, :-1].contiguous(),
                directions[:, -1].contiguous(),
            )
            if self._controller_projection_dtype == "triton_fp32_rounded":
                projected = projected.to(x.dtype).float()
        elif self._controller_projection_dtype == "bf16":
            # The controller has only two outputs.  Casting its tiny normalized
            # weight/bias avoids a logical full-size x.float() allocation and
            # lets the projection use the production BF16 activation boundary.
            # Keep the two projected controls in FP32 for their nonlinear math.
            projected = F.linear(
                x,
                directions[:, :-1].to(x.dtype),
                directions[:, -1].to(x.dtype),
            ).float()
        else:
            projected = F.linear(
                x.float(), directions[:, :-1], directions[:, -1]
            )
            if self._controller_projection_dtype == "fp32_rounded":
                projected = projected.to(x.dtype).float()
        if self._controller_projection_dtype in (
                "fp32_cache_save", "fp32_fused_coords"):
            return formal_samu_coordinates(
                projected, self.phase_amplitude, self.radial_amplitude,
                self.modes,
            )
        phase_coordinate = torch.tanh(projected[..., 0])
        radial_raw = torch.tanh(projected[..., 1])
        radial_coordinate = radial_raw / (1.0 + radial_raw.square())
        scale = 1.0 / math.sqrt(self.modes)
        delta = scale * torch.tanh(self.phase_amplitude) * phase_coordinate
        eta = scale * torch.tanh(self.radial_amplitude) * radial_coordinate
        return eta, delta

    def _scan_inputs(self, x):
        eta, delta = self._controls(x)
        nu = torch.exp(self.nu_log)
        # canonical_grouped_samu_math.base_geometry adds epsilon after the
        # square root.  Keeping it outside is part of the SAMU equation.
        gamma = torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8
        write_r = x[..., :self.modes].float() * gamma
        write_i = x[..., self.modes:].float() * gamma
        return eta, delta, write_r.to(x.dtype), write_i.to(x.dtype)

    def forward(self, x, return_cache: bool = False):
        backend = self.scan_backend
        if backend == "auto":
            batch, length, width = x.shape
            if x.dtype == torch.bfloat16 and not return_cache:
                if batch >= 4 and length <= 2048:
                    backend = "fused_output_fused_write_shared_sfu_chunk32"
                elif batch == 1 and length >= 65536:
                    backend = (
                        "fused_output_fused_write_shared_sfu_"
                        "serial_forward_prefix_grouped_prefix64_"
                        "hierarchical_chunk32"
                    )
                elif batch == 1 and (
                    length >= 16384 or (length >= 8192 and width >= 2048)
                ):
                    backend = (
                        "fused_output_fused_write_shared_sfu_"
                        "grouped_prefix64_hierarchical_chunk32"
                    )
                elif batch == 1 and length >= 4096:
                    backend = "fused_output_fused_write_shared_sfu_chunk32"
                else:
                    backend = "chunk16" if length >= 1024 else "triton"
            else:
                backend = "chunk16" if length >= 1024 else "triton"
        fused_write = "fused_write" in backend and not return_cache
        fused_controller_backward = (
            "fused_controller_bwd" in backend and fused_write
        )
        controller_backward_cache = None
        packed_output = None
        if fused_write:
            # The chunk kernels generate gamma*x in registers and return its
            # exact input/nu gradients, so neither state-sized write tensor is
            # created in HBM on this path.
            if fused_controller_backward:
                # The recurrent autograd function owns the low-rank
                # controller backward, so do not retain x.float() or a second
                # state-sized controller input-gradient buffer.
                with torch.no_grad():
                    directions = F.normalize(
                        torch.stack((self.phase_direction,
                                     self.radial_direction)).float(),
                        dim=1,
                    )
                    projected = F.linear(
                        x.float(), directions[:, :-1], directions[:, -1]
                    )
                    eta, delta, phase_coordinate, radial_raw = (
                        formal_samu_coordinates_forward_cache(
                            projected, self.phase_amplitude,
                            self.radial_amplitude, self.modes,
                        )
                    )
                controller_backward_cache = (
                    self.phase_direction, self.radial_direction,
                    self.phase_amplitude, self.radial_amplitude,
                    phase_coordinate, radial_raw,
                )
            else:
                eta, delta = self._controls(x)
            write_r = write_i = x.new_empty((0,))
        else:
            eta, delta, write_r, write_i = self._scan_inputs(x)
        if return_cache:
            if self.modes <= 512:
                out_r, out_i, last_r, last_i = samu_scan_with_last(
                    eta, delta, self.nu_log, self.theta_log, write_r, write_i
                )
            else:
                out_r, out_i, last_r, last_i = samu_tiled_serial_scan(
                    eta, delta, self.nu_log, self.theta_log, write_r, write_i,
                    block_size=self._tiled_training_block_size,
                )
            cache = (last_r, last_i)
        elif backend.startswith("affine_tile_s"):
            configuration = backend.removeprefix("affine_tile_s")
            steps_text, mode_and_warps = configuration.split("_m", 1)
            mode_text, warps_text = mode_and_warps.split("_w", 1)
            out_r, out_i, _, _ = samu_affine_tile_scan(
                eta, delta, self.nu_log, self.theta_log, write_r, write_i,
                steps=int(steps_text), mode_block=int(mode_text),
                num_warps=int(warps_text),
            )
            cache = None
        elif backend == "tiled_serial":
            out_r, out_i, _, _ = samu_tiled_serial_scan(
                eta, delta, self.nu_log, self.theta_log, write_r, write_i,
                block_size=self._tiled_training_block_size,
            )
            cache = None
        elif backend == "triton":
            if self.modes <= 512:
                out_r, out_i = samu_scan(
                    eta, delta, self.nu_log, self.theta_log, write_r, write_i
                )
                cache = None
            else:
                # The fully fused serial kernel keeps all modes in one
                # program and is intentionally capped at 512 modes.  At
                # paper widths, materialize the exact transition once and
                # use the tiled state-stationary complex scan.  This is the
                # conservative wide-mode analogue of the Pallas linear scan;
                # the primary H800 path remains the exact chunk scan.
                nu, theta = torch.exp(self.nu_log), torch.exp(self.theta_log)
                radius = torch.exp(-nu * torch.exp(eta).unsqueeze(-1))
                phase = theta + delta.unsqueeze(-1)
                out_r, out_i = complex_scan(
                    radius * torch.cos(phase),
                    radius * torch.sin(phase),
                    write_r,
                    write_i,
                )
                cache = None
        elif backend == "precomputed_decay":
            out_r, out_i = samu_scan_precomputed_decay(
                eta, delta, self.nu_log, self.theta_log, write_r, write_i
            )
            cache = None
        elif "chunk" in backend and backend.rsplit("chunk", 1)[1].isdigit():
            hierarchical_prefix = "hierarchical" in backend
            precompute_shared = "shared_sfu" in backend
            compressed_transition = "compressed" in backend
            atomic_shared = "atomic" in backend
            two_stage_spectral = "spectral2" in backend
            precompute_spectral = "spectral_cache" in backend
            backward_chunk_group = 2 if "replayg2" in backend else 1
            fused_output_relu = "fused_output" in backend
            prefix_group_size = next(
                (size for size in (32, 64, 128)
                 if f"grouped_prefix{size}" in backend),
                0,
            )
            if "serial_forward_prefix" in backend and prefix_group_size:
                prefix_group_size = -prefix_group_size
            compact_control_cache = "cache" in self._controller_projection_dtype
            chunk_size = int(backend.rsplit("chunk", 1)[1])
            scan_output = samu_training_scan_dispatch(
                eta, delta, self.nu_log, self.theta_log,
                write_r, write_i, chunk_size=chunk_size,
                hierarchical_prefix=hierarchical_prefix,
                precompute_shared=precompute_shared,
                compressed_transition=compressed_transition,
                atomic_shared=atomic_shared,
                two_stage_spectral=two_stage_spectral,
                compact_control_cache=compact_control_cache,
                precompute_spectral=precompute_spectral,
                raw_x=x if fused_write else None,
                fused_write=fused_write,
                backward_chunk_group=backward_chunk_group,
                fused_output_relu=fused_output_relu,
                prefix_group_size=prefix_group_size,
                controller_backward_cache=controller_backward_cache,
                reset_first=True,
                block_size=self._tiled_training_block_size,
            )
            if fused_output_relu:
                packed_output = scan_output
            else:
                out_r, out_i = scan_output
            cache = None
        else:
            nu, theta = torch.exp(self.nu_log), torch.exp(self.theta_log)
            radius = torch.exp(-nu * torch.exp(eta).unsqueeze(-1))
            phase = theta + delta.unsqueeze(-1)
            ar, ai = radius * torch.cos(phase), radius * torch.sin(phase)
            if backend == "framework_eager":
                out_r, out_i = eager_complex_scan(
                    ar, ai, write_r, write_i
                )
            elif backend in ("associative_bf16", "associative_fp32"):
                scan_dtype = (
                    torch.bfloat16
                    if backend == "associative_bf16"
                    else torch.float32
                )
                out_r, out_i = associative_complex_scan(
                    ar.to(scan_dtype), ai.to(scan_dtype),
                    write_r.to(scan_dtype), write_i.to(scan_dtype),
                )
                out_r, out_i = out_r.to(x.dtype), out_i.to(x.dtype)
            else:
                raise ValueError(f"unknown scan backend: {backend}")
            cache = (
                (out_r[:, -1].float(), out_i[:, -1].float())
                if return_cache else None
            )
        # Canonical SAMU/RTU layers expose the concatenated real state through
        # ReLU before the surrounding multiplicative output branch.
        output = (packed_output if packed_output is not None
                  else torch.cat((out_r, out_i), dim=-1).relu())
        return output, cache

    def set_scan_backend(self, backend: str) -> None:
        self.scan_backend = backend

    def set_tiled_training_block_size(self, block_size: int) -> None:
        if block_size not in (32, 64, 128, 256):
            raise ValueError(block_size)
        self._tiled_training_block_size = block_size

    def set_controller_projection_dtype(self, dtype: str) -> None:
        if dtype not in (
            "fp32", "bf16", "triton_fp32",
            "fp32_rounded", "triton_fp32_rounded", "fp32_recompute",
            "fp32_triton_backward", "fp32_cache_save",
            "fp32_cache_recompute", "fp32_fused_coords",
        ):
            raise ValueError(dtype)
        self._controller_projection_dtype = dtype

    def prepare_inference(self):
        self._inference_fused = True
        with torch.no_grad():
            nu = torch.exp(self.nu_log.float()).contiguous()
            theta = torch.exp(self.theta_log.float())
            scale = 1.0 / math.sqrt(self.modes)
            self._inference_pack = (
                F.normalize(self.phase_direction.float(), dim=0).contiguous(),
                F.normalize(self.radial_direction.float(), dim=0).contiguous(),
                (scale * torch.tanh(self.phase_amplitude.float())).reshape(1),
                (scale * torch.tanh(self.radial_amplitude.float())).reshape(1),
                nu,
                torch.cos(theta).contiguous(),
                torch.sin(theta).contiguous(),
                (torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8).contiguous(),
            )

    def set_decode_backend(self, backend: str):
        if backend not in ("auto", "fused", "packed", "blocked16", "blocked32", "blocked64"):
            raise ValueError(backend)
        self._decode_backend = backend

    def step(self, x, state, segment_pos):
        del segment_pos
        if self._inference_fused:
            backend = self._decode_backend
            if backend == "auto":
                backend = "packed" if x.shape[0] == 16 else "fused"
            if backend == "packed":
                return samu_decode_blocked(
                    x, *self._inference_pack, state[0], state[1],
                    block_m=1 << (self.modes - 1).bit_length(),
                )
            if backend.startswith("blocked"):
                return samu_decode_blocked(
                    x, *self._inference_pack, state[0], state[1],
                    block_m=int(backend.removeprefix("blocked")),
                )
            return samu_decode(
                x, self.phase_direction, self.radial_direction,
                self.phase_amplitude, self.radial_amplitude,
                self.nu_log, self.theta_log, state[0], state[1],
            )
        eta, delta, write_r, write_i = self._scan_inputs(x[:, None])
        nu, theta = torch.exp(self.nu_log), torch.exp(self.theta_log)
        radius = torch.exp(-nu * torch.exp(eta[:, 0]).unsqueeze(-1))
        phase = theta + delta[:, 0].unsqueeze(-1)
        ar, ai = radius * torch.cos(phase), radius * torch.sin(phase)
        write_r, write_i = write_r[:, 0].float(), write_i[:, 0].float()
        old_r, old_i = state
        new_r = ar * old_r - ai * old_i + write_r
        new_i = ai * old_r + ar * old_i + write_i
        return torch.cat((new_r, new_i), -1).relu().to(x.dtype), (new_r, new_i)


class GatedMLP(nn.Module):
    def __init__(
        self,
        width: int,
        expanded_width: int,
        final_w_init_variance_scale: float = 1.0,
    ):
        super().__init__()
        self.up = nn.Linear(width, 2 * expanded_width)
        self.down = nn.Linear(expanded_width, width)
        nn.init.normal_(self.up.weight, std=math.sqrt(1.0 / width))
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(
            self.down.weight,
            std=math.sqrt(final_w_init_variance_scale / expanded_width),
        )
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.gelu(gate, approximate="tanh") * value)


class RecurrentBlock(nn.Module):
    def __init__(self, config: StudyConfig, architecture: str):
        super().__init__()
        self.architecture = architecture
        self.temporal_norm = RMSNorm(config.width)
        self.linear_y = nn.Linear(config.width, config.rnn_width)
        self.linear_x = nn.Linear(config.width, config.rnn_width)
        self.conv = CausalDepthwiseConv1D(config.rnn_width, config.conv_width)
        self.mixer = (
            RGLRUMixer(config.rnn_width, config.num_gate_blocks)
            if architecture == "rglru" else SAMUMixer(config.rnn_width)
        )
        self.linear_out = nn.Linear(config.rnn_width, config.width)
        self.channel_norm = RMSNorm(config.width)
        self.mlp = GatedMLP(
            config.width,
            config.mlp_width,
            config.resolved_final_w_init_variance_scale,
        )
        for linear in (self.linear_y, self.linear_x):
            nn.init.normal_(linear.weight, std=math.sqrt(1.0 / config.width))
            nn.init.zeros_(linear.bias)
        nn.init.normal_(
            self.linear_out.weight,
            std=math.sqrt(
                config.resolved_final_w_init_variance_scale / config.rnn_width
            ),
        )
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, return_cache: bool = False):
        raw = x
        normalized = self.temporal_norm(x)
        y = F.gelu(self.linear_y(normalized), approximate="tanh")
        recurrent_input = self.linear_x(normalized)
        recurrent_input, conv_cache = self.conv(recurrent_input, return_cache)
        recurrent, recurrent_cache = self.mixer(recurrent_input, return_cache)
        x = self.linear_out(recurrent * y) + raw
        x = self.mlp(self.channel_norm(x)) + x
        return x, (conv_cache, recurrent_cache) if return_cache else None

    def step(self, x, cache, segment_pos):
        raw = x
        normalized = self.temporal_norm(x)
        y = F.gelu(self.linear_y(normalized), approximate="tanh")
        recurrent_input = self.linear_x(normalized)
        recurrent_input, conv_cache = self.conv.step(recurrent_input, cache[0])
        recurrent, recurrent_cache = self.mixer.step(
            recurrent_input, cache[1], segment_pos
        )
        x = self.linear_out(recurrent * y) + raw
        x = self.mlp(self.channel_norm(x)) + x
        return x, (conv_cache, recurrent_cache)


class SmallHawkLM(nn.Module):
    def __init__(self, config: StudyConfig, architecture: str):
        super().__init__()
        if architecture not in ("samu", "rglru"):
            raise ValueError(architecture)
        self.config = config
        self.architecture = architecture
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        nn.init.normal_(self.embedding.weight, std=math.sqrt(1.0 / config.width))
        self.blocks = nn.ModuleList([
            RecurrentBlock(config, architecture) for _ in range(config.depth)
        ])
        self.final_norm = RMSNorm(config.width)

    def forward(self, tokens, return_cache: bool = False):
        x = self.embedding(tokens)
        if self.config.embedding_scale_by_sqrt_dim:
            x = x * math.sqrt(self.config.width)
        caches = []
        for block in self.blocks:
            x, cache = block(x, return_cache)
            if return_cache:
                caches.append(cache)
        x = self.final_norm(x)
        logits = x @ self.embedding.weight.transpose(0, 1)
        return (logits, caches) if return_cache else logits

    def init_cache(self, batch: int, device, dtype):
        caches = []
        for _ in self.blocks:
            conv = torch.zeros(
                batch, self.config.conv_width - 1, self.config.rnn_width,
                device=device, dtype=dtype,
            )
            if self.architecture == "rglru":
                recurrent = torch.zeros(batch, self.config.rnn_width, device=device, dtype=torch.float32)
            else:
                modes = self.config.rnn_width // 2
                recurrent = (
                    torch.zeros(batch, modes, device=device, dtype=torch.float32),
                    torch.zeros(batch, modes, device=device, dtype=torch.float32),
                )
            caches.append((conv, recurrent))
        return caches

    def prepare_inference(self):
        for block in self.blocks:
            block.mixer.prepare_inference()

    def step(self, tokens, caches, segment_pos):
        x = self.embedding(tokens)
        if self.config.embedding_scale_by_sqrt_dim:
            x = x * math.sqrt(self.config.width)
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block.step(x, cache, segment_pos)
            new_caches.append(new_cache)
        logits = self.final_norm(x) @ self.embedding.weight.transpose(0, 1)
        return logits, new_caches


def copy_shared_parameters(reference: SmallHawkLM, target: SmallHawkLM):
    source, destination = reference.state_dict(), target.state_dict()
    copied, count = [], 0
    for name, value in destination.items():
        candidate = source.get(name)
        if candidate is None or candidate.shape != value.shape or ".mixer." in name:
            continue
        destination[name] = candidate.detach().clone()
        copied.append(name)
        count += value.numel()
    target.load_state_dict(destination)
    return {"tensors": len(copied), "parameters": count}


def build_model(config: StudyConfig, architecture: str, seed: int):
    set_seed(seed)
    reference = SmallHawkLM(config, "rglru")
    if architecture == "rglru":
        return reference, {"reference": "all"}
    set_seed(seed ^ 0x5A17)
    model = SmallHawkLM(config, "samu")
    copied = copy_shared_parameters(reference, model)
    del reference
    return model, copied


def load_character_data(path: Path):
    raw = path.read_text("utf-8")
    alphabet = sorted(set(raw))
    lookup = {character: index for index, character in enumerate(alphabet)}
    encoded = torch.tensor([lookup[c] for c in raw], dtype=torch.long)
    train_end, validation_end = int(0.9 * len(encoded)), int(0.95 * len(encoded))
    return {
        "alphabet": alphabet,
        "train": encoded[:train_end],
        "validation": encoded[train_end:validation_end],
        "test": encoded[validation_end:],
    }


def sample_batch(data, batch_size: int, length: int, generator):
    starts = torch.randint(0, data.numel() - length - 1, (batch_size,), generator=generator)
    windows = torch.stack([data[s:s + length + 1] for s in starts.tolist()])
    return windows[:, :-1], windows[:, 1:]


@torch.no_grad()
def evaluate(model, data, batches: int, batch_size: int, length: int, seed: int):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        tokens, targets = sample_batch(data, batch_size, length, generator)
        tokens, targets = tokens.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        losses.append(float(loss))
    loss = statistics.fmean(losses)
    return {"loss": loss, "bits_per_character": loss / math.log(2.0)}


def train_one(config, architecture, seed, data, output, updates, learning_rate):
    model, copied = build_model(config, architecture, seed)
    model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    generator = torch.Generator().manual_seed(seed + 1_000_003)
    logs, step_times = [], []
    best = {"loss": float("inf"), "step": 0}
    warmup = max(20, int(0.05 * updates))
    model.train()
    started = time.perf_counter()
    for step in range(1, updates + 1):
        tokens, targets = sample_batch(
            data["train"], config.train_batch_size,
            config.train_sequence_length, generator,
        )
        tokens, targets = tokens.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        if step <= warmup:
            scale = step / warmup
        else:
            progress = (step - warmup) / max(1, updates - warmup)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - tick) * 1000.0
        if step > 20:
            step_times.append(step_ms)
        if step == 1 or step % 100 == 0 or step == updates:
            record = {
                "step": step, "train_loss": float(loss), "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(gradient_norm), "step_ms": step_ms,
            }
            if step % 250 == 0 or step == updates:
                record["validation"] = evaluate(
                    model, data["validation"], 16, 16,
                    config.train_sequence_length, seed + 3_000_003,
                )
                model.train()
                if record["validation"]["loss"] < best["loss"]:
                    best = {**record["validation"], "step": step}
                    torch.save(model.state_dict(), output / f"{architecture}-seed{seed}-best.pt")
            logs.append(record)
            print(architecture, seed, record, flush=True)
    elapsed = time.perf_counter() - started
    # Report test quality from the validation-selected checkpoint, never from
    # the final (and often overfit) optimizer step.
    best_checkpoint = output / f"{architecture}-seed{seed}-best.pt"
    if not best_checkpoint.exists():
        raise RuntimeError(f"validation did not produce {best_checkpoint}")
    model.load_state_dict(torch.load(best_checkpoint, map_location="cuda"))
    final_test = evaluate(
        model, data["test"], 32, 16, config.train_sequence_length,
        seed + 4_000_003,
    )
    result = {
        "architecture": architecture,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "recurrent_mixer_parameters": sum(
            p.numel() for name, p in model.named_parameters() if ".mixer." in name
        ),
        "shared_initialization": copied,
        "updates": updates,
        "tokens_seen": updates * config.train_batch_size * config.train_sequence_length,
        "wall_seconds": elapsed,
        "median_complete_step_ms_after_warmup": statistics.median(step_times or [logs[-1]["step_ms"]]),
        "best_validation": best,
        "final_test": final_test,
        "log": logs,
    }
    atomic_json(output / f"{architecture}-seed{seed}.json", result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def eager_real_scan(a, b):
    state, output = torch.zeros_like(b[:, 0], dtype=torch.float32), []
    for t in range(a.shape[1]):
        state = a[:, t].float() * state + b[:, t].float()
        output.append(state.to(b.dtype))
    return torch.stack(output, 1)


def eager_complex_scan(ar, ai, br, bi):
    xr, xi = torch.zeros_like(br[:, 0], dtype=torch.float32), torch.zeros_like(bi[:, 0], dtype=torch.float32)
    real, imag = [], []
    for t in range(ar.shape[1]):
        nr = ar[:, t].float() * xr - ai[:, t].float() * xi + br[:, t].float()
        ni = ai[:, t].float() * xr + ar[:, t].float() * xi + bi[:, t].float()
        xr, xi = nr, ni
        real.append(xr.to(br.dtype)); imag.append(xi.to(bi.dtype))
    return torch.stack(real, 1), torch.stack(imag, 1)


def associative_real_scan(a, b):
    """Inclusive parallel prefix for ``h_t = a_t h_{t-1} + b_t``.

    This materialized reference intentionally keeps the transition at every
    token.  It corresponds to the generic associative-scan family in Griffin
    Figure 8; it is not the state-stationary Triton implementation.
    """

    prefix_a, prefix_b = a, b
    offset = 1
    while offset < a.shape[1]:
        combined_a = prefix_a[:, offset:] * prefix_a[:, :-offset]
        combined_b = (
            prefix_a[:, offset:] * prefix_b[:, :-offset]
            + prefix_b[:, offset:]
        )
        prefix_a = torch.cat((prefix_a[:, :offset], combined_a), dim=1)
        prefix_b = torch.cat((prefix_b[:, :offset], combined_b), dim=1)
        offset *= 2
    return prefix_b


def associative_complex_scan(ar, ai, br, bi):
    """Inclusive complex affine prefix scan using a real-pair layout."""

    par, pai, pbr, pbi = ar, ai, br, bi
    offset = 1
    while offset < ar.shape[1]:
        right_ar, right_ai = par[:, offset:], pai[:, offset:]
        left_ar, left_ai = par[:, :-offset], pai[:, :-offset]
        left_br, left_bi = pbr[:, :-offset], pbi[:, :-offset]
        right_br, right_bi = pbr[:, offset:], pbi[:, offset:]
        combined_ar = right_ar * left_ar - right_ai * left_ai
        combined_ai = right_ar * left_ai + right_ai * left_ar
        combined_br = right_ar * left_br - right_ai * left_bi + right_br
        combined_bi = right_ar * left_bi + right_ai * left_br + right_bi
        par = torch.cat((par[:, :offset], combined_ar), dim=1)
        pai = torch.cat((pai[:, :offset], combined_ai), dim=1)
        pbr = torch.cat((pbr[:, :offset], combined_br), dim=1)
        pbi = torch.cat((pbi[:, :offset], combined_bi), dim=1)
        offset *= 2
    return pbr, pbi


def correctness_checks():
    torch.manual_seed(123)
    reports = {}
    for dtype in (torch.float32, torch.bfloat16):
        a0 = (0.8 + 0.15 * torch.rand(2, 17, 37, device="cuda", dtype=torch.float32)).to(dtype)
        b0 = (0.1 * torch.randn(2, 17, 37, device="cuda", dtype=torch.float32)).to(dtype)
        a1, b1 = a0.detach().clone().requires_grad_(), b0.detach().clone().requires_grad_()
        a2, b2 = a0.detach().clone().requires_grad_(), b0.detach().clone().requires_grad_()
        weight = torch.randn_like(b0)
        out1, out2 = real_scan(a1, b1), eager_real_scan(a2, b2)
        (out1.float() * weight.float()).sum().backward()
        (out2.float() * weight.float()).sum().backward()
        reports[f"real_{dtype}"] = {
            "output_max_abs": float((out1.float() - out2.float()).abs().max()),
            "grad_a_max_abs": float((a1.grad.float() - a2.grad.float()).abs().max()),
            "grad_b_max_abs": float((b1.grad.float() - b2.grad.float()).abs().max()),
        }
        chunk_a = torch.cat((a0.detach(), a0.detach()[:, :15]), dim=1)
        chunk_b = torch.cat((b0.detach(), b0.detach()[:, :15]), dim=1)
        ca1, cb1 = chunk_a.clone().requires_grad_(), chunk_b.clone().requires_grad_()
        ca2, cb2 = chunk_a.clone().requires_grad_(), chunk_b.clone().requires_grad_()
        chunk_weight = torch.randn_like(chunk_b)
        chunk_real = real_chunk_scan(ca1, cb1, 16)
        chunk_reference = eager_real_scan(ca2, cb2)
        (chunk_real.float() * chunk_weight.float()).sum().backward()
        (chunk_reference.float() * chunk_weight.float()).sum().backward()
        reports[f"real_chunk16_{dtype}"] = {
            "output_max_abs": float(
                (chunk_real.float() - chunk_reference.float()).abs().max()
            ),
            "grad_a_max_abs": float(
                (ca1.grad.float() - ca2.grad.float()).abs().max()
            ),
            "grad_b_max_abs": float(
                (cb1.grad.float() - cb2.grad.float()).abs().max()
            ),
        }
        tensors1 = [
            (0.8 + 0.1 * torch.rand(2, 17, 19, device="cuda")).to(dtype),
            (0.05 * torch.randn(2, 17, 19, device="cuda")).to(dtype),
            (0.1 * torch.randn(2, 17, 19, device="cuda")).to(dtype),
            (0.1 * torch.randn(2, 17, 19, device="cuda")).to(dtype),
        ]
        tensors1 = [x.requires_grad_() for x in tensors1]
        tensors2 = [x.detach().clone().requires_grad_() for x in tensors1]
        outputs1 = complex_scan(*tensors1)
        outputs2 = eager_complex_scan(*tensors2)
        weights = [torch.randn_like(outputs1[0]), torch.randn_like(outputs1[1])]
        sum((o.float() * w.float()).sum() for o, w in zip(outputs1, weights)).backward()
        sum((o.float() * w.float()).sum() for o, w in zip(outputs2, weights)).backward()
        reports[f"complex_{dtype}"] = {
            "output_max_abs": max(float((x.float() - y.float()).abs().max()) for x, y in zip(outputs1, outputs2)),
            "gradient_max_abs": max(float((x.grad.float() - y.grad.float()).abs().max()) for x, y in zip(tensors1, tensors2)),
        }
        eta0 = (0.05 * torch.randn(2, 17, device="cuda")).requires_grad_()
        delta0 = (0.05 * torch.randn(2, 17, device="cuda")).requires_grad_()
        nu_log0 = torch.randn(19, device="cuda").sub_(1.0).requires_grad_()
        theta_log0 = torch.empty(19, device="cuda").uniform_(-2.0, 1.5).requires_grad_()
        write_r0 = (0.1 * torch.randn(2, 17, 19, device="cuda")).to(dtype).requires_grad_()
        write_i0 = (0.1 * torch.randn(2, 17, 19, device="cuda")).to(dtype).requires_grad_()
        fused_inputs = (eta0, delta0, nu_log0, theta_log0, write_r0, write_i0)
        eager_inputs = tuple(t.detach().clone().requires_grad_() for t in fused_inputs)
        fused_outputs = samu_scan(*fused_inputs)
        eta2, delta2, nu_log2, theta_log2, wr2, wi2 = eager_inputs
        nu2, theta2 = nu_log2.exp(), theta_log2.exp()
        rho2 = torch.exp(-nu2 * torch.exp(eta2).unsqueeze(-1))
        phase2 = theta2 + delta2.unsqueeze(-1)
        eager_outputs = eager_complex_scan(
            rho2 * torch.cos(phase2), rho2 * torch.sin(phase2), wr2, wi2
        )
        fused_weights = [torch.randn_like(fused_outputs[0]), torch.randn_like(fused_outputs[1])]
        sum((o.float() * w.float()).sum() for o, w in zip(fused_outputs, fused_weights)).backward()
        sum((o.float() * w.float()).sum() for o, w in zip(eager_outputs, fused_weights)).backward()
        reports[f"samu_fused_{dtype}"] = {
            "output_max_abs": max(
                float((observed.float() - expected.float()).abs().max())
                for observed, expected in zip(fused_outputs, eager_outputs)
            ),
            "gradient_max_abs": max(
                float((observed.grad.float() - expected.grad.float()).abs().max())
                for observed, expected in zip(fused_inputs, eager_inputs)
            ),
        }
        precomputed_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in fused_inputs
        )
        precomputed_outputs = samu_scan_precomputed_decay(*precomputed_inputs)
        sum(
            (output.float() * weight.float()).sum()
            for output, weight in zip(precomputed_outputs, fused_weights)
        ).backward()
        reports[f"samu_precomputed_decay_{dtype}"] = {
            "output_max_abs": max(
                float((observed.float() - expected.float()).abs().max())
                for observed, expected in zip(precomputed_outputs, eager_outputs)
            ),
            "gradient_max_abs": max(
                float((observed.grad.float() - expected.grad.float()).abs().max())
                for observed, expected in zip(precomputed_inputs, eager_inputs)
            ),
        }
        chunk_inputs = tuple(
            torch.cat((tensor.detach(), tensor.detach()[:, :15]), dim=1).requires_grad_()
            if tensor.ndim >= 2 and tensor.shape[1] == 17
            else tensor.detach().clone().requires_grad_()
            for tensor in fused_inputs
        )
        chunk_eager_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in chunk_inputs
        )
        chunk_outputs = samu_chunk_scan(*chunk_inputs, 16)
        ce, cd, cn, ct, cwr, cwi = chunk_eager_inputs
        cnu, ctheta = cn.exp(), ct.exp()
        crho = torch.exp(-cnu * torch.exp(ce).unsqueeze(-1))
        cphase = ctheta + cd.unsqueeze(-1)
        chunk_eager_outputs = eager_complex_scan(
            crho * torch.cos(cphase), crho * torch.sin(cphase), cwr, cwi
        )
        chunk_weights = [torch.randn_like(output) for output in chunk_outputs]
        sum(
            (output.float() * weight.float()).sum()
            for output, weight in zip(chunk_outputs, chunk_weights)
        ).backward()
        sum(
            (output.float() * weight.float()).sum()
            for output, weight in zip(chunk_eager_outputs, chunk_weights)
        ).backward()
        chunk_gradient_errors = {
            name: float((observed.grad.float() - expected.grad.float()).abs().max())
            for name, observed, expected in zip(
                ("eta", "delta", "nu_log", "theta_log", "write_r", "write_i"),
                chunk_inputs,
                chunk_eager_inputs,
            )
        }
        chunk_gradient_relative = {
            name: error / max(float(expected.grad.float().abs().max()), 1e-12)
            for name, error, expected in zip(
                ("eta", "delta", "nu_log", "theta_log", "write_r", "write_i"),
                chunk_gradient_errors.values(),
                chunk_eager_inputs,
            )
        }
        reports[f"samu_chunk16_{dtype}"] = {
            "output_max_abs": max(
                float((observed.float() - expected.float()).abs().max())
                for observed, expected in zip(chunk_outputs, chunk_eager_outputs)
            ),
            "gradient_max_abs": max(chunk_gradient_errors.values()),
            "gradient_max_relative": max(chunk_gradient_relative.values()),
            "gradient_errors": chunk_gradient_errors,
            "gradient_relative_errors": chunk_gradient_relative,
        }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch, width = 5, 384
        decode_x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([0, 1, 7, 0, 31], device="cuda")
        rg = RGLRUMixer(width, 16).cuda().eval()
        rg_state = torch.randn(batch, width, device="cuda")
        expected_out, expected_state = rg.step(decode_x, rg_state, positions)
        rg.prepare_inference()
        for backend in ("fused", "bmm"):
            rg.set_decode_backend(backend)
            rg.set_decode_num_warps(4)
            observed_out, observed_state = rg.step(
                decode_x, rg_state, positions
            )
            reports[
                "rglru_decode_bfloat16"
                if backend == "fused" else "rglru_decode_bmm_bfloat16"
            ] = {
                "output_max_abs": float(
                    (observed_out.float() - expected_out.float()).abs().max()
                ),
                "state_max_abs": float(
                    (observed_state - expected_state).abs().max()
                ),
            }
        samu = SAMUMixer(width).cuda().eval()
        samu.phase_amplitude.fill_(0.7)
        samu.radial_amplitude.fill_(0.5)
        modes = width // 2
        samu_state = (
            torch.randn(batch, modes, device="cuda"),
            torch.randn(batch, modes, device="cuda"),
        )
        expected_out, expected_state = samu.step(decode_x, samu_state, positions)
        samu.prepare_inference()
        observed_out, observed_state = samu.step(decode_x, samu_state, positions)
        reports["samu_decode_bfloat16"] = {
            "output_max_abs": float((observed_out.float() - expected_out.float()).abs().max()),
            "state_max_abs": max(
                float((observed - expected).abs().max())
                for observed, expected in zip(observed_state, expected_state)
            ),
        }
        for blocked_backend in ("packed", "blocked16", "blocked32", "blocked64"):
            samu.set_decode_backend(blocked_backend)
            blocked_out, blocked_state = samu.step(
                decode_x, samu_state, positions
            )
            reports[f"samu_decode_{blocked_backend}_bfloat16"] = {
                "output_max_abs": float(
                    (blocked_out.float() - expected_out.float()).abs().max()
                ),
                "state_max_abs": max(
                    float((observed - expected).abs().max())
                    for observed, expected in zip(blocked_state, expected_state)
                ),
            }
    return reports


def benchmark_complete_training_steps(
    model,
    config,
    lengths=(256, 2048, 4096, 8192),
    tokens_per_step=8192,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rows = []
    for length in lengths:
        batch = tokens_per_step // length
        tokens = torch.randint(0, config.vocab_size, (batch, length + 1), device="cuda")
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(7):
            tick = time.perf_counter(); step(); torch.cuda.synchronize()
            samples.append((time.perf_counter() - tick) * 1000.0)
        median = statistics.median(samples)
        rows.append({
            "sequence_length": length, "batch_size": batch,
            "tokens_per_step": tokens_per_step, "median_step_ms": median,
            "tokens_per_second": tokens_per_step / (median / 1000.0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "samples_ms": samples,
        })
    del optimizer
    return rows


class CapturedDecoder:
    def __init__(self, model: SmallHawkLM, batch: int):
        self.model = model.eval()
        self.model.prepare_inference()
        self.batch = batch
        self.tokens = torch.zeros(batch, dtype=torch.long, device="cuda")
        self.segment_pos = torch.zeros(batch, dtype=torch.long, device="cuda")
        self.caches = model.init_cache(batch, "cuda", torch.bfloat16)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._body()

    def _body(self):
        logits, new_caches = self.model.step(
            self.tokens, self.caches, self.segment_pos
        )
        for old, new in zip(self.caches, new_caches):
            old[0].copy_(new[0])
            if self.model.architecture == "rglru":
                old[1].copy_(new[1])
            else:
                old[1][0].copy_(new[1][0]); old[1][1].copy_(new[1][1])
        self.tokens.copy_(logits.argmax(dim=-1))
        self.segment_pos.add_(1)

    def reset(self, token: int = 0, caches=None, segment_pos: int = 0):
        self.tokens.fill_(token)
        self.segment_pos.fill_(segment_pos)
        source = caches or self.model.init_cache(self.batch, "cuda", torch.bfloat16)
        for target, value in zip(self.caches, source):
            target[0].copy_(value[0])
            if self.model.architecture == "rglru":
                target[1].copy_(value[1])
            else:
                target[1][0].copy_(value[1][0]); target[1][1].copy_(value[1][1])

    def trajectory_ms(self, steps: int):
        torch.cuda.synchronize(); tick = time.perf_counter()
        for _ in range(steps):
            self.graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - tick) * 1000.0


@torch.no_grad()
def make_prompt_cache(model, prompt):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, caches = model(prompt, return_cache=True)
    return caches


def benchmark_decode(model, data, lengths=(128, 256, 512, 1024, 2048, 4096)):
    latency = []
    decoder = CapturedDecoder(model, 16)
    prompt_source = data["validation"][:4096].cuda()[None].expand(16, -1).contiguous()
    prompt_cache = make_prompt_cache(model, prompt_source)
    for prompt_name, cache in (("empty", None), ("4096", prompt_cache)):
        for length in lengths:
            samples = []
            for _ in range(3):
                decoder.reset(
                    caches=cache,
                    segment_pos=0 if prompt_name == "empty" else 4096,
                )
                samples.append(decoder.trajectory_ms(length))
            latency.append({
                "prompt_tokens": 0 if prompt_name == "empty" else 4096,
                "generated_tokens": length, "batch_size": 16,
                "median_trajectory_ms": statistics.median(samples),
                "samples_ms": samples,
            })
    throughput = []
    for batch in (1, 4, 16, 32, 64, 128, 256):
        candidate = CapturedDecoder(model, batch)
        for length in (512, 1024, 2048, 4096):
            samples = []
            for _ in range(3):
                candidate.reset()
                elapsed_ms = candidate.trajectory_ms(length)
                samples.append(batch * length / (elapsed_ms / 1000.0))
            throughput.append({
                "batch_size": batch, "generated_tokens": length,
                "median_tokens_per_second": statistics.median(samples),
                "samples_tokens_per_second": samples,
            })
        del candidate
    return {"latency": latency, "throughput": throughput}


def load_trained(config, architecture, seed, output):
    model, _ = build_model(config, architecture, seed)
    model.load_state_dict(torch.load(output / f"{architecture}-seed{seed}-best.pt", map_location="cpu"))
    return model.cuda()


def aggregate_training(results):
    by_architecture = {}
    for architecture in ("samu", "rglru"):
        rows = [r for r in results if r["architecture"] == architecture]
        by_architecture[architecture] = {
            "seeds": [r["seed"] for r in rows],
            "parameters": rows[0]["parameters"],
            "recurrent_mixer_parameters": rows[0]["recurrent_mixer_parameters"],
            "mean_best_validation_bpc": statistics.fmean(r["best_validation"]["bits_per_character"] for r in rows),
            "std_best_validation_bpc": statistics.stdev(r["best_validation"]["bits_per_character"] for r in rows) if len(rows) > 1 else 0.0,
            "mean_test_bpc": statistics.fmean(r["final_test"]["bits_per_character"] for r in rows),
            "std_test_bpc": statistics.stdev(r["final_test"]["bits_per_character"] for r in rows) if len(rows) > 1 else 0.0,
            "median_training_step_ms": statistics.median(r["median_complete_step_ms_after_warmup"] for r in rows),
        }
    return by_architecture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "train", "benchmark", "all"), default="all")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_character_data(args.data)
    config = StudyConfig(vocab_size=len(data["alphabet"]))
    environment = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device_capability": torch.cuda.get_device_capability(),
        "data_sha256": sha256(args.data), "data_characters": sum(v.numel() for k, v in data.items() if k != "alphabet"),
        "alphabet_size": len(data["alphabet"]), "official_recurrentgemma_commit": OFFICIAL_RECURRENTGEMMA_COMMIT,
        "config": asdict(config),
    }
    atomic_json(args.output / "environment.json", environment)
    if args.mode in ("test", "all"):
        checks = correctness_checks()
        atomic_json(args.output / "correctness.json", checks)
        print(json.dumps(checks, indent=2), flush=True)
        if any(
            value > (0.12 if "samu_chunk" in key and "bfloat16" in key
                     else 0.04 if "bfloat16" in key else 3e-4)
            for key, row in checks.items()
            for name, value in row.items()
            if isinstance(value, (int, float)) and "relative" not in name
        ):
            raise RuntimeError("scan correctness threshold failed")
    if args.mode in ("train", "all"):
        results = []
        # Alternating order prevents a systematic warm-machine advantage.
        for seed_index, seed in enumerate(args.seeds):
            order = ("samu", "rglru") if seed_index % 2 == 0 else ("rglru", "samu")
            for architecture in order:
                results.append(train_one(
                    config, architecture, seed, data, args.output,
                    args.updates, args.learning_rate,
                ))
        atomic_json(args.output / "training_summary.json", {
            "aggregate": aggregate_training(results), "runs": results,
        })
    if args.mode in ("benchmark", "all"):
        seed = args.seeds[0]
        comparison = {}
        for architecture in ("samu", "rglru"):
            model = load_trained(config, architecture, seed, args.output)
            training_steps = benchmark_complete_training_steps(model, config)
            # The full-step benchmark intentionally applies optimizer updates.
            # Reload the saved checkpoint before measuring inference so decode
            # always starts from the reported trained model.
            del model
            torch.cuda.empty_cache()
            model = load_trained(config, architecture, seed, args.output)
            comparison[architecture] = {
                "complete_training_steps": training_steps,
                "decode": benchmark_decode(model, data),
            }
            del model
            torch.cuda.empty_cache()
        atomic_json(args.output / "full_model_benchmark.json", comparison)
    print("complete", args.output, flush=True)


if __name__ == "__main__":
    main()
