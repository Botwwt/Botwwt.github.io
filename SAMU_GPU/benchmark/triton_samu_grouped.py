"""Grouped-write SAMU candidate for a paper-configuration RG-LRU comparison.

The canonical SAMU write map is dense.  RecurrentGemma instead uses sixteen
block-diagonal groups for each of its two gates.  With G=8 write groups, SAMU's
two real write maps contain D^2/8 weights, matching the D^2/8 gate weights of
RG-LRU-16.  SAMU still uses two dense scalar selectors, so exact parameter and
MAC counts are reported rather than silently called identical.

This module is an architecture ablation that requires retraining and quality
validation.  It is not a drop-in optimization of a trained dense-write model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from kernels import SamuParameters
from triton_samu import (
    PackedSamu,
    _chunk_prefix_kernel,
    _chunk_replay_kernel,
    _chunk_summary_kernel,
    _control,
    _launch_meta,
    _transition,
    pack_samu_parameters,
)


@triton.jit
def _bf16_round(value):
    return value.to(tl.bfloat16).to(tl.float32)


@dataclass
class PackedGroupedSamu:
    base: PackedSamu
    wr: torch.Tensor
    wi: torch.Tensor
    write_weight: torch.Tensor
    phase_direction: torch.Tensor
    radial_direction: torch.Tensor
    control_weight: torch.Tensor
    groups: int
    group_input: int
    group_modes: int
    learned_parameter_count: int
    projection_macs_per_token: int


def pack_grouped_samu_parameters(
    parameters: SamuParameters,
    groups: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> PackedGroupedSamu:
    d_model, modes = parameters.wr.shape
    if d_model % groups or modes % groups:
        raise ValueError("d_model and modes must both be divisible by groups")
    group_input, group_modes = d_model // groups, modes // groups
    wr, wi = [], []
    for group in range(groups):
        input_slice = slice(group * group_input, (group + 1) * group_input)
        mode_slice = slice(group * group_modes, (group + 1) * group_modes)
        wr.append(parameters.wr[input_slice, mode_slice])
        wi.append(parameters.wi[input_slice, mode_slice])
    compact_wr = torch.stack(wr).to(dtype).contiguous()
    compact_wi = torch.stack(wi).to(dtype).contiguous()
    write_weight = torch.stack((compact_wr, compact_wi), dim=-1).reshape(
        groups, group_input, 2 * group_modes
    ).contiguous()
    base = pack_samu_parameters(parameters, dtype)
    # Compact learned tensors: two grouped writes, two dense selectors with
    # biases, and the static nu/theta/amplitude vectors/scalars.
    learned_parameter_count = (
        compact_wr.numel() + compact_wi.numel()
        + 2 * (d_model + 1) + 2 * modes + 2
    )
    return PackedGroupedSamu(
        base=base,
        wr=compact_wr,
        wi=compact_wi,
        write_weight=write_weight,
        phase_direction=parameters.phase_direction[:-1].to(dtype).contiguous(),
        radial_direction=parameters.radial_direction[:-1].to(dtype).contiguous(),
        control_weight=torch.stack((
            parameters.phase_direction[:-1], parameters.radial_direction[:-1]
        ), dim=-1).to(dtype).contiguous(),
        groups=groups,
        group_input=group_input,
        group_modes=group_modes,
        learned_parameter_count=learned_parameter_count,
        projection_macs_per_token=compact_wr.numel() + compact_wi.numel() + 2 * d_model,
    )


@triton.jit
def _grouped_decode_kernel(
    u, wr_ptr, wi_ptr, phase_dir, radial_dir, state_r, state_i,
    nu_ptr, ct_ptr, st_ptr, gamma_ptr, out, last_r, last_i,
    d_model: tl.constexpr, modes: tl.constexpr,
    GROUPS: tl.constexpr, GROUP_INPUT: tl.constexpr,
    GROUP_MODES: tl.constexpr,
    phase_bias: tl.constexpr, radial_bias: tl.constexpr,
    phase_scale: tl.constexpr, radial_scale: tl.constexpr,
    DIRECT_CONTROL: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_per_group: tl.constexpr = (GROUP_MODES + BLOCK_M - 1) // BLOCK_M
    group = tile // tiles_per_group
    local_tile = tile - group * tiles_per_group
    local_mode = local_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mode = group * GROUP_MODES + local_mode
    mode_mask = local_mode < GROUP_MODES

    d = tl.arange(0, BLOCK_D)
    dmask = d < d_model
    all_u = tl.load(u + batch * d_model + d, mask=dmask, other=0.0).to(tl.float32)
    if DIRECT_CONTROL:
        phase = tl.sum(tl.where(d == 0, all_u, 0.0), axis=0)
        radial = tl.sum(tl.where(d == 1, all_u, 0.0), axis=0)
    else:
        phase = _bf16_round(tl.sum(
            all_u * tl.load(phase_dir + d, mask=dmask, other=0.0).to(tl.float32),
            axis=0,
        ))
        radial = _bf16_round(tl.sum(
            all_u * tl.load(radial_dir + d, mask=dmask, other=0.0).to(tl.float32),
            axis=0,
        ))
    c, delta = _control(
        phase, radial, phase_bias, radial_bias, phase_scale, radial_scale
    )

    i = tl.arange(0, BLOCK_I)
    imask = i < GROUP_INPUT
    grouped_u = tl.load(
        u + batch * d_model + group * GROUP_INPUT + i,
        mask=imask,
        other=0.0,
    ).to(tl.float32)
    weight_offset = (
        (group * GROUP_INPUT + i[:, None]) * GROUP_MODES
        + local_mode[None, :]
    )
    weight_mask = imask[:, None] & mode_mask[None, :]
    wr = _bf16_round(tl.sum(
        grouped_u[:, None]
        * tl.load(wr_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32),
        axis=0,
    ))
    wi = _bf16_round(tl.sum(
        grouped_u[:, None]
        * tl.load(wi_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32),
        axis=0,
    ))

    nu = tl.load(nu_ptr + mode, mask=mode_mask, other=0.0)
    cos_theta = tl.load(ct_ptr + mode, mask=mode_mask, other=1.0)
    sin_theta = tl.load(st_ptr + mode, mask=mode_mask, other=0.0)
    gamma = tl.load(gamma_ptr + mode, mask=mode_mask, other=0.0)
    ar, ai, _ = _transition(
        nu, cos_theta, sin_theta, c, delta,
        0.0, 0.0, 0.0, 0.0, 0.0,
        USE_BOUNDED_POLY=False, USE_RHO_TAYLOR=False,
    )
    state_offset = batch * modes + mode
    x = tl.load(state_r + state_offset, mask=mode_mask, other=0.0)
    y = tl.load(state_i + state_offset, mask=mode_mask, other=0.0)
    nx = ar * x - ai * y + wr * gamma
    ny = ai * x + ar * y + wi * gamma
    output_offset = (batch * modes + mode) * 2
    tl.store(out + output_offset, nx, mask=mode_mask)
    tl.store(out + output_offset + 1, ny, mask=mode_mask)
    tl.store(last_r + state_offset, nx, mask=mode_mask)
    tl.store(last_i + state_offset, ny, mask=mode_mask)


@triton.jit
def _rmsnorm_controller_kernel(
    x, norm_weight, phase_dir, radial_dir, normalized, controls,
    d_model: tl.constexpr,
    phase_bias: tl.constexpr, radial_bias: tl.constexpr,
    phase_scale: tl.constexpr, radial_scale: tl.constexpr,
    DIRECT_CONTROL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fuse a shared producer (RMSNorm) with SAMU's two scalar selectors."""

    batch = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < d_model
    value = tl.load(x + batch * d_model + d, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(norm_weight + d, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / d_model + 1e-6)
    norm = (value * inverse_rms * weight).to(tl.bfloat16)
    tl.store(normalized + batch * d_model + d, norm, mask=mask)
    if DIRECT_CONTROL:
        phase = tl.sum(tl.where(d == 0, norm.to(tl.float32), 0.0), axis=0)
        radial = tl.sum(tl.where(d == 1, norm.to(tl.float32), 0.0), axis=0)
    else:
        phase = _bf16_round(tl.sum(
            norm.to(tl.float32)
            * tl.load(phase_dir + d, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        ))
        radial = _bf16_round(tl.sum(
            norm.to(tl.float32)
            * tl.load(radial_dir + d, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        ))
    c, delta = _control(
        phase, radial, phase_bias, radial_bias, phase_scale, radial_scale
    )
    tl.store(controls + batch * 2, c)
    tl.store(controls + batch * 2 + 1, delta)


@triton.jit
def _rmsnorm_kernel(
    x, norm_weight, normalized,
    d_model: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """The identical RMSNorm producer used by the RG-LRU comparison path."""

    batch = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < d_model
    value = tl.load(x + batch * d_model + d, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(norm_weight + d, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / d_model + 1e-6)
    norm = (value * inverse_rms * weight).to(tl.bfloat16)
    tl.store(normalized + batch * d_model + d, norm, mask=mask)


@triton.jit
def _grouped_decode_from_control_kernel(
    u, controls, wr_ptr, wi_ptr, state_r, state_i,
    nu_ptr, ct_ptr, st_ptr, gamma_ptr, out, last_r, last_i,
    modes: tl.constexpr, GROUPS: tl.constexpr,
    GROUP_INPUT: tl.constexpr, GROUP_MODES: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Grouped write/update after the controller has been fused into RMSNorm."""

    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_per_group: tl.constexpr = (GROUP_MODES + BLOCK_M - 1) // BLOCK_M
    group = tile // tiles_per_group
    local_tile = tile - group * tiles_per_group
    local_mode = local_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mode = group * GROUP_MODES + local_mode
    mode_mask = local_mode < GROUP_MODES
    i = tl.arange(0, BLOCK_I)
    imask = i < GROUP_INPUT
    grouped_u = tl.load(
        u + batch * (GROUPS * GROUP_INPUT) + group * GROUP_INPUT + i,
        mask=imask,
        other=0.0,
    ).to(tl.float32)
    weight_offset = (
        (group * GROUP_INPUT + i[:, None]) * GROUP_MODES
        + local_mode[None, :]
    )
    weight_mask = imask[:, None] & mode_mask[None, :]
    wr = _bf16_round(tl.sum(
        grouped_u[:, None]
        * tl.load(wr_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32),
        axis=0,
    ))
    wi = _bf16_round(tl.sum(
        grouped_u[:, None]
        * tl.load(wi_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32),
        axis=0,
    ))
    c = tl.load(controls + batch * 2)
    delta = tl.load(controls + batch * 2 + 1)
    nu = tl.load(nu_ptr + mode, mask=mode_mask, other=0.0)
    cos_theta = tl.load(ct_ptr + mode, mask=mode_mask, other=1.0)
    sin_theta = tl.load(st_ptr + mode, mask=mode_mask, other=0.0)
    gamma = tl.load(gamma_ptr + mode, mask=mode_mask, other=0.0)
    ar, ai, _ = _transition(
        nu, cos_theta, sin_theta, c, delta,
        0.0, 0.0, 0.0, 0.0, 0.0,
        USE_BOUNDED_POLY=False, USE_RHO_TAYLOR=False,
    )
    state_offset = batch * modes + mode
    x = tl.load(state_r + state_offset, mask=mode_mask, other=0.0)
    y = tl.load(state_i + state_offset, mask=mode_mask, other=0.0)
    nx = ar * x - ai * y + wr * gamma
    ny = ai * x + ar * y + wi * gamma
    output_offset = (batch * modes + mode) * 2
    tl.store(out + output_offset, nx, mask=mode_mask)
    tl.store(out + output_offset + 1, ny, mask=mode_mask)
    tl.store(last_r + state_offset, nx, mask=mode_mask)
    tl.store(last_i + state_offset, ny, mask=mode_mask)


def grouped_samu_triton_decode(
    u: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    packed: PackedGroupedSamu,
    *,
    block_m: int | None = None,
    num_warps: int | None = None,
    direct_control: bool = False,
):
    if u.ndim != 2 or u.shape[1] != packed.group_input * packed.groups:
        raise ValueError("grouped decode expects [B, D]")
    # A token sliced from [B,L,D] can retain a sequence-sized batch stride.
    # The fused kernel deliberately consumes a compact [B,D] row-major view.
    u = u.contiguous()
    batch, d_model = u.shape
    modes = packed.group_modes * packed.groups
    if state[0].shape != (batch, modes) or state[1].shape != (batch, modes):
        raise ValueError("grouped SAMU state shape mismatch")
    block_m = block_m or triton.next_power_of_2(packed.group_modes)
    if block_m > triton.next_power_of_2(packed.group_modes):
        raise ValueError("block_m must not exceed the padded modes per group")
    num_warps = num_warps or (8 if block_m >= 64 else 4)
    block_d = triton.next_power_of_2(d_model)
    block_i = triton.next_power_of_2(packed.group_input)
    out = torch.empty(batch, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty_like(state[0], dtype=torch.float32)
    last_i = torch.empty_like(state[1], dtype=torch.float32)
    tiles = packed.groups * triton.cdiv(packed.group_modes, block_m)
    base = packed.base
    _grouped_decode_kernel[(tiles, batch)](
        u, packed.wr, packed.wi, packed.phase_direction, packed.radial_direction,
        state[0], state[1], base.nu, base.cos_theta, base.sin_theta, base.gamma,
        out, last_r, last_i,
        d_model=d_model, modes=modes, GROUPS=packed.groups,
        GROUP_INPUT=packed.group_input, GROUP_MODES=packed.group_modes,
        phase_bias=base.phase_bias, radial_bias=base.radial_bias,
        phase_scale=base.phase_scale, radial_scale=base.radial_scale,
        DIRECT_CONTROL=direct_control,
        BLOCK_D=block_d, BLOCK_I=block_i, BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return out.reshape(batch, 2 * modes), (last_r, last_i)


def triton_rmsnorm(x: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or norm_weight.shape != (x.shape[1],):
        raise ValueError("RMSNorm expects x=[B,D] and weight=[D]")
    batch, d_model = x.shape
    normalized = torch.empty_like(x)
    block_d = triton.next_power_of_2(d_model)
    _rmsnorm_kernel[(batch,)](
        x, norm_weight, normalized, d_model=d_model, BLOCK_D=block_d,
        num_warps=8 if block_d >= 1024 else 4,
    )
    return normalized


def grouped_samu_rmsnorm_decode(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    packed: PackedGroupedSamu,
    *,
    block_m: int | None = None,
    num_warps: int | None = None,
    direct_control: bool = False,
):
    """Two-launch system path: RMSNorm+controller, then grouped write/update."""

    if x.ndim != 2 or norm_weight.shape != (x.shape[1],):
        raise ValueError("RMSNorm decode expects x=[B,D] and weight=[D]")
    batch, d_model = x.shape
    modes = packed.group_modes * packed.groups
    if state[0].shape != (batch, modes) or state[1].shape != (batch, modes):
        raise ValueError("grouped SAMU state shape mismatch")
    block_d = triton.next_power_of_2(d_model)
    normalized = torch.empty_like(x)
    controls = torch.empty(batch, 2, device=x.device, dtype=torch.float32)
    base = packed.base
    _rmsnorm_controller_kernel[(batch,)](
        x, norm_weight, packed.phase_direction, packed.radial_direction,
        normalized, controls, d_model=d_model,
        phase_bias=base.phase_bias, radial_bias=base.radial_bias,
        phase_scale=base.phase_scale, radial_scale=base.radial_scale,
        DIRECT_CONTROL=direct_control,
        BLOCK_D=block_d, num_warps=8 if block_d >= 1024 else 4,
    )
    block_m = block_m or triton.next_power_of_2(packed.group_modes)
    if block_m > triton.next_power_of_2(packed.group_modes):
        raise ValueError("block_m must not exceed the padded modes per group")
    num_warps = num_warps or (8 if block_m >= 64 else 4)
    block_i = triton.next_power_of_2(packed.group_input)
    out = torch.empty(batch, modes, 2, device=x.device, dtype=x.dtype)
    last_r = torch.empty_like(state[0], dtype=torch.float32)
    last_i = torch.empty_like(state[1], dtype=torch.float32)
    tiles = packed.groups * triton.cdiv(packed.group_modes, block_m)
    _grouped_decode_from_control_kernel[(tiles, batch)](
        normalized, controls, packed.wr, packed.wi, state[0], state[1],
        base.nu, base.cos_theta, base.sin_theta, base.gamma,
        out, last_r, last_i, modes=modes, GROUPS=packed.groups,
        GROUP_INPUT=packed.group_input, GROUP_MODES=packed.group_modes,
        BLOCK_I=block_i, BLOCK_M=block_m, num_warps=num_warps,
    )
    return out.reshape(batch, 2 * modes), (last_r, last_i)


def grouped_samu_triton_prefill(
    u: torch.Tensor,
    packed: PackedGroupedSamu,
    *,
    chunk_size: int = 32,
    num_warps: int = 2,
    direct_control: bool = False,
):
    """Parallel prefill for the grouped-write architecture ablations.

    A strided batched GEMM computes each independent write group.  The two
    controls are either a single dense two-output GEMM or two carried token
    channels.  The existing three-stage SAMU chunk scan then advances the
    exact same recurrence used by decode.
    """

    if u.ndim != 3:
        raise ValueError("grouped prefill expects [B,L,D]")
    batch, length, d_model = u.shape
    if length % chunk_size:
        raise ValueError("length must be divisible by chunk_size")
    if d_model != packed.groups * packed.group_input:
        raise ValueError("grouped prefill width mismatch")
    modes = packed.groups * packed.group_modes
    rows = batch * length
    blocks = u.reshape(rows, packed.groups, packed.group_input).permute(1, 0, 2)
    grouped_writes = torch.bmm(blocks, packed.write_weight)
    writes = grouped_writes.permute(1, 0, 2).reshape(batch, length, 2 * modes)
    controls = u[..., :2] if direct_control else u @ packed.control_weight
    projected = torch.empty(
        batch, length, packed.base.padded_width,
        device=u.device, dtype=u.dtype,
    )
    projected[..., :2 * modes] = writes
    projected[..., 2 * modes:2 * modes + 2] = controls

    chunks = length // chunk_size
    shape = (batch, chunks, modes)
    pr, pi, qr, qi = [torch.empty(shape, device=u.device) for _ in range(4)]
    in_r, in_i = torch.empty(shape, device=u.device), torch.empty(shape, device=u.device)
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty(batch, modes, device=u.device)
    last_i = torch.empty_like(last_r)
    block = min(128, triton.next_power_of_2(modes))
    grid = (triton.cdiv(modes, block), batch * chunks)
    meta = dict(
        length=length, modes=modes, chunks=chunks,
        packed_width=packed.base.padded_width, CHUNK=chunk_size,
        BLOCK_M=block, USE_COMPRESSED_P=False, num_warps=num_warps,
        **_launch_meta(packed.base),
    )
    _chunk_summary_kernel[grid](
        projected, packed.base.nu, packed.base.cos_theta,
        packed.base.sin_theta, packed.base.cos_theta, packed.base.sin_theta,
        packed.base.gamma, pr, pi, qr, qi, **meta,
    )
    _chunk_prefix_kernel[(triton.cdiv(modes, block), batch)](
        pr, pi, qr, qi, in_r, in_i, modes=modes, chunks=chunks,
        last_r=last_r, last_i=last_i, RETURN_CACHE=True,
        BLOCK_M=block, num_warps=num_warps,
    )
    replay_meta = {key: value for key, value in meta.items() if key != "USE_COMPRESSED_P"}
    _chunk_replay_kernel[grid](
        projected, packed.base.nu, packed.base.cos_theta,
        packed.base.sin_theta, packed.base.gamma, in_r, in_i, out,
        **replay_meta,
    )
    return out.reshape(batch, length, 2 * modes), (last_r, last_i)


def grouped_samu_reference(
    u: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    packed: PackedGroupedSamu,
    *,
    direct_control: bool = False,
):
    batch, _ = u.shape
    modes = packed.group_modes * packed.groups
    writes_r, writes_i = [], []
    for group in range(packed.groups):
        start = group * packed.group_input
        stop = start + packed.group_input
        token = u[:, start:stop]
        writes_r.append(token @ packed.wr[group])
        writes_i.append(token @ packed.wi[group])
    wr = torch.cat(writes_r, dim=-1).float() * packed.base.gamma
    wi = torch.cat(writes_i, dim=-1).float() * packed.base.gamma
    if direct_control:
        phase, radial = u[:, 0], u[:, 1]
    else:
        phase = u @ packed.phase_direction
        radial = u @ packed.radial_direction
    sp = torch.tanh(phase.float() + packed.base.phase_bias)
    sr = torch.tanh(radial.float() + packed.base.radial_bias)
    d = packed.base.phase_scale * sp
    c = packed.base.radial_scale * sr / (1.0 + sr.square())
    rho = torch.exp(-packed.base.nu * torch.exp(c).unsqueeze(-1))
    cd, sd = torch.cos(d).unsqueeze(-1), torch.sin(d).unsqueeze(-1)
    cp = packed.base.cos_theta * cd - packed.base.sin_theta * sd
    si = packed.base.sin_theta * cd + packed.base.cos_theta * sd
    ar, ai = rho * cp, rho * si
    x, y = state[0].float(), state[1].float()
    nx = ar * x - ai * y + wr
    ny = ai * x + ar * y + wi
    return torch.stack((nx, ny), dim=-1).to(u.dtype).reshape(batch, 2 * modes), (nx, ny)
