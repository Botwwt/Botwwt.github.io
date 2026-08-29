"""Fused Triton inference kernels for canonical single-group SAMU.

The prefill path performs one padded BF16 projection followed by either a
state-stationary serial kernel or a three-kernel chunk summary/prefix/replay
pipeline. The decode path fuses projection, bounded control, transition, and
state update in one launch. Accumulated complex state is FP32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from kernels import SamuParameters


@dataclass
class PackedSamu:
    weight: torch.Tensor
    selector_bias: torch.Tensor
    phase_scale: float
    radial_scale: float
    nu: torch.Tensor
    cos_theta: torch.Tensor
    sin_theta: torch.Tensor
    gamma: torch.Tensor
    padded_width: int


def pack_samu_parameters(p: SamuParameters, dtype: torch.dtype = torch.bfloat16) -> PackedSamu:
    modes = p.nu.numel()
    interleaved = torch.stack((p.wr, p.wi), dim=-1).reshape(p.wr.shape[0], 2 * modes)
    selectors = torch.stack((p.phase_direction[:-1], p.radial_direction[:-1]), dim=-1)
    useful = torch.cat((interleaved, selectors.to(interleaved.dtype)), dim=-1)
    padded = math.ceil(useful.shape[-1] / 16) * 16
    weight = torch.zeros(useful.shape[0], padded, device=useful.device, dtype=dtype)
    weight[:, : useful.shape[-1]] = useful.to(dtype)
    return PackedSamu(
        weight=weight.contiguous(),
        selector_bias=torch.stack((p.phase_direction[-1], p.radial_direction[-1])).float().contiguous(),
        phase_scale=float(torch.tanh(p.phase_amplitude).item() / math.sqrt(modes)),
        radial_scale=float(torch.tanh(p.radial_amplitude).item() / math.sqrt(modes)),
        nu=p.nu.float().contiguous(),
        cos_theta=torch.cos(p.theta).float().contiguous(),
        sin_theta=torch.sin(p.theta).float().contiguous(),
        gamma=torch.sqrt((1.0 - torch.exp(-2.0 * p.nu)).clamp_min(1e-8)).float().contiguous(),
        padded_width=padded,
    )


@triton.jit
def _control(raw_phase, raw_radial, phase_bias: tl.constexpr, radial_bias: tl.constexpr,
             phase_scale: tl.constexpr, radial_scale: tl.constexpr):
    sp = 2.0 * tl.sigmoid(2.0 * (raw_phase.to(tl.float32) + phase_bias)) - 1.0
    sr = 2.0 * tl.sigmoid(2.0 * (raw_radial.to(tl.float32) + radial_bias)) - 1.0
    d = phase_scale * sp
    c = radial_scale * sr / (1.0 + sr * sr)
    return c, d


@triton.jit
def _transition(nu, cos_theta, sin_theta, c, d):
    g = tl.exp(c)
    rho = tl.exp(-nu * g)
    cd, sd = tl.cos(d), tl.sin(d)
    cp = cos_theta * cd - sin_theta * sd
    sp = sin_theta * cd + cos_theta * sd
    return rho * cp, rho * sp


@triton.jit
def _serial_prefill_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, out,
                           length, modes: tl.constexpr, packed_width: tl.constexpr,
                           phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                           phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                           BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    x = tl.zeros((BLOCK_M,), tl.float32)
    y = tl.zeros((BLOCK_M,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai = _transition(nu, ct, st, c, d)
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx = ar * x - ai * y + wr
        ny = ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask)
        tl.store(out + o + 1, y, mask=mask)


@triton.jit
def _chunk_summary_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr,
                          pr, pi, qr, qi, length, modes: tl.constexpr,
                          chunks, packed_width: tl.constexpr,
                          phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                          phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                          CHUNK: tl.constexpr, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    px, py = tl.full((BLOCK_M,), 1.0, tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    x, y = tl.zeros((BLOCK_M,), tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai = _transition(nu, ct, st, c, d)
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        npx, npy = ar * px - ai * py, ai * px + ar * py
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        px, py, x, y = npx, npy, nx, ny
    o = (chunk_program * modes + m)
    tl.store(pr + o, px, mask=mask); tl.store(pi + o, py, mask=mask)
    tl.store(qr + o, x, mask=mask); tl.store(qi + o, y, mask=mask)


@triton.jit
def _chunk_prefix_kernel(pr, pi, qr, qi, in_r, in_i, modes: tl.constexpr,
                         chunks, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    x, y = tl.zeros((BLOCK_M,), tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        o = (batch * chunks + chunk) * modes + m
        tl.store(in_r + o, x, mask=mask); tl.store(in_i + o, y, mask=mask)
        ar = tl.load(pr + o, mask=mask, other=1.0); ai = tl.load(pi + o, mask=mask, other=0.0)
        br = tl.load(qr + o, mask=mask, other=0.0); bi = tl.load(qi + o, mask=mask, other=0.0)
        nx, ny = ar * x - ai * y + br, ai * x + ar * y + bi
        x, y = nx, ny


@triton.jit
def _chunk_replay_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, in_r, in_i, out,
                         length, modes: tl.constexpr, chunks,
                         packed_width: tl.constexpr, phase_bias: tl.constexpr,
                         radial_bias: tl.constexpr, phase_scale: tl.constexpr,
                         radial_scale: tl.constexpr, CHUNK: tl.constexpr,
                         BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    state_o = chunk_program * modes + m
    x = tl.load(in_r + state_o, mask=mask, other=0.0)
    y = tl.load(in_i + state_o, mask=mask, other=0.0)
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai = _transition(nu, ct, st, c, d)
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask); tl.store(out + o + 1, y, mask=mask)


@triton.jit
def _decode_kernel(u, wr_ptr, wi_ptr, phase_dir, radial_dir, state_r, state_i,
                   nu_ptr, ct_ptr, st_ptr, gamma_ptr, out_r, out_i,
                   d_model: tl.constexpr, modes: tl.constexpr,
                   phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                   phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                   BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    mmask, dmask = m < modes, d < d_model
    uv = tl.load(u + batch * d_model + d, mask=dmask, other=0.0).to(tl.float32)
    phase = tl.sum(uv * tl.load(phase_dir + d, mask=dmask, other=0.0), axis=0)
    radial = tl.sum(uv * tl.load(radial_dir + d, mask=dmask, other=0.0), axis=0)
    c, delta = _control(phase, radial, phase_bias, radial_bias, phase_scale, radial_scale)
    weights_o = d[:, None] * modes + m[None, :]
    wr = tl.sum(uv[:, None] * tl.load(wr_ptr + weights_o, mask=dmask[:, None] & mmask[None, :], other=0.0).to(tl.float32), axis=0)
    wi = tl.sum(uv[:, None] * tl.load(wi_ptr + weights_o, mask=dmask[:, None] & mmask[None, :], other=0.0).to(tl.float32), axis=0)
    nu = tl.load(nu_ptr + m, mask=mmask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mmask, other=1.0), tl.load(st_ptr + m, mask=mmask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mmask, other=0.0)
    ar, ai = _transition(nu, ct, st, c, delta)
    x = tl.load(state_r + batch * modes + m, mask=mmask, other=0.0)
    y = tl.load(state_i + batch * modes + m, mask=mmask, other=0.0)
    nx, ny = ar * x - ai * y + wr * gamma, ai * x + ar * y + wi * gamma
    tl.store(out_r + batch * modes + m, nx, mask=mmask)
    tl.store(out_i + batch * modes + m, ny, mask=mmask)


def _launch_meta(packed: PackedSamu):
    return dict(
        phase_bias=float(packed.selector_bias[0].item()), radial_bias=float(packed.selector_bias[1].item()),
        phase_scale=packed.phase_scale, radial_scale=packed.radial_scale,
    )


def samu_triton_serial(u: torch.Tensor, p: SamuParameters, packed: PackedSamu | None = None) -> torch.Tensor:
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    block = min(128, triton.next_power_of_2(modes))
    _serial_prefill_kernel[(triton.cdiv(modes, block), batch)](
        projected, packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma, out,
        length, modes=modes, packed_width=packed.padded_width, BLOCK_M=block,
        num_warps=4 if block >= 64 else 2, **_launch_meta(packed),
    )
    return out


def samu_triton_chunked(u: torch.Tensor, p: SamuParameters, chunk_size: int,
                        packed: PackedSamu | None = None, *,
                        num_warps: int | None = None) -> torch.Tensor:
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    if length % chunk_size:
        raise ValueError("length must be divisible by chunk_size")
    chunks = length // chunk_size
    projected = u @ packed.weight
    shape = (batch, chunks, modes)
    pr, pi, qr, qi = [torch.empty(shape, device=u.device, dtype=torch.float32) for _ in range(4)]
    in_r, in_i = [torch.empty(shape, device=u.device, dtype=torch.float32) for _ in range(2)]
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    block = min(128, triton.next_power_of_2(modes)); grid = (triton.cdiv(modes, block), batch * chunks)
    launch_warps = num_warps or (2 if block <= 64 else 4)
    meta = dict(length=length, modes=modes, chunks=chunks, packed_width=packed.padded_width,
                CHUNK=chunk_size, BLOCK_M=block, num_warps=launch_warps, **_launch_meta(packed))
    _chunk_summary_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                                packed.gamma, pr, pi, qr, qi, **meta)
    _chunk_prefix_kernel[(triton.cdiv(modes, block), batch)](pr, pi, qr, qi, in_r, in_i,
                                                             modes=modes, chunks=chunks,
                                                             BLOCK_M=block, num_warps=launch_warps)
    _chunk_replay_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                               packed.gamma, in_r, in_i, out, **meta)
    return out


def samu_triton_auto(u: torch.Tensor, p: SamuParameters,
                     packed: PackedSamu | None = None) -> torch.Tensor:
    """Latency-oriented RTX 3090 dispatch policy for the canonical M=64 path.

    The thresholds are deliberately simple and are recorded in benchmark rows;
    they are not claimed to transfer to other GPUs or shapes.
    """
    length = u.shape[1]
    if length <= 256:
        return samu_triton_serial(u, p, packed)
    return samu_triton_chunked(u, p, 16 if length <= 512 else 32, packed)


def samu_triton_decode(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                       p: SamuParameters, packed: PackedSamu | None = None, *,
                       block_m: int | None = None, num_warps: int = 4):
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, d_model = u.shape; modes = p.nu.numel()
    out_r, out_i = torch.empty_like(state[0]), torch.empty_like(state[1])
    block_d = triton.next_power_of_2(d_model)
    block_m = block_m or min(32, triton.next_power_of_2(modes))
    _decode_kernel[(triton.cdiv(modes, block_m), batch)](
        u, p.wr, p.wi, p.phase_direction, p.radial_direction, state[0], state[1],
        packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma, out_r, out_i,
        d_model=d_model, modes=modes, BLOCK_D=block_d, BLOCK_M=block_m,
        num_warps=num_warps, **_launch_meta(packed),
    )
    return out_r, out_i


def samu_packed_reference(u: torch.Tensor, p: SamuParameters,
                          packed: PackedSamu | None = None, *, initial=None):
    """PyTorch reference for the exact packed-BF16 inference policy."""
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    raw_phase, raw_radial = projected[..., 2*modes].float(), projected[..., 2*modes+1].float()
    sp, sr = torch.tanh(raw_phase + packed.selector_bias[0]), torch.tanh(raw_radial + packed.selector_bias[1])
    d = packed.phase_scale * sp; c = packed.radial_scale * sr / (1 + sr.square())
    if initial is None:
        x = torch.zeros(batch, modes, device=u.device, dtype=torch.float32)
        y = torch.zeros_like(x)
    else:
        x, y = initial[0].float(), initial[1].float()
    outputs=[]
    for t in range(length):
        rho = torch.exp(-packed.nu * torch.exp(c[:, t]).unsqueeze(-1))
        cd, sd = torch.cos(d[:, t]).unsqueeze(-1), torch.sin(d[:, t]).unsqueeze(-1)
        cp = packed.cos_theta * cd - packed.sin_theta * sd; si = packed.sin_theta * cd + packed.cos_theta * sd
        wr = projected[:, t, 0:2*modes:2].float() * packed.gamma
        wi = projected[:, t, 1:2*modes:2].float() * packed.gamma
        nx = rho*cp*x - rho*si*y + wr; ny = rho*si*x + rho*cp*y + wi
        x,y=nx,ny; outputs.append(torch.stack((x,y),-1).to(u.dtype))
    return torch.stack(outputs,1)
