"""Fused Triton inference kernels for canonical single-group SAMU.

The prefill path performs one padded BF16 projection followed by either a
state-stationary serial kernel or a three-kernel chunk summary/prefix/replay
pipeline. The decode path fuses projection, bounded control, transition, and
state update in one launch. A two-launch split decode is also provided for
full-model widths where the projection should remain a Tensor-Core GEMM.
Accumulated complex state is FP32.
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
    phase_bias: float
    radial_bias: float
    phase_scale: float
    radial_scale: float
    nu: torch.Tensor
    cos_theta: torch.Tensor
    sin_theta: torch.Tensor
    chunk_cos_theta: dict[int, torch.Tensor]
    chunk_sin_theta: dict[int, torch.Tensor]
    gamma: torch.Tensor
    padded_width: int
    bounded_poly_safe: bool
    use_bounded_poly: bool
    max_abs_c: float
    max_decay_exponent: float
    rho_taylor_error_bound: float
    rho_taylor_safe: bool
    use_rho_taylor: bool


def pack_samu_parameters(p: SamuParameters, dtype: torch.dtype = torch.bfloat16, *,
                         enable_bounded_poly: bool = False,
                         enable_rho_taylor: bool = False) -> PackedSamu:
    modes = p.nu.numel()
    interleaved = torch.stack((p.wr, p.wi), dim=-1).reshape(p.wr.shape[0], 2 * modes)
    selectors = torch.stack((p.phase_direction[:-1], p.radial_direction[:-1]), dim=-1)
    useful = torch.cat((interleaved, selectors.to(interleaved.dtype)), dim=-1)
    padded = math.ceil(useful.shape[-1] / 16) * 16
    weight = torch.zeros(useful.shape[0], padded, device=useful.device, dtype=dtype)
    weight[:, : useful.shape[-1]] = useful.to(dtype)
    phase_scale = float(torch.tanh(p.phase_amplitude).item() / math.sqrt(modes))
    radial_scale = float(torch.tanh(p.radial_amplitude).item() / math.sqrt(modes))
    # sr / (1 + sr**2) is bounded by 1/2.  The resulting interval is a
    # property of SAMU's controller, not an empirical range guessed from one
    # batch.  It lets the kernel replace general exp implementations only when
    # the packed parameters certify a small, non-positive exponent.
    max_abs_c = abs(radial_scale) * 0.5
    min_nu = float(p.nu.min().item())
    max_decay_exponent = float(p.nu.max().item()) * math.exp(max_abs_c)
    bounded_poly_safe = min_nu >= 0.0 and max_abs_c <= 0.125 and max_decay_exponent <= 0.375
    use_bounded_poly = enable_bounded_poly and bounded_poly_safe
    if enable_bounded_poly and not bounded_poly_safe:
        raise ValueError(
            "bounded polynomial exp requested outside its certified interval: "
            f"max_abs_c={max_abs_c:.6g}, max_decay_exponent={max_decay_exponent:.6g}"
        )
    # f_nu(c) = exp(-nu * exp(c)).  SAMU's token-shared radial coordinate is
    # analytically bounded, while nu is static.  Store the degree-4 Taylor
    # coefficients around c=0 so the hot loop needs one Horner chain instead
    # of exp(c) followed by exp(-nu * exp(c)).
    # Taylor's theorem: |R_4| <= max |f^(5)(xi)| |c|^5 / 5!.
    # For q=nu*exp(xi), f^(5)=(-q+15q^2-25q^3+10q^4-q^5)exp(-q).
    # Dropping signs and exp(-q)<=1 gives a conservative closed-form bound.
    qmax = max_decay_exponent
    fifth_derivative_bound = (
        qmax + 15.0 * qmax**2 + 25.0 * qmax**3
        + 10.0 * qmax**4 + qmax**5
    )
    rho_taylor_error_bound = fifth_derivative_bound * max_abs_c**5 / math.factorial(5)
    rho_taylor_safe = min_nu >= 0.0 and rho_taylor_error_bound <= 1e-6
    use_rho_taylor = enable_rho_taylor and rho_taylor_safe
    if enable_rho_taylor and not rho_taylor_safe:
        raise ValueError(
            "rho Taylor path requested outside its certified error budget: "
            f"bound={rho_taylor_error_bound:.6g}"
        )
    return PackedSamu(
        weight=weight.contiguous(),
        # Resolve static GPU scalars once while packing.  Calling .item() in the
        # hot launch path would introduce a Device-to-Host synchronization.
        phase_bias=float(p.phase_direction[-1].item()),
        radial_bias=float(p.radial_direction[-1].item()),
        phase_scale=phase_scale,
        radial_scale=radial_scale,
        nu=p.nu.float().contiguous(),
        cos_theta=torch.cos(p.theta).float().contiguous(),
        sin_theta=torch.sin(p.theta).float().contiguous(),
        chunk_cos_theta={size: torch.cos(size * p.theta).float().contiguous() for size in (8, 16, 32)},
        chunk_sin_theta={size: torch.sin(size * p.theta).float().contiguous() for size in (8, 16, 32)},
        gamma=torch.sqrt((1.0 - torch.exp(-2.0 * p.nu)).clamp_min(1e-8)).float().contiguous(),
        padded_width=padded,
        bounded_poly_safe=bounded_poly_safe,
        use_bounded_poly=use_bounded_poly,
        max_abs_c=max_abs_c,
        max_decay_exponent=max_decay_exponent,
        rho_taylor_error_bound=rho_taylor_error_bound,
        rho_taylor_safe=rho_taylor_safe,
        use_rho_taylor=use_rho_taylor,
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
def _exp_small_symmetric(x):
    """Degree-6 exp on |x| <= 0.125, evaluated as an FMA-friendly Horner chain."""

    return 1.0 + x * (1.0 + x * (0.5 + x * (
        0.16666666666666666 + x * (0.041666666666666664 + x * (
            0.008333333333333333 + x * 0.001388888888888889)))))


@triton.jit
def _exp_small_negative(x):
    """Degree-8 exp on -0.375 <= x <= 0."""

    return 1.0 + x * (1.0 + x * (0.5 + x * (
        0.16666666666666666 + x * (0.041666666666666664 + x * (
            0.008333333333333333 + x * (0.001388888888888889 + x * (
                0.0001984126984126984 + x * 0.0000248015873015873)))))))


@triton.jit
def _rho_taylor_coefficients(nu, gamma):
    """Degree-4 coefficients without another parameter load.

    gamma^2 = 1-exp(-2nu), hence exp(-nu)=sqrt(1-gamma^2).  This
    reconstructs the constant coefficient from an array already resident in
    the recurrence kernel and forms the remaining static coefficients once per
    program, outside its time loop.
    """

    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu2 * nu2
    rho0 = tl.sqrt(tl.maximum(1.0 - gamma * gamma, 0.0))
    return (
        rho0,
        -nu * rho0,
        (nu2 - nu) * rho0 * 0.5,
        (-nu + 3.0 * nu2 - nu3) * rho0 * (1.0 / 6.0),
        (-nu + 7.0 * nu2 - 6.0 * nu3 + nu4) * rho0 * (1.0 / 24.0),
    )


@triton.jit
def _transition(nu, cos_theta, sin_theta, c, d,
                rho0, rho1, rho2, rho3, rho4,
                USE_BOUNDED_POLY: tl.constexpr,
                USE_RHO_TAYLOR: tl.constexpr):
    if USE_RHO_TAYLOR:
        rho = rho0 + c * (rho1 + c * (rho2 + c * (rho3 + c * rho4)))
        # Only the compressed-P chunk prototype consumes g.  Its dispatcher
        # forbids the Taylor path, so this value is dead-code eliminated.
        g = 0.0
    elif USE_BOUNDED_POLY:
        g = _exp_small_symmetric(c)
        rho = _exp_small_negative(-nu * g)
    else:
        g = tl.exp(c)
        rho = tl.exp(-nu * g)
    # The controller guarantees |d| <= tanh(phase_amplitude)/sqrt(M).
    # For M=64 this is < 0.078 rad (and <=0.125 for every M>=64), so these
    # alternating Taylor polynomials have <1e-11 absolute truncation error.
    # This avoids libdevice's unnecessary general-angle range reduction and
    # the associated local-memory spills on Ampere.
    d2 = d * d
    d4 = d2 * d2
    d6 = d4 * d2
    cd = 1.0 - 0.5 * d2 + d4 * (1.0 / 24.0) - d6 * (1.0 / 720.0)
    sd = d * (1.0 - d2 * (1.0 / 6.0) + d4 * (1.0 / 120.0) - d6 * (1.0 / 5040.0))
    cp = cos_theta * cd - sin_theta * sd
    sp = sin_theta * cd + cos_theta * sd
    return rho * cp, rho * sp, g


@triton.jit
def _serial_prefill_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, out,
                           last_r, last_i, length, modes: tl.constexpr,
                           packed_width: tl.constexpr, RETURN_CACHE: tl.constexpr,
                           phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                           phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                           USE_BOUNDED_POLY: tl.constexpr,
                           USE_RHO_TAYLOR: tl.constexpr,
                           BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    x = tl.zeros((BLOCK_M,), tl.float32)
    y = tl.zeros((BLOCK_M,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, _ = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx = ar * x - ai * y + wr
        ny = ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask)
        tl.store(out + o + 1, y, mask=mask)
    if RETURN_CACHE:
        tl.store(last_r + batch * modes + m, x, mask=mask)
        tl.store(last_i + batch * modes + m, y, mask=mask)


@triton.jit
def _chunk_summary_kernel(packed, nu_ptr, ct_ptr, st_ptr, chunk_ct_ptr,
                          chunk_st_ptr, gamma_ptr,
                          pr, pi, qr, qi, length, modes: tl.constexpr,
                          chunks, packed_width: tl.constexpr,
                          phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                          phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                          USE_BOUNDED_POLY: tl.constexpr,
                          USE_RHO_TAYLOR: tl.constexpr,
                          USE_COMPRESSED_P: tl.constexpr,
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
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    if USE_COMPRESSED_P:
        total_g = 0.0
        total_d = 0.0
    else:
        px, py = tl.full((BLOCK_M,), 1.0, tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    x, y = tl.zeros((BLOCK_M,), tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, g = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        if USE_COMPRESSED_P:
            total_g += g
            total_d += d
        else:
            npx, npy = ar * px - ai * py, ai * px + ar * py
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        if not USE_COMPRESSED_P:
            px, py = npx, npy
        x, y = nx, ny
    if USE_COMPRESSED_P:
        # SAMU's transition family is closed under composition:
        # P_j = exp(-nu_j * sum(exp(c_t))) * exp(i(CHUNK*theta_j + sum(d_t))).
        # Keep only the shared sufficient statistics live through the loop,
        # then reconstruct the mode vector once at the chunk boundary.
        base_c = tl.load(chunk_ct_ptr + m, mask=mask, other=1.0)
        base_s = tl.load(chunk_st_ptr + m, mask=mask, other=0.0)
        delta_c, delta_s = tl.cos(total_d), tl.sin(total_d)
        magnitude = tl.exp(-nu * total_g)
        px = magnitude * (base_c * delta_c - base_s * delta_s)
        py = magnitude * (base_s * delta_c + base_c * delta_s)
    o = (chunk_program * modes + m)
    tl.store(pr + o, px, mask=mask); tl.store(pi + o, py, mask=mask)
    tl.store(qr + o, x, mask=mask); tl.store(qi + o, y, mask=mask)


@triton.jit
def _chunk_prefix_kernel(pr, pi, qr, qi, in_r, in_i, modes: tl.constexpr,
                         chunks, last_r, last_i, RETURN_CACHE: tl.constexpr,
                         BLOCK_M: tl.constexpr):
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
    if RETURN_CACHE:
        tl.store(last_r + batch * modes + m, x, mask=mask)
        tl.store(last_i + batch * modes + m, y, mask=mask)


@triton.jit
def _chunk_replay_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, in_r, in_i, out,
                         length, modes: tl.constexpr, chunks,
                         packed_width: tl.constexpr, phase_bias: tl.constexpr,
                         radial_bias: tl.constexpr, phase_scale: tl.constexpr,
                         radial_scale: tl.constexpr,
                         USE_BOUNDED_POLY: tl.constexpr, CHUNK: tl.constexpr,
                         USE_RHO_TAYLOR: tl.constexpr,
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
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, _ = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask); tl.store(out + o + 1, y, mask=mask)


@triton.jit
def _decode_kernel(u, wr_ptr, wi_ptr, phase_dir, radial_dir, state_r, state_i,
                   nu_ptr, ct_ptr, st_ptr, gamma_ptr,
                   out_r, out_i,
                   d_model: tl.constexpr, modes: tl.constexpr,
                   phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                   phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                   USE_BOUNDED_POLY: tl.constexpr,
                   USE_RHO_TAYLOR: tl.constexpr,
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
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    ar, ai, _ = _transition(
        nu, ct, st, c, delta, rho0, rho1, rho2, rho3, rho4,
        USE_BOUNDED_POLY, USE_RHO_TAYLOR,
    )
    x = tl.load(state_r + batch * modes + m, mask=mmask, other=0.0)
    y = tl.load(state_i + batch * modes + m, mask=mmask, other=0.0)
    nx, ny = ar * x - ai * y + wr * gamma, ai * x + ar * y + wi * gamma
    tl.store(out_r + batch * modes + m, nx, mask=mmask)
    tl.store(out_i + batch * modes + m, ny, mask=mmask)


@triton.jit
def _decode_projected_kernel(projected, nu_ptr, ct_ptr, st_ptr, gamma_ptr,
                             state_r, state_i, out, last_r, last_i,
                             modes: tl.constexpr, packed_width: tl.constexpr,
                             phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                             phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                             USE_BOUNDED_POLY: tl.constexpr,
                             USE_RHO_TAYLOR: tl.constexpr,
                             BLOCK_M: tl.constexpr):
    """Pointwise half of split decode after one packed BF16 GEMM."""

    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    row = batch * packed_width
    raw_phase = tl.load(projected + row + 2 * modes)
    raw_radial = tl.load(projected + row + 2 * modes + 1)
    c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias,
                    phase_scale, radial_scale)
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct = tl.load(ct_ptr + m, mask=mask, other=1.0)
    st = tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    ar, ai, _ = _transition(
        nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
        USE_BOUNDED_POLY, USE_RHO_TAYLOR,
    )
    x = tl.load(state_r + batch * modes + m, mask=mask, other=0.0)
    y = tl.load(state_i + batch * modes + m, mask=mask, other=0.0)
    wr = tl.load(projected + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
    wi = tl.load(projected + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
    nx = ar * x - ai * y + wr
    ny = ai * x + ar * y + wi
    output = (batch * modes + m) * 2
    tl.store(out + output, nx, mask=mask)
    tl.store(out + output + 1, ny, mask=mask)
    tl.store(last_r + batch * modes + m, nx, mask=mask)
    tl.store(last_i + batch * modes + m, ny, mask=mask)


def _launch_meta(packed: PackedSamu, bounded_poly: bool | None = None,
                 rho_taylor: bool | None = None):
    use_bounded_poly = packed.use_bounded_poly if bounded_poly is None else bounded_poly
    use_rho_taylor = packed.use_rho_taylor if rho_taylor is None else rho_taylor
    if use_bounded_poly and not packed.bounded_poly_safe:
        raise ValueError("bounded polynomial exp requested outside its certified interval")
    if use_rho_taylor and not packed.rho_taylor_safe:
        raise ValueError("rho Taylor path requested outside its certified error budget")
    if use_bounded_poly and use_rho_taylor:
        raise ValueError("select either the generic bounded-exp path or rho Taylor, not both")
    return dict(
        phase_bias=packed.phase_bias, radial_bias=packed.radial_bias,
        phase_scale=packed.phase_scale, radial_scale=packed.radial_scale,
        USE_BOUNDED_POLY=use_bounded_poly,
        USE_RHO_TAYLOR=use_rho_taylor,
    )


def samu_triton_serial(u: torch.Tensor, p: SamuParameters,
                       packed: PackedSamu | None = None, *,
                       num_warps: int | None = None,
                       return_cache: bool = False):
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    last_i = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    block = min(128, triton.next_power_of_2(modes))
    launch_warps = num_warps or (4 if block >= 64 else 2)
    _serial_prefill_kernel[(triton.cdiv(modes, block), batch)](
        projected, packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        out, last_r, last_i, length, modes=modes,
        packed_width=packed.padded_width, RETURN_CACHE=return_cache, BLOCK_M=block,
        num_warps=launch_warps, **_launch_meta(packed),
    )
    return (out, (last_r, last_i)) if return_cache else out


def samu_triton_chunked(u: torch.Tensor, p: SamuParameters, chunk_size: int,
                        packed: PackedSamu | None = None, *,
                        num_warps: int | None = None,
                        compressed_p: bool = False,
                        bounded_poly: bool | None = None,
                        rho_taylor: bool | None = None,
                        return_cache: bool = False):
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
    last_r = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    last_i = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    block = min(128, triton.next_power_of_2(modes)); grid = (triton.cdiv(modes, block), batch * chunks)
    launch_warps = num_warps or (2 if block <= 64 else 4)
    effective_bounded_poly = packed.use_bounded_poly if bounded_poly is None else bounded_poly
    effective_rho_taylor = packed.use_rho_taylor if rho_taylor is None else rho_taylor
    use_compressed_p = (
        compressed_p and chunk_size in packed.chunk_cos_theta
        and not effective_bounded_poly and not effective_rho_taylor
    )
    chunk_ct = packed.chunk_cos_theta[chunk_size] if use_compressed_p else packed.cos_theta
    chunk_st = packed.chunk_sin_theta[chunk_size] if use_compressed_p else packed.sin_theta
    meta = dict(length=length, modes=modes, chunks=chunks, packed_width=packed.padded_width,
                CHUNK=chunk_size, BLOCK_M=block, USE_COMPRESSED_P=use_compressed_p,
                num_warps=launch_warps,
                **_launch_meta(packed, effective_bounded_poly, effective_rho_taylor))
    _chunk_summary_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                                chunk_ct, chunk_st, packed.gamma,
                                pr, pi, qr, qi, **meta)
    _chunk_prefix_kernel[(triton.cdiv(modes, block), batch)](
        pr, pi, qr, qi, in_r, in_i, modes=modes, chunks=chunks,
        last_r=last_r, last_i=last_i, RETURN_CACHE=return_cache,
        BLOCK_M=block, num_warps=launch_warps,
    )
    _chunk_replay_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                               packed.gamma, in_r, in_i, out,
                               **{key: value for key, value in meta.items() if key != "USE_COMPRESSED_P"})
    return (out, (last_r, last_i)) if return_cache else out


def samu_triton_auto(u: torch.Tensor, p: SamuParameters,
                     packed: PackedSamu | None = None, *,
                     compressed_p: bool = False,
                     return_cache: bool = False):
    """Length-aware default dispatch for the canonical M=64 path.

    Architecture-specific calibration and the independent final timings are
    recorded separately; callers may explicitly select the calibrated kernels.
    """
    length = u.shape[1]
    if length <= 256:
        return samu_triton_chunked(
            u, p, 8, packed, num_warps=4, compressed_p=compressed_p,
            bounded_poly=packed.bounded_poly_safe if length <= 128 else False,
            return_cache=return_cache,
        )
    return samu_triton_chunked(
        u, p, 16 if length <= 512 else 32, packed,
        num_warps=2,
        compressed_p=compressed_p,
        return_cache=return_cache,
    )


def samu_triton_decode(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                       p: SamuParameters, packed: PackedSamu | None = None, *,
                       block_m: int | None = None, num_warps: int | None = None):
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, d_model = u.shape; modes = p.nu.numel()
    out_r, out_i = torch.empty_like(state[0]), torch.empty_like(state[1])
    block_d = triton.next_power_of_2(d_model)
    if block_m is None:
        block_m = 64 if batch > 16 else min(32, triton.next_power_of_2(modes))
    if num_warps is None:
        num_warps = 8
    _decode_kernel[(triton.cdiv(modes, block_m), batch)](
        u, p.wr, p.wi, p.phase_direction, p.radial_direction, state[0], state[1],
        packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        out_r, out_i,
        d_model=d_model, modes=modes, BLOCK_D=block_d, BLOCK_M=block_m,
        num_warps=num_warps, **_launch_meta(packed),
    )
    return out_r, out_i


def samu_triton_decode_split(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                             packed: PackedSamu, *, block_m: int | None = None,
                             num_warps: int | None = None):
    """Two-launch decode: packed Tensor-Core GEMM then pointwise recurrence.

    Projection and recurrence are both inside the measured call.  This path
    computes the same packed-BF16 inference equation as prefill without placing
    a full-width matrix-vector product in every recurrence program.
    """

    if u.ndim != 2:
        raise ValueError("split decode expects [B, D]")
    batch, _ = u.shape
    modes = packed.nu.numel()
    if state[0].shape != (batch, modes) or state[1].shape != (batch, modes):
        raise ValueError("split decode state shape mismatch")
    projected = u @ packed.weight
    out = torch.empty(batch, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty_like(state[0], dtype=torch.float32)
    last_i = torch.empty_like(state[1], dtype=torch.float32)
    block_m = block_m or (64 if batch <= 16 else min(128, triton.next_power_of_2(modes)))
    num_warps = num_warps or 4
    _decode_projected_kernel[(triton.cdiv(modes, block_m), batch)](
        projected, packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        state[0], state[1], out, last_r, last_i,
        modes=modes, packed_width=packed.padded_width, BLOCK_M=block_m,
        num_warps=num_warps, **_launch_meta(packed),
    )
    return out.reshape(batch, 2 * modes), (last_r, last_i)


def samu_packed_reference(u: torch.Tensor, p: SamuParameters,
                          packed: PackedSamu | None = None, *, initial=None,
                          return_cache: bool = False):
    """PyTorch reference for the exact packed-BF16 inference policy."""
    packed = packed or pack_samu_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    raw_phase, raw_radial = projected[..., 2*modes].float(), projected[..., 2*modes+1].float()
    sp = torch.tanh(raw_phase + packed.phase_bias)
    sr = torch.tanh(raw_radial + packed.radial_bias)
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
    output = torch.stack(outputs, 1)
    return (output, (x, y)) if return_cache else output
