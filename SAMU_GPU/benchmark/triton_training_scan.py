"""Differentiable state-stationary scans for the small-model H800 study.

The forward pass keeps one recurrent state vector in registers while walking
the sequence.  The custom backward pass walks the sequence in reverse and
keeps the adjoint in registers.  This is the NVIDIA analogue of the custom-VJP
linear Pallas scan in the official RecurrentGemma repository: the recurrence
and gradients are the same, while the grid and tile sizes target CUDA warps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _real_forward(a, b, out, last, length: tl.constexpr, width: tl.constexpr,
                  BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    state = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        offset = (batch * length + t) * width + lane
        at = tl.load(a + offset, mask=mask, other=0.0).to(tl.float32)
        bt = tl.load(b + offset, mask=mask, other=0.0).to(tl.float32)
        state = at * state + bt
        tl.store(out + offset, state, mask=mask)
    tl.store(last + batch * width + lane, state, mask=mask)


@triton.jit
def _real_backward(a, out, grad_out, grad_a, grad_b,
                   length: tl.constexpr, width: tl.constexpr,
                   BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    adjoint = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        offset = (batch * length + t) * width + lane
        direct = tl.load(grad_out + offset, mask=mask, other=0.0).to(tl.float32)
        adjoint += direct
        previous = tl.load(
            out + (batch * length + t - 1) * width + lane,
            mask=mask & (t > 0), other=0.0,
        ).to(tl.float32)
        at = tl.load(a + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_a + offset, adjoint * previous, mask=mask)
        tl.store(grad_b + offset, adjoint, mask=mask)
        adjoint *= at


@triton.jit
def _real_chunk_summary(a, b, p, q,
                        length: tl.constexpr, width: tl.constexpr,
                        chunks: tl.constexpr, CHUNK: tl.constexpr,
                        BLOCK: tl.constexpr):
    """Summarize a real affine chunk as h -> P*h + Q."""
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    product = tl.full((BLOCK,), 1.0, tl.float32)
    state = tl.zeros((BLOCK,), tl.float32)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        position = (batch * length + t) * width + lane
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        write = tl.load(b + position, mask=mask, other=0.0).to(tl.float32)
        product = transition * product
        state = transition * state + write
    summary = chunk_program * width + lane
    tl.store(p + summary, product, mask=mask)
    tl.store(q + summary, state, mask=mask)


@triton.jit
def _real_chunk_prefix(p, q, chunk_in,
                       width: tl.constexpr, chunks: tl.constexpr,
                       BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    state = tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        position = (batch * chunks + chunk) * width + lane
        tl.store(chunk_in + position, state, mask=mask)
        product = tl.load(p + position, mask=mask, other=1.0)
        write = tl.load(q + position, mask=mask, other=0.0)
        state = product * state + write


@triton.jit
def _real_chunk_replay(a, b, chunk_in, out,
                       length: tl.constexpr, width: tl.constexpr,
                       chunks: tl.constexpr, CHUNK: tl.constexpr,
                       BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    summary = chunk_program * width + lane
    state = tl.load(chunk_in + summary, mask=mask, other=0.0)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        position = (batch * length + t) * width + lane
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        write = tl.load(b + position, mask=mask, other=0.0).to(tl.float32)
        state = transition * state + write
        tl.store(out + position, state, mask=mask)


@triton.jit
def _real_reverse_chunk_summary(a, grad_out, p, q,
                                length: tl.constexpr, width: tl.constexpr,
                                chunks: tl.constexpr, CHUNK: tl.constexpr,
                                BLOCK: tl.constexpr):
    """Summarize the reverse real adjoint map for one time chunk."""
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    product = tl.full((BLOCK,), 1.0, tl.float32)
    adjoint = tl.zeros((BLOCK,), tl.float32)
    for reverse_local in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_local)
        position = (batch * length + t) * width + lane
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        direct = tl.load(grad_out + position, mask=mask, other=0.0).to(tl.float32)
        product = transition * product
        adjoint = transition * adjoint + transition * direct
    summary = chunk_program * width + lane
    tl.store(p + summary, product, mask=mask)
    tl.store(q + summary, adjoint, mask=mask)


@triton.jit
def _real_reverse_chunk_prefix(p, q, future,
                               width: tl.constexpr, chunks: tl.constexpr,
                               BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    adjoint = tl.zeros((BLOCK,), tl.float32)
    for reverse_chunk in tl.range(0, chunks, 1, num_stages=1):
        chunk = chunks - 1 - reverse_chunk
        position = (batch * chunks + chunk) * width + lane
        tl.store(future + position, adjoint, mask=mask)
        product = tl.load(p + position, mask=mask, other=1.0)
        write = tl.load(q + position, mask=mask, other=0.0)
        adjoint = product * adjoint + write


@triton.jit
def _real_reverse_chunk_replay(a, out, grad_out, future, grad_a, grad_b,
                               length: tl.constexpr, width: tl.constexpr,
                               chunks: tl.constexpr, CHUNK: tl.constexpr,
                               BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    summary = chunk_program * width + lane
    adjoint = tl.load(future + summary, mask=mask, other=0.0)
    for reverse_local in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_local)
        position = (batch * length + t) * width + lane
        direct = tl.load(grad_out + position, mask=mask, other=0.0).to(tl.float32)
        adjoint += direct
        previous = tl.load(
            out + (batch * length + t - 1) * width + lane,
            mask=mask & (t > 0), other=0.0,
        ).to(tl.float32)
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        tl.store(grad_a + position, adjoint * previous, mask=mask)
        tl.store(grad_b + position, adjoint, mask=mask)
        adjoint *= transition


@triton.jit
def _complex_forward(ar, ai, br, bi, out_r, out_i, last_r, last_i,
                     length: tl.constexpr, modes: tl.constexpr,
                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        offset = (batch * length + t) * modes + lane
        atr = tl.load(ar + offset, mask=mask, other=0.0).to(tl.float32)
        ati = tl.load(ai + offset, mask=mask, other=0.0).to(tl.float32)
        btr = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        bti = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = atr * xr - ati * xi + btr
        ni = ati * xr + atr * xi + bti
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)


@triton.jit
def _complex_backward(ar, ai, out_r, out_i, grad_out_r, grad_out_i,
                      grad_ar, grad_ai, grad_br, grad_bi,
                      length: tl.constexpr, modes: tl.constexpr,
                      BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    adj_r = tl.zeros((BLOCK,), tl.float32)
    adj_i = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        offset = (batch * length + t) * modes + lane
        adj_r += tl.load(grad_out_r + offset, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + offset, mask=mask, other=0.0).to(tl.float32)
        previous_offset = (batch * length + t - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        atr = tl.load(ar + offset, mask=mask, other=0.0).to(tl.float32)
        ati = tl.load(ai + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_ar + offset, adj_r * previous_r + adj_i * previous_i, mask=mask)
        tl.store(grad_ai + offset, -adj_r * previous_i + adj_i * previous_r, mask=mask)
        tl.store(grad_br + offset, adj_r, mask=mask)
        tl.store(grad_bi + offset, adj_i, mask=mask)
        previous_adj_r = atr * adj_r + ati * adj_i
        previous_adj_i = -ati * adj_r + atr * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i


@triton.jit
def _samu_forward(eta, delta, nu_log, theta_log, br, bi,
                  out_r, out_i, last_r, last_i,
                  length: tl.constexpr, modes: tl.constexpr,
                  BLOCK: tl.constexpr):
    """Fuse factorized transition reconstruction with a tiled linear scan."""
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    # theta is mode-specific but time-invariant.  Compute its trigonometric
    # functions once, then use the angle-addition identity with the one shared
    # delta per token.  This is algebraically the same transition as
    # cos(theta + delta), sin(theta + delta), but removes O(L*M) general
    # trigonometric evaluations.
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        rho = tl.exp(-nu * tl.exp(radial))
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = rho * cosine, rho * sine
        offset = token_offset * modes + lane
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = ar * xr - ai * xi + write_r
        ni = ai * xr + ar * xi + write_i
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)


@triton.jit
def _samu_precompute_decay(eta, nu_log, rho,
                           rows: tl.constexpr, modes: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Compute every mode decay with enough independent programs to hide SFU latency."""
    mode_block = tl.program_id(0)
    row = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (lane < modes)
    radial = tl.load(eta + row, mask=row < rows, other=0.0).to(tl.float32)
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    value = tl.exp(-nu * tl.exp(radial))
    tl.store(rho + row * modes + lane, value, mask=mask)


@triton.jit
def _samu_forward_precomputed(rho, delta, theta_log, br, bi,
                              out_r, out_i, last_r, last_i,
                              length: tl.constexpr, modes: tl.constexpr,
                              BLOCK: tl.constexpr):
    """State-stationary scan whose mode-wise exponentials were precomputed."""
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        offset = token_offset * modes + lane
        radius = tl.load(rho + offset, mask=mask, other=0.0)
        ar, ai = radius * cosine, radius * sine
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = ar * xr - ai * xi + write_r
        ni = ai * xr + ar * xi + write_i
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)


@triton.jit
def _samu_backward(eta, delta, nu_log, theta_log, out_r, out_i,
                   grad_out_r, grad_out_i, grad_eta, grad_delta,
                   grad_nu_log, grad_theta_log, grad_br, grad_bi,
                   length: tl.constexpr, modes: tl.constexpr,
                   BLOCK: tl.constexpr):
    """Reverse scan that recomputes SAMU transitions instead of saving them."""
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    adj_r = tl.zeros((BLOCK,), tl.float32)
    adj_i = tl.zeros((BLOCK,), tl.float32)
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_theta = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        token_offset = batch * length + t
        offset = token_offset * modes + lane
        adj_r += tl.load(grad_out_r + offset, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + offset, mask=mask, other=0.0).to(tl.float32)
        previous_offset = (token_offset - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        exp_radial = tl.exp(radial)
        log_radius = -nu * exp_radial
        rho = tl.exp(log_radius)
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = rho * cosine, rho * sine

        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
        phase_gradient = -grad_ar * ai + grad_ai * ar
        tl.store(
            grad_eta + token_offset,
            tl.sum(tl.where(mask, radial_gradient, 0.0), axis=0),
        )
        tl.store(
            grad_delta + token_offset,
            tl.sum(tl.where(mask, phase_gradient, 0.0), axis=0),
        )
        total_nu += radial_gradient
        total_theta += theta * phase_gradient
        tl.store(grad_br + offset, adj_r, mask=mask)
        tl.store(grad_bi + offset, adj_i, mask=mask)
        previous_adj_r = ar * adj_r + ai * adj_i
        previous_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i
    tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
    tl.atomic_add(grad_theta_log + lane, total_theta, mask=mask)


@triton.jit
def _samu_tiled_serial_forward(
    eta, delta, nu_log, theta_log, br, bi, segment_pos, h0_r, h0_i,
    out_r, out_i, last_r, last_i,
    length: tl.constexpr, modes: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    HAS_H0: tl.constexpr, BLOCK: tl.constexpr,
):
    """Wide state-stationary serial scan with no global transition tensor."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if HAS_H0:
        xr = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
        xi = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        xr = tl.zeros((BLOCK,), tl.float32)
        xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        radius = tl.exp(-nu * tl.exp(radial))
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + token_offset) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        ar = tl.where(reset, 0.0, ar)
        ai = tl.where(reset, 0.0, ai)
        offset = token_offset * modes + lane
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        next_r = ar * xr - ai * xi + write_r
        next_i = ai * xr + ar * xi + write_i
        xr, xi = next_r, next_i
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)


@triton.jit
def _samu_tiled_serial_backward(
    eta, delta, nu_log, theta_log, segment_pos, h0_r, h0_i,
    out_r, out_i, grad_out_r, grad_out_i, grad_last_r, grad_last_i,
    partial_eta, partial_delta, grad_nu_log, grad_theta_log,
    grad_br, grad_bi, grad_h0_r, grad_h0_i,
    length: tl.constexpr, modes: tl.constexpr,
    mode_blocks: tl.constexpr, HAS_SEGMENTS: tl.constexpr,
    RESET_FIRST: tl.constexpr, HAS_H0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reverse tiled scan with transition recomputation and shared-control reduction."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    adj_r = tl.load(grad_last_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    adj_i = tl.load(grad_last_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_theta = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        token_offset = batch * length + t
        offset = token_offset * modes + lane
        adj_r += tl.load(grad_out_r + offset, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + offset, mask=mask, other=0.0).to(tl.float32)
        previous_offset = (token_offset - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        if HAS_H0:
            initial_r = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
            initial_i = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
            previous_r = tl.where(t == 0, initial_r, previous_r)
            previous_i = tl.where(t == 0, initial_i, previous_i)
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        exp_radial = tl.exp(radial)
        log_radius = -nu * exp_radial
        radius = tl.exp(log_radius)
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + token_offset) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
        phase_gradient = -grad_ar * ai + grad_ai * ar
        radial_gradient = tl.where(reset, 0.0, radial_gradient)
        phase_gradient = tl.where(reset, 0.0, phase_gradient)
        partial_offset = token_offset * mode_blocks + mode_block
        tl.store(
            partial_eta + partial_offset,
            tl.sum(tl.where(mask, radial_gradient, 0.0), axis=0),
        )
        tl.store(
            partial_delta + partial_offset,
            tl.sum(tl.where(mask, phase_gradient, 0.0), axis=0),
        )
        total_nu += radial_gradient
        total_theta += theta * phase_gradient
        tl.store(grad_br + offset, adj_r, mask=mask)
        tl.store(grad_bi + offset, adj_i, mask=mask)
        ar = tl.where(reset, 0.0, ar)
        ai = tl.where(reset, 0.0, ai)
        previous_adj_r = ar * adj_r + ai * adj_i
        previous_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i
    tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
    tl.atomic_add(grad_theta_log + lane, total_theta, mask=mask)
    tl.store(grad_h0_r + batch * modes + lane, adj_r, mask=mask)
    tl.store(grad_h0_i + batch * modes + lane, adj_i, mask=mask)


@triton.jit
def _samu_backward_precomputed(eta, delta, nu_log, theta_log, rho,
                               out_r, out_i, grad_out_r, grad_out_i,
                               grad_eta, grad_delta, grad_nu_log,
                               grad_theta_log, grad_br, grad_bi,
                               length: tl.constexpr, modes: tl.constexpr,
                               BLOCK: tl.constexpr):
    """Reverse scan reusing precomputed mode decays instead of recomputing exp."""
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    adj_r = tl.zeros((BLOCK,), tl.float32)
    adj_i = tl.zeros((BLOCK,), tl.float32)
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_theta = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        token_offset = batch * length + t
        offset = token_offset * modes + lane
        adj_r += tl.load(grad_out_r + offset, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + offset, mask=mask, other=0.0).to(tl.float32)
        previous_offset = (token_offset - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        radial = tl.load(eta + token_offset).to(tl.float32)
        log_radius = -nu * tl.exp(radial)
        radius = tl.load(rho + offset, mask=mask, other=0.0)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine

        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
        phase_gradient = -grad_ar * ai + grad_ai * ar
        tl.store(
            grad_eta + token_offset,
            tl.sum(tl.where(mask, radial_gradient, 0.0), axis=0),
        )
        tl.store(
            grad_delta + token_offset,
            tl.sum(tl.where(mask, phase_gradient, 0.0), axis=0),
        )
        total_nu += radial_gradient
        total_theta += theta * phase_gradient
        tl.store(grad_br + offset, adj_r, mask=mask)
        tl.store(grad_bi + offset, adj_i, mask=mask)
        previous_adj_r = ar * adj_r + ai * adj_i
        previous_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i
    tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
    tl.atomic_add(grad_theta_log + lane, total_theta, mask=mask)


@triton.jit
def _samu_precompute_shared_controls(eta, delta, exp_eta, cos_delta,
                                     sin_delta, rows: tl.constexpr,
                                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offset = block * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    radial = tl.load(eta + offset, mask=mask, other=0.0).to(tl.float32)
    phase = tl.load(delta + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(exp_eta + offset, tl.exp(radial), mask=mask)
    tl.store(cos_delta + offset, tl.cos(phase), mask=mask)
    tl.store(sin_delta + offset, tl.sin(phase), mask=mask)


@triton.jit
def _samu_precompute_static_spectrum(
    nu_log, theta_log, nu, theta, cos_theta, sin_theta,
    modes: tl.constexpr, BLOCK: tl.constexpr,
):
    """Evaluate the O(M) static spectrum once per complete recurrence."""
    program = tl.program_id(0)
    lane = program * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu_value = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta_value = tl.exp(
        tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32)
    )
    tl.store(nu + lane, nu_value, mask=mask)
    tl.store(theta + lane, theta_value, mask=mask)
    tl.store(cos_theta + lane, tl.cos(theta_value), mask=mask)
    tl.store(sin_theta + lane, tl.sin(theta_value), mask=mask)


@triton.jit
def _samu_chunk_summary(eta, delta, nu_log, theta_log, br, bi, raw_x,
                        write_gamma,
                        exp_eta_shared, cos_delta_shared, sin_delta_shared,
                        spectrum_nu, spectrum_theta,
                        spectrum_cos, spectrum_sin,
                        compressed_g, compressed_d,
                        pr, pi, qr, qi,
                        length: tl.constexpr, modes: tl.constexpr,
                        chunks: tl.constexpr, CHUNK: tl.constexpr,
                        PRECOMPUTED_SHARED: tl.constexpr,
                        PRECOMPUTED_SPECTRAL: tl.constexpr,
                        COMPRESSED_TRANSITION: tl.constexpr,
                        FUSED_WRITE: tl.constexpr,
                        ROUND_WRITE_BF16: tl.constexpr,
                        BLOCK: tl.constexpr):
    """Summarize each independent time chunk as z -> P*z + Q."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:
        # This O(M) value is evaluated once with the canonical PyTorch
        # expression so the eliminated write tensor keeps its BF16 boundary.
        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
    px = tl.full((BLOCK,), 1.0, tl.float32)
    py = tl.zeros((BLOCK,), tl.float32)
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    g_sum = 0.0
    d_sum = 0.0
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            cos_delta = tl.load(cos_delta_shared + token_offset).to(tl.float32)
            sin_delta = tl.load(sin_delta_shared + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
            cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        if COMPRESSED_TRANSITION:
            # D is loaded explicitly in the precomputed-SFU variant because
            # only sin/cos(delta) are cached there.
            if PRECOMPUTED_SHARED:
                phase_delta = tl.load(delta + token_offset).to(tl.float32)
            g_sum += exp_radial
            d_sum += phase_delta
        radius = tl.exp(-nu * exp_radial)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        if FUSED_WRITE:
            raw_position = token_offset * (2 * modes) + lane
            raw_r = tl.load(raw_x + raw_position, mask=mask, other=0.0).to(tl.float32)
            raw_i = tl.load(
                raw_x + raw_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            write_r = raw_r * gamma
            write_i = raw_i * gamma
            # Match the canonical PyTorch path's explicit write.to(BF16)
            # activation boundary without materializing either write tensor.
            if ROUND_WRITE_BF16:
                write_r = write_r.to(tl.bfloat16).to(tl.float32)
                write_i = write_i.to(tl.bfloat16).to(tl.float32)
        else:
            write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
            write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        if not COMPRESSED_TRANSITION:
            next_px = ar * px - ai * py
            next_py = ai * px + ar * py
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        if not COMPRESSED_TRANSITION:
            px, py = next_px, next_py
        x, y = next_x, next_y
    summary_offset = chunk_program * modes + lane
    if COMPRESSED_TRANSITION:
        first_mode_block = mode_block == 0
        tl.store(compressed_g + chunk_program, g_sum, mask=first_mode_block)
        tl.store(compressed_d + chunk_program, d_sum, mask=first_mode_block)
    else:
        tl.store(pr + summary_offset, px, mask=mask)
        tl.store(pi + summary_offset, py, mask=mask)
    tl.store(qr + summary_offset, x, mask=mask)
    tl.store(qi + summary_offset, y, mask=mask)


@triton.jit
def _samu_chunk_prefix(pr, pi, qr, qi, chunk_in_r, chunk_in_i,
                       modes: tl.constexpr, chunks: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Serial prefix over the much shorter sequence of chunk transforms."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(chunk_in_r + offset, x, mask=mask)
        tl.store(chunk_in_i + offset, y, mask=mask)
        ar = tl.load(pr + offset, mask=mask, other=1.0)
        ai = tl.load(pi + offset, mask=mask, other=0.0)
        write_r = tl.load(qr + offset, mask=mask, other=0.0)
        write_i = tl.load(qi + offset, mask=mask, other=0.0)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y


@triton.jit
def _samu_compressed_chunk_prefix(
    compressed_g, compressed_d, nu_log, theta_log, qr, qi,
    spectrum_nu, spectrum_theta,
    boundary_r, boundary_i,
    modes: tl.constexpr, chunks: tl.constexpr, CHUNK: tl.constexpr,
    REVERSE: tl.constexpr, PRECOMPUTED_SPECTRAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Serial prefix reconstructing exact mode transitions from G/D."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for logical_chunk in tl.range(0, chunks, 1, num_stages=1):
        if REVERSE:
            chunk = chunks - 1 - logical_chunk
        else:
            chunk = logical_chunk
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(boundary_r + offset, x, mask=mask)
        tl.store(boundary_i + offset, y, mask=mask)
        scalar_offset = batch * chunks + chunk
        g_value = tl.load(compressed_g + scalar_offset).to(tl.float32)
        d_value = tl.load(compressed_d + scalar_offset).to(tl.float32)
        radius = tl.exp(-nu * g_value)
        phase = theta * CHUNK + d_value
        ar = radius * tl.cos(phase)
        ai = radius * tl.sin(phase)
        if REVERSE:
            ai = -ai
        write_r = tl.load(qr + offset, mask=mask, other=0.0)
        write_i = tl.load(qi + offset, mask=mask, other=0.0)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y


@triton.jit
def _samu_reconstruct_chunk_transition(
    compressed_g, compressed_d, nu_log, theta_log, pr, pi,
    spectrum_nu, spectrum_theta,
    modes: tl.constexpr, chunks: tl.constexpr, CHUNK: tl.constexpr,
    REVERSE: tl.constexpr, PRECOMPUTED_SPECTRAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expand G/D to transient P scratch for a hierarchical prefix."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    g_value = tl.load(compressed_g + chunk_program).to(tl.float32)
    d_value = tl.load(compressed_d + chunk_program).to(tl.float32)
    radius = tl.exp(-nu * g_value)
    phase = theta * CHUNK + d_value
    real = radius * tl.cos(phase)
    imaginary = radius * tl.sin(phase)
    if REVERSE:
        imaginary = -imaginary
    offset = chunk_program * modes + lane
    tl.store(pr + offset, real, mask=mask)
    tl.store(pi + offset, imaginary, mask=mask)


@triton.jit
def _complex_affine_prefix_stage(
    input_pr, input_pi, input_qr, input_qi,
    output_pr, output_pi, output_qr, output_qi,
    modes: tl.constexpr, chunks: tl.constexpr,
    STRIDE: tl.constexpr, REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One race-free Hillis--Steele stage for z -> P*z + Q."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    offset = (batch * chunks + chunk) * modes + lane
    cpr = tl.load(input_pr + offset, mask=mask, other=1.0)
    cpi = tl.load(input_pi + offset, mask=mask, other=0.0)
    cqr = tl.load(input_qr + offset, mask=mask, other=0.0)
    cqi = tl.load(input_qi + offset, mask=mask, other=0.0)
    if REVERSE:
        has_previous = chunk + STRIDE < chunks
        previous_chunk = chunk + STRIDE
    else:
        has_previous = chunk >= STRIDE
        previous_chunk = chunk - STRIDE
    previous_offset = (batch * chunks + previous_chunk) * modes + lane
    ppr = tl.load(
        input_pr + previous_offset, mask=mask & has_previous, other=1.0
    )
    ppi = tl.load(
        input_pi + previous_offset, mask=mask & has_previous, other=0.0
    )
    pqr = tl.load(
        input_qr + previous_offset, mask=mask & has_previous, other=0.0
    )
    pqi = tl.load(
        input_qi + previous_offset, mask=mask & has_previous, other=0.0
    )
    # current o previous: (Pc*Pp, Pc*Qp + Qc)
    result_pr = cpr * ppr - cpi * ppi
    result_pi = cpi * ppr + cpr * ppi
    result_qr = cpr * pqr - cpi * pqi + cqr
    result_qi = cpi * pqr + cpr * pqi + cqi
    tl.store(output_pr + offset, result_pr, mask=mask)
    tl.store(output_pi + offset, result_pi, mask=mask)
    tl.store(output_qr + offset, result_qr, mask=mask)
    tl.store(output_qi + offset, result_qi, mask=mask)


@triton.jit
def _complex_affine_prefix_extract(prefix_qr, prefix_qi,
                                   boundary_r, boundary_i,
                                   modes: tl.constexpr,
                                   chunks: tl.constexpr,
                                   REVERSE: tl.constexpr,
                                   BLOCK: tl.constexpr):
    """Extract exclusive zero-initial chunk boundary states."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if REVERSE:
        first = chunk == chunks - 1
        previous_chunk = chunk + 1
    else:
        first = chunk == 0
        previous_chunk = chunk - 1
    previous_offset = (batch * chunks + previous_chunk) * modes + lane
    state_r = tl.load(
        prefix_qr + previous_offset, mask=mask & ~first, other=0.0
    )
    state_i = tl.load(
        prefix_qi + previous_offset, mask=mask & ~first, other=0.0
    )
    offset = (batch * chunks + chunk) * modes + lane
    tl.store(boundary_r + offset, state_r, mask=mask)
    tl.store(boundary_i + offset, state_i, mask=mask)


@triton.jit
def _complex_affine_group_local(
    chunk_pr, chunk_pi, chunk_qr, chunk_qi,
    group_pr, group_pi, group_qr, group_qi,
    modes: tl.constexpr, chunks: tl.constexpr, groups: tl.constexpr,
    GROUP: tl.constexpr, REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):
    """Exclusive affine scan inside independent contiguous chunk groups.

    The chunk arrays are dead after their summaries have been consumed, so
    this kernel overwrites them with the local exclusive (P,Q) prefix.  The
    much smaller group arrays retain the inclusive transform for the outer
    hierarchy.  Composition is always ``current o accumulated``; this is
    required because affine composition is associative but not commutative.
    """
    mode_block = tl.program_id(0)
    group_program = tl.program_id(1)
    batch = group_program // groups
    group = group_program - batch * groups
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    accumulated_pr = tl.full((BLOCK,), 1.0, tl.float32)
    accumulated_pi = tl.zeros((BLOCK,), tl.float32)
    accumulated_qr = tl.zeros((BLOCK,), tl.float32)
    accumulated_qi = tl.zeros((BLOCK,), tl.float32)
    # Keep the group loop rolled. Unrolling 32--128 affine compositions
    # explodes compile time and instruction-cache pressure without exposing
    # useful ILP because every iteration depends on the accumulated map.
    for logical_chunk in tl.range(0, GROUP, 1, num_stages=1):
        if REVERSE:
            local_chunk = GROUP - 1 - logical_chunk
        else:
            local_chunk = logical_chunk
        chunk = group * GROUP + local_chunk
        valid = chunk < chunks
        offset = (batch * chunks + chunk) * modes + lane
        current_pr = tl.load(
            chunk_pr + offset, mask=mask & valid, other=1.0
        )
        current_pi = tl.load(
            chunk_pi + offset, mask=mask & valid, other=0.0
        )
        current_qr = tl.load(
            chunk_qr + offset, mask=mask & valid, other=0.0
        )
        current_qi = tl.load(
            chunk_qi + offset, mask=mask & valid, other=0.0
        )
        tl.store(chunk_pr + offset, accumulated_pr, mask=mask & valid)
        tl.store(chunk_pi + offset, accumulated_pi, mask=mask & valid)
        tl.store(chunk_qr + offset, accumulated_qr, mask=mask & valid)
        tl.store(chunk_qi + offset, accumulated_qi, mask=mask & valid)
        next_pr = current_pr * accumulated_pr - current_pi * accumulated_pi
        next_pi = current_pi * accumulated_pr + current_pr * accumulated_pi
        next_qr = current_pr * accumulated_qr - current_pi * accumulated_qi + current_qr
        next_qi = current_pi * accumulated_qr + current_pr * accumulated_qi + current_qi
        accumulated_pr = tl.where(valid, next_pr, accumulated_pr)
        accumulated_pi = tl.where(valid, next_pi, accumulated_pi)
        accumulated_qr = tl.where(valid, next_qr, accumulated_qr)
        accumulated_qi = tl.where(valid, next_qi, accumulated_qi)
    group_offset = group_program * modes + lane
    tl.store(group_pr + group_offset, accumulated_pr, mask=mask)
    tl.store(group_pi + group_offset, accumulated_pi, mask=mask)
    tl.store(group_qr + group_offset, accumulated_qr, mask=mask)
    tl.store(group_qi + group_offset, accumulated_qi, mask=mask)


@triton.jit
def _complex_affine_group_outer(
    group_pr, group_pi, group_qr, group_qi,
    group_boundary_r, group_boundary_i,
    modes: tl.constexpr, groups: tl.constexpr,
    REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):
    """Scan the small sequence of group summaries from a zero boundary."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    state_r = tl.zeros((BLOCK,), tl.float32)
    state_i = tl.zeros((BLOCK,), tl.float32)
    for logical_group in tl.range(0, groups, 1, num_stages=1):
        if REVERSE:
            group = groups - 1 - logical_group
        else:
            group = logical_group
        offset = (batch * groups + group) * modes + lane
        tl.store(group_boundary_r + offset, state_r, mask=mask)
        tl.store(group_boundary_i + offset, state_i, mask=mask)
        transform_r = tl.load(group_pr + offset, mask=mask, other=1.0)
        transform_i = tl.load(group_pi + offset, mask=mask, other=0.0)
        write_r = tl.load(group_qr + offset, mask=mask, other=0.0)
        write_i = tl.load(group_qi + offset, mask=mask, other=0.0)
        next_r = transform_r * state_r - transform_i * state_i + write_r
        next_i = transform_i * state_r + transform_r * state_i + write_i
        state_r, state_i = next_r, next_i


@triton.jit
def _complex_affine_group_correct(
    local_pr, local_pi, local_qr, local_qi,
    group_boundary_r, group_boundary_i, boundary_r, boundary_i,
    modes: tl.constexpr, chunks: tl.constexpr,
    groups: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Apply each local exclusive prefix to its outer-group boundary."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    group = chunk // GROUP
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    local_offset = chunk_program * modes + lane
    group_offset = (batch * groups + group) * modes + lane
    transform_r = tl.load(local_pr + local_offset, mask=mask, other=1.0)
    transform_i = tl.load(local_pi + local_offset, mask=mask, other=0.0)
    write_r = tl.load(local_qr + local_offset, mask=mask, other=0.0)
    write_i = tl.load(local_qi + local_offset, mask=mask, other=0.0)
    outer_r = tl.load(group_boundary_r + group_offset, mask=mask, other=0.0)
    outer_i = tl.load(group_boundary_i + group_offset, mask=mask, other=0.0)
    result_r = transform_r * outer_r - transform_i * outer_i + write_r
    result_i = transform_i * outer_r + transform_r * outer_i + write_i
    tl.store(boundary_r + local_offset, result_r, mask=mask)
    tl.store(boundary_i + local_offset, result_i, mask=mask)


@triton.jit
def _samu_chunk_replay(eta, delta, nu_log, theta_log, br, bi, raw_x,
                       write_gamma,
                       exp_eta_shared, cos_delta_shared, sin_delta_shared,
                       spectrum_nu, spectrum_theta,
                       spectrum_cos, spectrum_sin,
                       chunk_in_r, chunk_in_i, out_r, out_i, visible_out,
                       length: tl.constexpr, modes: tl.constexpr,
                       chunks: tl.constexpr, CHUNK: tl.constexpr,
                       PRECOMPUTED_SHARED: tl.constexpr,
                       PRECOMPUTED_SPECTRAL: tl.constexpr,
                       COMPRESSED_TRANSITION: tl.constexpr,
                       FUSED_WRITE: tl.constexpr,
                       ROUND_WRITE_BF16: tl.constexpr,
                       FUSED_OUTPUT_RELU: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Replay each chunk in parallel from the state supplied by the prefix."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    summary_offset = chunk_program * modes + lane
    x = tl.load(chunk_in_r + summary_offset, mask=mask, other=0.0)
    y = tl.load(chunk_in_i + summary_offset, mask=mask, other=0.0)
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:
        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            cos_delta = tl.load(cos_delta_shared + token_offset).to(tl.float32)
            sin_delta = tl.load(sin_delta_shared + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
            cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        radius = tl.exp(-nu * exp_radial)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        if FUSED_WRITE:
            raw_position = token_offset * (2 * modes) + lane
            raw_r = tl.load(raw_x + raw_position, mask=mask, other=0.0).to(tl.float32)
            raw_i = tl.load(
                raw_x + raw_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            write_r = raw_r * gamma
            write_i = raw_i * gamma
            if ROUND_WRITE_BF16:
                write_r = write_r.to(tl.bfloat16).to(tl.float32)
                write_i = write_i.to(tl.bfloat16).to(tl.float32)
        else:
            write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
            write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y
        tl.store(out_r + position, x, mask=mask)
        tl.store(out_i + position, y, mask=mask)
        if FUSED_OUTPUT_RELU:
            visible_position = token_offset * (2 * modes) + lane
            tl.store(visible_out + visible_position, tl.maximum(x, 0.0), mask=mask)
            tl.store(
                visible_out + visible_position + modes,
                tl.maximum(y, 0.0), mask=mask,
            )


@triton.jit
def _samu_reverse_chunk_summary(eta, delta, nu_log, theta_log,
                                exp_eta_shared, cos_delta_shared,
                                sin_delta_shared,
                                spectrum_nu, spectrum_theta,
                                spectrum_cos, spectrum_sin,
                                out_r, out_i, grad_out_r, grad_out_i,
                                grad_visible, pr, pi, qr, qi,
                                length: tl.constexpr, modes: tl.constexpr,
                                chunks: tl.constexpr, CHUNK: tl.constexpr,
                                PRECOMPUTED_SHARED: tl.constexpr,
                                PRECOMPUTED_SPECTRAL: tl.constexpr,
                                COMPRESSED_TRANSITION: tl.constexpr,
                                FUSED_OUTPUT_RELU: tl.constexpr,
                                BLOCK: tl.constexpr):
    """Summarize the reverse adjoint map of each time chunk."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    px = tl.full((BLOCK,), 1.0, tl.float32)
    py = tl.zeros((BLOCK,), tl.float32)
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for reverse_offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_offset)
        token_offset = batch * length + t
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            cos_delta = tl.load(cos_delta_shared + token_offset).to(tl.float32)
            sin_delta = tl.load(sin_delta_shared + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
            cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        radius = tl.exp(-nu * exp_radial)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        # The reverse transition is the complex conjugate of the forward one.
        ar, ai = radius * cosine, -radius * sine
        position = token_offset * modes + lane
        if FUSED_OUTPUT_RELU:
            visible_position = token_offset * (2 * modes) + lane
            state_r = tl.load(out_r + position, mask=mask, other=0.0).to(tl.float32)
            state_i = tl.load(out_i + position, mask=mask, other=0.0).to(tl.float32)
            direct_r = tl.load(
                grad_visible + visible_position, mask=mask, other=0.0
            ).to(tl.float32)
            direct_i = tl.load(
                grad_visible + visible_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            direct_r = tl.where(state_r > 0.0, direct_r, 0.0)
            direct_i = tl.where(state_i > 0.0, direct_i, 0.0)
        else:
            direct_r = tl.load(
                grad_out_r + position, mask=mask, other=0.0
            ).to(tl.float32)
            direct_i = tl.load(
                grad_out_i + position, mask=mask, other=0.0
            ).to(tl.float32)
        write_r = ar * direct_r - ai * direct_i
        write_i = ai * direct_r + ar * direct_i
        if not COMPRESSED_TRANSITION:
            next_px = ar * px - ai * py
            next_py = ai * px + ar * py
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        if not COMPRESSED_TRANSITION:
            px, py = next_px, next_py
        x, y = next_x, next_y
    summary_offset = chunk_program * modes + lane
    if not COMPRESSED_TRANSITION:
        tl.store(pr + summary_offset, px, mask=mask)
        tl.store(pi + summary_offset, py, mask=mask)
    tl.store(qr + summary_offset, x, mask=mask)
    tl.store(qi + summary_offset, y, mask=mask)


@triton.jit
def _samu_reverse_chunk_prefix(pr, pi, qr, qi, future_r, future_i,
                               modes: tl.constexpr, chunks: tl.constexpr,
                               BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for reverse_chunk in tl.range(0, chunks, 1, num_stages=1):
        chunk = chunks - 1 - reverse_chunk
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(future_r + offset, x, mask=mask)
        tl.store(future_i + offset, y, mask=mask)
        ar = tl.load(pr + offset, mask=mask, other=1.0)
        ai = tl.load(pi + offset, mask=mask, other=0.0)
        write_r = tl.load(qr + offset, mask=mask, other=0.0)
        write_i = tl.load(qi + offset, mask=mask, other=0.0)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y


@triton.jit
def _samu_reverse_chunk_replay(eta, delta, nu_log, theta_log, raw_x,
                               write_gamma,
                               exp_eta_shared, cos_delta_shared,
                               sin_delta_shared,
                               spectrum_nu, spectrum_theta,
                               spectrum_cos, spectrum_sin,
                               out_r, out_i, grad_out_r, grad_out_i,
                               grad_visible,
                               future_r, future_i, partial_eta, partial_delta,
                               grad_eta, grad_delta,
                               grad_nu_log, grad_theta_log,
                               partial_nu_log, partial_theta_log,
                               grad_br, grad_bi, grad_raw_x,
                               length: tl.constexpr, modes: tl.constexpr,
                               chunks: tl.constexpr, mode_blocks: tl.constexpr,
                               CHUNK: tl.constexpr,
                               PRECOMPUTED_SHARED: tl.constexpr,
                               PRECOMPUTED_SPECTRAL: tl.constexpr,
                               COMPRESSED_TRANSITION: tl.constexpr,
                               ATOMIC_SHARED: tl.constexpr,
                               TWO_STAGE_SPECTRAL: tl.constexpr,
                               FUSED_WRITE: tl.constexpr,
                               ROUND_WRITE_BF16: tl.constexpr,
                               FUSED_OUTPUT_RELU: tl.constexpr,
                               BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    summary_offset = chunk_program * modes + lane
    adj_r = tl.load(future_r + summary_offset, mask=mask, other=0.0)
    adj_i = tl.load(future_i + summary_offset, mask=mask, other=0.0)
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:
        exp_minus_two_nu = tl.exp(-2.0 * nu)
        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
        gamma_root = gamma - 1.0e-8
        gamma_log_derivative = nu * exp_minus_two_nu / gamma_root
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_theta = tl.zeros((BLOCK,), tl.float32)
    for reverse_offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_offset)
        token_offset = batch * length + t
        position = token_offset * modes + lane
        if FUSED_OUTPUT_RELU:
            visible_position = token_offset * (2 * modes) + lane
            current_r = tl.load(
                out_r + position, mask=mask, other=0.0
            ).to(tl.float32)
            current_i = tl.load(
                out_i + position, mask=mask, other=0.0
            ).to(tl.float32)
            direct_r = tl.load(
                grad_visible + visible_position, mask=mask, other=0.0
            ).to(tl.float32)
            direct_i = tl.load(
                grad_visible + visible_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            adj_r += tl.where(current_r > 0.0, direct_r, 0.0)
            adj_i += tl.where(current_i > 0.0, direct_i, 0.0)
        else:
            adj_r += tl.load(
                grad_out_r + position, mask=mask, other=0.0
            ).to(tl.float32)
            adj_i += tl.load(
                grad_out_i + position, mask=mask, other=0.0
            ).to(tl.float32)
        previous_position = (token_offset - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_position, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_position, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            cos_delta = tl.load(cos_delta_shared + token_offset).to(tl.float32)
            sin_delta = tl.load(sin_delta_shared + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
            cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        log_radius = -nu * exp_radial
        radius = tl.exp(log_radius)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
        phase_gradient = -grad_ar * ai + grad_ai * ar
        partial_offset = token_offset * mode_blocks + mode_block
        radial_partial = tl.sum(tl.where(mask, radial_gradient, 0.0), axis=0)
        phase_partial = tl.sum(tl.where(mask, phase_gradient, 0.0), axis=0)
        if ATOMIC_SHARED:
            tl.atomic_add(grad_eta + token_offset, radial_partial)
            tl.atomic_add(grad_delta + token_offset, phase_partial)
        else:
            tl.store(partial_eta + partial_offset, radial_partial)
            tl.store(partial_delta + partial_offset, phase_partial)
        if FUSED_WRITE:
            raw_position = token_offset * (2 * modes) + lane
            raw_r = tl.load(raw_x + raw_position, mask=mask, other=0.0).to(tl.float32)
            raw_i = tl.load(
                raw_x + raw_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            grad_write_r, grad_write_i = adj_r, adj_i
            # The unfused custom scan returns BF16 write gradients before the
            # outer FP32 normalization multiply. Preserve that boundary.
            if ROUND_WRITE_BF16:
                grad_write_r = grad_write_r.to(tl.bfloat16).to(tl.float32)
                grad_write_i = grad_write_i.to(tl.bfloat16).to(tl.float32)
            write_nu_gradient = (
                grad_write_r * raw_r + grad_write_i * raw_i
            ) * gamma_log_derivative
            grad_raw_r = grad_write_r * gamma
            grad_raw_i = grad_write_i * gamma
            tl.store(grad_raw_x + raw_position, grad_raw_r, mask=mask)
            tl.store(
                grad_raw_x + raw_position + modes, grad_raw_i, mask=mask
            )
            total_nu += radial_gradient + write_nu_gradient
        else:
            total_nu += radial_gradient
        total_theta += theta * phase_gradient
        if not FUSED_WRITE:
            tl.store(grad_br + position, adj_r, mask=mask)
            tl.store(grad_bi + position, adj_i, mask=mask)
        previous_adj_r = ar * adj_r + ai * adj_i
        previous_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i
    if TWO_STAGE_SPECTRAL:
        tl.store(partial_nu_log + summary_offset, total_nu, mask=mask)
        tl.store(partial_theta_log + summary_offset, total_theta, mask=mask)
    else:
        tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
        tl.atomic_add(grad_theta_log + lane, total_theta, mask=mask)


@triton.jit
def _samu_reverse_chunk_replay_grouped(
    eta, delta, nu_log, theta_log, raw_x, write_gamma,
    exp_eta_shared, cos_delta_shared, sin_delta_shared,
    spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
    out_r, out_i, grad_out_r, grad_out_i,
    future_r, future_i, partial_eta, partial_delta,
    grad_eta, grad_delta, grad_nu_log, grad_theta_log,
    grad_br, grad_bi, grad_raw_x,
    length: tl.constexpr, modes: tl.constexpr,
    chunks: tl.constexpr, chunk_groups: tl.constexpr,
    mode_blocks: tl.constexpr, CHUNK: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    PRECOMPUTED_SHARED: tl.constexpr,
    PRECOMPUTED_SPECTRAL: tl.constexpr,
    ATOMIC_SHARED: tl.constexpr,
    FUSED_WRITE: tl.constexpr,
    ROUND_WRITE_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Replay adjacent chunks and issue one spectral atomic per group."""
    mode_block = tl.program_id(0)
    group_program = tl.program_id(1)
    batch = group_program // chunk_groups
    chunk_group = group_program - batch * chunk_groups
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(
            tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32)
        )
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:
        exp_minus_two_nu = tl.exp(-2.0 * nu)
        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
        gamma_root = gamma - 1.0e-8
        gamma_log_derivative = nu * exp_minus_two_nu / gamma_root
    grouped_nu = tl.zeros((BLOCK,), tl.float32)
    grouped_theta = tl.zeros((BLOCK,), tl.float32)
    for local_chunk in tl.static_range(0, CHUNKS_PER_PROGRAM):
        chunk = chunk_group * CHUNKS_PER_PROGRAM + local_chunk
        summary_offset = (batch * chunks + chunk) * modes + lane
        adj_r = tl.load(future_r + summary_offset, mask=mask, other=0.0)
        adj_i = tl.load(future_i + summary_offset, mask=mask, other=0.0)
        for reverse_offset in tl.static_range(0, CHUNK):
            t = chunk * CHUNK + (CHUNK - 1 - reverse_offset)
            token_offset = batch * length + t
            position = token_offset * modes + lane
            adj_r += tl.load(
                grad_out_r + position, mask=mask, other=0.0
            ).to(tl.float32)
            adj_i += tl.load(
                grad_out_i + position, mask=mask, other=0.0
            ).to(tl.float32)
            previous_position = (token_offset - 1) * modes + lane
            previous_r = tl.load(
                out_r + previous_position, mask=mask & (t > 0), other=0.0
            ).to(tl.float32)
            previous_i = tl.load(
                out_i + previous_position, mask=mask & (t > 0), other=0.0
            ).to(tl.float32)
            if PRECOMPUTED_SHARED:
                exp_radial = tl.load(
                    exp_eta_shared + token_offset
                ).to(tl.float32)
                cos_delta = tl.load(
                    cos_delta_shared + token_offset
                ).to(tl.float32)
                sin_delta = tl.load(
                    sin_delta_shared + token_offset
                ).to(tl.float32)
            else:
                radial = tl.load(eta + token_offset).to(tl.float32)
                phase_delta = tl.load(delta + token_offset).to(tl.float32)
                exp_radial = tl.exp(radial)
                cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
            log_radius = -nu * exp_radial
            radius = tl.exp(log_radius)
            cosine = cos_theta * cos_delta - sin_theta * sin_delta
            sine = sin_theta * cos_delta + cos_theta * sin_delta
            ar, ai = radius * cosine, radius * sine
            grad_ar = adj_r * previous_r + adj_i * previous_i
            grad_ai = -adj_r * previous_i + adj_i * previous_r
            radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
            phase_gradient = -grad_ar * ai + grad_ai * ar
            partial_offset = token_offset * mode_blocks + mode_block
            radial_partial = tl.sum(
                tl.where(mask, radial_gradient, 0.0), axis=0
            )
            phase_partial = tl.sum(
                tl.where(mask, phase_gradient, 0.0), axis=0
            )
            if ATOMIC_SHARED:
                tl.atomic_add(grad_eta + token_offset, radial_partial)
                tl.atomic_add(grad_delta + token_offset, phase_partial)
            else:
                tl.store(partial_eta + partial_offset, radial_partial)
                tl.store(partial_delta + partial_offset, phase_partial)
            if FUSED_WRITE:
                raw_position = token_offset * (2 * modes) + lane
                raw_r = tl.load(
                    raw_x + raw_position, mask=mask, other=0.0
                ).to(tl.float32)
                raw_i = tl.load(
                    raw_x + raw_position + modes, mask=mask, other=0.0
                ).to(tl.float32)
                grad_write_r, grad_write_i = adj_r, adj_i
                if ROUND_WRITE_BF16:
                    grad_write_r = grad_write_r.to(tl.bfloat16).to(tl.float32)
                    grad_write_i = grad_write_i.to(tl.bfloat16).to(tl.float32)
                write_nu_gradient = (
                    grad_write_r * raw_r + grad_write_i * raw_i
                ) * gamma_log_derivative
                tl.store(
                    grad_raw_x + raw_position, grad_write_r * gamma, mask=mask
                )
                tl.store(
                    grad_raw_x + raw_position + modes,
                    grad_write_i * gamma, mask=mask,
                )
                grouped_nu += radial_gradient + write_nu_gradient
            else:
                grouped_nu += radial_gradient
                tl.store(grad_br + position, adj_r, mask=mask)
                tl.store(grad_bi + position, adj_i, mask=mask)
            grouped_theta += theta * phase_gradient
            previous_adj_r = ar * adj_r + ai * adj_i
            previous_adj_i = -ai * adj_r + ar * adj_i
            adj_r, adj_i = previous_adj_r, previous_adj_i
    tl.atomic_add(grad_nu_log + lane, grouped_nu, mask=mask)
    tl.atomic_add(grad_theta_log + lane, grouped_theta, mask=mask)


@triton.jit
def _reduce_shared_control_gradients(partial_eta, partial_delta,
                                     grad_eta, grad_delta,
                                     rows: tl.constexpr,
                                     mode_blocks: tl.constexpr,
                                     REDUCE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    mask = offsets < mode_blocks
    base = row * mode_blocks + offsets
    eta_value = tl.sum(tl.load(partial_eta + base, mask=mask, other=0.0), axis=0)
    delta_value = tl.sum(tl.load(partial_delta + base, mask=mask, other=0.0), axis=0)
    tl.store(grad_eta + row, eta_value)
    tl.store(grad_delta + row, delta_value)


@triton.jit
def _samu_fused_controller_backward(
    raw_x, grad_write_x, grad_eta, grad_delta,
    phase_coordinate, radial_raw,
    normalized_phase, normalized_radial,
    phase_amplitude, radial_amplitude,
    grad_x, direction_partial, amplitude_partial,
    rows: tl.constexpr, width: tl.constexpr,
    row_chunks: tl.constexpr, ROW_CHUNK: tl.constexpr,
    SCALE: tl.constexpr, ROUND_INPUT_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse low-rank controller dx with recurrent write dx.

    Each CTA owns one augmented-direction tile and one row chunk. It writes
    the final state-sized input gradient once and emits only deterministic
    O(N_row_chunks * D) direction-gradient partials.
    """
    width_block = tl.program_id(0)
    row_chunk = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    augmented_width = width + 1
    lane_mask = lane < augmented_width
    input_lane_mask = lane < width
    norm_phase = tl.load(
        normalized_phase + lane, mask=lane_mask, other=0.0
    ).to(tl.float32)
    norm_radial = tl.load(
        normalized_radial + lane, mask=lane_mask, other=0.0
    ).to(tl.float32)
    phase_strength = libdevice.tanh(
        tl.load(phase_amplitude).to(tl.float32)
    )
    radial_strength = libdevice.tanh(
        tl.load(radial_amplitude).to(tl.float32)
    )
    partial_phase = tl.zeros((BLOCK,), tl.float32)
    partial_radial = tl.zeros((BLOCK,), tl.float32)
    partial_phase_amplitude = 0.0
    partial_radial_amplitude = 0.0
    for local_row in tl.range(0, ROW_CHUNK, 1, num_stages=1):
        row = row_chunk * ROW_CHUNK + local_row
        valid_row = row < rows
        phase = tl.load(
            phase_coordinate + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        radial = tl.load(
            radial_raw + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        g_eta = tl.load(
            grad_eta + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        g_delta = tl.load(
            grad_delta + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        radial_square = radial * radial
        denominator = 1.0 + radial_square
        radial_coordinate = radial / denominator
        radial_q_derivative = (
            (1.0 - radial_square) / (denominator * denominator)
        )
        grad_phase_projection = (
            g_delta * SCALE * phase_strength * (1.0 - phase * phase)
        )
        grad_radial_projection = (
            g_eta * SCALE * radial_strength
            * radial_q_derivative * (1.0 - radial_square)
        )
        input_value = tl.load(
            raw_x + row * width + lane,
            mask=valid_row & input_lane_mask, other=0.0,
        ).to(tl.float32)
        augmented_value = tl.where(lane < width, input_value, 1.0)
        partial_phase += tl.where(
            valid_row & lane_mask,
            grad_phase_projection * augmented_value, 0.0,
        )
        partial_radial += tl.where(
            valid_row & lane_mask,
            grad_radial_projection * augmented_value, 0.0,
        )
        controller_dx = (
            grad_phase_projection * norm_phase
            + grad_radial_projection * norm_radial
        )
        if ROUND_INPUT_BF16:
            controller_dx = controller_dx.to(tl.bfloat16).to(tl.float32)
        write_dx = tl.load(
            grad_write_x + row * width + lane,
            mask=valid_row & input_lane_mask, other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_x + row * width + lane, write_dx + controller_dx,
            mask=valid_row & input_lane_mask,
        )
        partial_phase_amplitude += tl.where(
            valid_row,
            g_delta * SCALE * phase *
            (1.0 - phase_strength * phase_strength), 0.0,
        )
        partial_radial_amplitude += tl.where(
            valid_row,
            g_eta * SCALE * radial_coordinate *
            (1.0 - radial_strength * radial_strength), 0.0,
        )
    partial_offset = row_chunk * 2 * augmented_width + lane
    tl.store(direction_partial + partial_offset, partial_phase, mask=lane_mask)
    tl.store(
        direction_partial + partial_offset + augmented_width,
        partial_radial, mask=lane_mask,
    )
    first_width_block = width_block == 0
    tl.store(
        amplitude_partial + row_chunk * 2, partial_phase_amplitude,
        mask=first_width_block,
    )
    tl.store(
        amplitude_partial + row_chunk * 2 + 1,
        partial_radial_amplitude, mask=first_width_block,
    )


@triton.jit
def _samu_decode(x, phase_direction, radial_direction,
                 phase_amplitude, radial_amplitude,
                 nu_log, theta_log, state_r, state_i,
                 out, last_r, last_i,
                 width: tl.constexpr, modes: tl.constexpr,
                 SCALE: tl.constexpr,
                 BLOCK: tl.constexpr):
    """One-launch SAMU controller, transition, write and state update."""
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    width_mask = lane < width
    values = tl.load(x + batch * width + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_vector = tl.load(phase_direction + lane, mask=width_mask, other=0.0).to(tl.float32)
    radial_vector = tl.load(radial_direction + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_bias = tl.load(phase_direction + width).to(tl.float32)
    radial_bias = tl.load(radial_direction + width).to(tl.float32)
    phase_norm = tl.sqrt(tl.sum(phase_vector * phase_vector, axis=0) + phase_bias * phase_bias)
    radial_norm = tl.sqrt(tl.sum(radial_vector * radial_vector, axis=0) + radial_bias * radial_bias)
    phase_raw = (tl.sum(values * phase_vector, axis=0) + phase_bias) / tl.maximum(phase_norm, 1e-12)
    radial_raw = (tl.sum(values * radial_vector, axis=0) + radial_bias) / tl.maximum(radial_norm, 1e-12)
    # tanh(q) = 2*sigmoid(2q)-1.  This keeps the complete decode path inside
    # one portable Triton kernel.
    phase_selector = 2.0 * tl.sigmoid(2.0 * phase_raw) - 1.0
    radial_selector = 2.0 * tl.sigmoid(2.0 * radial_raw) - 1.0
    physical_radial = radial_selector / (1.0 + radial_selector * radial_selector)
    raw_phase_amplitude = tl.load(phase_amplitude).to(tl.float32)
    raw_radial_amplitude = tl.load(radial_amplitude).to(tl.float32)
    bounded_phase = 2.0 * tl.sigmoid(2.0 * raw_phase_amplitude) - 1.0
    bounded_radial = 2.0 * tl.sigmoid(2.0 * raw_radial_amplitude) - 1.0
    phase_delta = SCALE * bounded_phase * phase_selector
    radial_delta = SCALE * bounded_radial * physical_radial

    mode_mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    rho = tl.exp(-nu * tl.exp(radial_delta))
    phase = theta + phase_delta
    ar, ai = rho * tl.cos(phase), rho * tl.sin(phase)
    gamma = tl.sqrt(tl.maximum(1.0 - tl.exp(-2.0 * nu), 0.0)) + 1.0e-8
    write_r = values * gamma
    write_i_values = tl.load(
        x + batch * width + modes + lane, mask=mode_mask, other=0.0
    ).to(tl.float32)
    write_i = write_i_values * gamma
    previous_r = tl.load(state_r + batch * modes + lane, mask=mode_mask, other=0.0)
    previous_i = tl.load(state_i + batch * modes + lane, mask=mode_mask, other=0.0)
    new_r = ar * previous_r - ai * previous_i + write_r
    new_i = ai * previous_r + ar * previous_i + write_i
    tl.store(last_r + batch * modes + lane, new_r, mask=mode_mask)
    tl.store(last_i + batch * modes + lane, new_i, mask=mode_mask)
    tl.store(out + batch * width + lane, tl.maximum(new_r, 0.0), mask=mode_mask)
    tl.store(out + batch * width + modes + lane, tl.maximum(new_i, 0.0), mask=mode_mask)


@triton.jit
def _samu_decode_blocked(x, normalized_phase_direction,
                         normalized_radial_direction,
                         phase_scale, radial_scale,
                         nu, cos_theta, sin_theta, gamma,
                         state_r, state_i, out, last_r, last_i,
                         width: tl.constexpr, modes: tl.constexpr,
                         BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr):
    """Mode-parallel decode using inference-packed static spectral values."""
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < width
    values = tl.load(x + batch * width + d, mask=dmask, other=0.0).to(tl.float32)
    phase_vector = tl.load(
        normalized_phase_direction + d, mask=dmask, other=0.0
    ).to(tl.float32)
    radial_vector = tl.load(
        normalized_radial_direction + d, mask=dmask, other=0.0
    ).to(tl.float32)
    phase_bias = tl.load(normalized_phase_direction + width).to(tl.float32)
    radial_bias = tl.load(normalized_radial_direction + width).to(tl.float32)
    phase_raw = tl.sum(values * phase_vector, axis=0) + phase_bias
    radial_raw = tl.sum(values * radial_vector, axis=0) + radial_bias
    phase_selector = 2.0 * tl.sigmoid(2.0 * phase_raw) - 1.0
    radial_selector = 2.0 * tl.sigmoid(2.0 * radial_raw) - 1.0
    physical_radial = radial_selector / (1.0 + radial_selector * radial_selector)
    phase_delta = tl.load(phase_scale).to(tl.float32) * phase_selector
    radial_delta = tl.load(radial_scale).to(tl.float32) * physical_radial

    local = tl.arange(0, BLOCK_M)
    mode = mode_block * BLOCK_M + local
    mask = mode < modes
    mode_nu = tl.load(nu + mode, mask=mask, other=0.0)
    base_cos = tl.load(cos_theta + mode, mask=mask, other=1.0)
    base_sin = tl.load(sin_theta + mode, mask=mask, other=0.0)
    radius = tl.exp(-mode_nu * tl.exp(radial_delta))
    cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
    cosine = base_cos * cos_delta - base_sin * sin_delta
    sine = base_sin * cos_delta + base_cos * sin_delta
    ar, ai = radius * cosine, radius * sine
    mode_gamma = tl.load(gamma + mode, mask=mask, other=0.0)
    write_r = tl.load(
        x + batch * width + mode, mask=mask, other=0.0
    ).to(tl.float32) * mode_gamma
    write_i = tl.load(
        x + batch * width + modes + mode, mask=mask, other=0.0
    ).to(tl.float32) * mode_gamma
    previous_r = tl.load(
        state_r + batch * modes + mode, mask=mask, other=0.0
    )
    previous_i = tl.load(
        state_i + batch * modes + mode, mask=mask, other=0.0
    )
    new_r = ar * previous_r - ai * previous_i + write_r
    new_i = ai * previous_r + ar * previous_i + write_i
    tl.store(last_r + batch * modes + mode, new_r, mask=mask)
    tl.store(last_i + batch * modes + mode, new_i, mask=mask)
    tl.store(out + batch * width + mode, tl.maximum(new_r, 0.0), mask=mask)
    tl.store(
        out + batch * width + modes + mode,
        tl.maximum(new_i, 0.0), mask=mask,
    )


def _launch_shape(width: int) -> tuple[int, int]:
    block = min(128, triton.next_power_of_2(width))
    return block, triton.cdiv(width, block)


class _RealScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape or a.ndim != 3:
            raise ValueError("real scan expects matching [B, L, D] tensors")
        a, b = a.contiguous(), b.contiguous()
        batch, length, width = a.shape
        out = torch.empty_like(b)
        last = torch.empty(batch, width, device=a.device, dtype=torch.float32)
        block, blocks = _launch_shape(width)
        _real_forward[(blocks, batch)](
            a, b, out, last, length=length, width=width, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        ctx.save_for_backward(a, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, out = ctx.saved_tensors
        batch, length, width = a.shape
        grad_out = grad_out.contiguous()
        grad_a, grad_b = torch.empty_like(a), torch.empty_like(grad_out)
        block, blocks = _launch_shape(width)
        _real_backward[(blocks, batch)](
            a, out, grad_out, grad_a, grad_b,
            length=length, width=width, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        return grad_a, grad_b


class _RealChunkScan(torch.autograd.Function):
    """Exact differentiable two-level scan for a real diagonal recurrence."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, chunk_size: int):
        if a.shape != b.shape or a.ndim != 3:
            raise ValueError("real chunk scan expects matching [B, L, D] tensors")
        batch, length, width = a.shape
        if chunk_size not in (16, 32, 64) or length % chunk_size:
            raise ValueError("chunk size must be 16, 32 or 64 and divide length")
        a, b = a.contiguous(), b.contiguous()
        chunks = length // chunk_size
        block, width_blocks = _launch_shape(width)
        shape = (batch, chunks, width)
        p, q, chunk_in = (
            torch.empty(shape, device=a.device, dtype=torch.float32)
            for _ in range(3)
        )
        grid = (width_blocks, batch * chunks)
        launch = dict(
            length=length, width=width, chunks=chunks, CHUNK=chunk_size,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _real_chunk_summary[grid](a, b, p, q, **launch)
        _real_chunk_prefix[(width_blocks, batch)](
            p, q, chunk_in, width=width, chunks=chunks, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        out = torch.empty_like(b)
        _real_chunk_replay[grid](a, b, chunk_in, out, **launch)
        ctx.save_for_backward(a, out)
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, out = ctx.saved_tensors
        batch, length, width = a.shape
        chunk_size = ctx.chunk_size
        chunks = length // chunk_size
        grad_out = grad_out.contiguous()
        block, width_blocks = _launch_shape(width)
        shape = (batch, chunks, width)
        p, q, future = (
            torch.empty(shape, device=a.device, dtype=torch.float32)
            for _ in range(3)
        )
        grid = (width_blocks, batch * chunks)
        launch = dict(
            length=length, width=width, chunks=chunks, CHUNK=chunk_size,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _real_reverse_chunk_summary[grid](a, grad_out, p, q, **launch)
        _real_reverse_chunk_prefix[(width_blocks, batch)](
            p, q, future, width=width, chunks=chunks, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        grad_a, grad_b = torch.empty_like(a), torch.empty_like(grad_out)
        _real_reverse_chunk_replay[grid](
            a, out, grad_out, future, grad_a, grad_b, **launch
        )
        return grad_a, grad_b, None


class _ComplexScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ar, ai, br, bi):
        shape = ar.shape
        if any(t.shape != shape for t in (ai, br, bi)) or ar.ndim != 3:
            raise ValueError("complex scan expects matching [B, L, M] tensors")
        ar, ai, br, bi = (t.contiguous() for t in (ar, ai, br, bi))
        batch, length, modes = shape
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        last_r = torch.empty(batch, modes, device=ar.device, dtype=torch.float32)
        last_i = torch.empty_like(last_r)
        block, blocks = _launch_shape(modes)
        _complex_forward[(blocks, batch)](
            ar, ai, br, bi, out_r, out_i, last_r, last_i,
            length=length, modes=modes, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        ctx.save_for_backward(ar, ai, out_r, out_i)
        return out_r, out_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i):
        ar, ai, out_r, out_i = ctx.saved_tensors
        batch, length, modes = ar.shape
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        gradients = [torch.empty_like(ar) for _ in range(4)]
        block, blocks = _launch_shape(modes)
        _complex_backward[(blocks, batch)](
            ar, ai, out_r, out_i, grad_out_r, grad_out_i, *gradients,
            length=length, modes=modes, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        return tuple(gradients)


class _SAMUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eta, delta, nu_log, theta_log, br, bi):
        if eta.shape != delta.shape or eta.ndim != 2:
            raise ValueError("SAMU controls must be matching [B, L] tensors")
        if br.shape != bi.shape or br.ndim != 3:
            raise ValueError("SAMU writes must be matching [B, L, M] tensors")
        batch, length, modes = br.shape
        if eta.shape != (batch, length) or nu_log.shape != (modes,) or theta_log.shape != (modes,):
            raise ValueError("SAMU control, spectrum and write shapes disagree")
        eta, delta, nu_log, theta_log, br, bi = (
            tensor.contiguous() for tensor in (eta, delta, nu_log, theta_log, br, bi)
        )
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        last_r = torch.empty(batch, modes, device=br.device, dtype=torch.float32)
        last_i = torch.empty_like(last_r)
        block = triton.next_power_of_2(modes)
        if block > 512:
            raise ValueError("fused SAMU training scan currently supports at most 512 modes")
        _samu_forward[(batch,)](
            eta, delta, nu_log, theta_log, br, bi,
            out_r, out_i, last_r, last_i,
            length=length, modes=modes, BLOCK=block,
            num_warps=8 if block >= 256 else 4,
        )
        ctx.save_for_backward(eta, delta, nu_log, theta_log, out_r, out_i)
        return out_r, out_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i):
        eta, delta, nu_log, theta_log, out_r, out_i = ctx.saved_tensors
        batch, length, modes = out_r.shape
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        grad_eta = torch.empty_like(eta)
        grad_delta = torch.empty_like(delta)
        grad_nu_log = torch.zeros_like(nu_log)
        grad_theta_log = torch.zeros_like(theta_log)
        grad_br, grad_bi = torch.empty_like(grad_out_r), torch.empty_like(grad_out_i)
        block = triton.next_power_of_2(modes)
        _samu_backward[(batch,)](
            eta, delta, nu_log, theta_log, out_r, out_i,
            grad_out_r, grad_out_i, grad_eta, grad_delta,
            grad_nu_log, grad_theta_log, grad_br, grad_bi,
            length=length, modes=modes, BLOCK=block,
            num_warps=8 if block >= 256 else 4,
        )
        return grad_eta, grad_delta, grad_nu_log, grad_theta_log, grad_br, grad_bi


class _SAMUTiledSerialScan(torch.autograd.Function):
    """Wide, reset-aware SAMU serial scan tiled over modes."""

    @staticmethod
    def forward(
        ctx, eta, delta, nu_log, theta_log, br, bi,
        segment_pos, h0_r, h0_i, block_size: int, reset_first: bool,
    ):
        if eta.shape != delta.shape or eta.ndim != 2:
            raise ValueError("SAMU controls must be matching [B,L] tensors")
        if br.shape != bi.shape or br.ndim != 3:
            raise ValueError("SAMU writes must be matching [B,L,M] tensors")
        batch, length, modes = br.shape
        if eta.shape != (batch, length):
            raise ValueError("SAMU controls and writes disagree")
        if nu_log.shape != (modes,) or theta_log.shape != (modes,):
            raise ValueError("SAMU static spectrum has the wrong shape")
        if block_size not in (32, 64, 128, 256):
            raise ValueError("block_size must be 32, 64, 128 or 256")
        has_segments = segment_pos.numel() != 0
        has_h0 = h0_r.numel() != 0
        if has_segments and segment_pos.shape != (batch, length):
            raise ValueError("segment_pos must have shape [B,L]")
        if has_h0 and (h0_r.shape != (batch, modes) or h0_i.shape != (batch, modes)):
            raise ValueError("initial SAMU states must have shape [B,M]")
        eta, delta, nu_log, theta_log, br, bi = (
            tensor.contiguous()
            for tensor in (eta, delta, nu_log, theta_log, br, bi)
        )
        segment_pos = segment_pos.contiguous()
        h0_r, h0_i = h0_r.contiguous(), h0_i.contiguous()
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        last_r = torch.empty((batch, modes), device=br.device, dtype=torch.float32)
        last_i = torch.empty_like(last_r)
        mode_blocks = triton.cdiv(modes, block_size)
        _samu_tiled_serial_forward[(mode_blocks, batch)](
            eta, delta, nu_log, theta_log, br, bi, segment_pos, h0_r, h0_i,
            out_r, out_i, last_r, last_i,
            length=length, modes=modes, HAS_SEGMENTS=has_segments,
            RESET_FIRST=bool(reset_first), HAS_H0=has_h0, BLOCK=block_size,
            num_warps=1 if block_size <= 64 else (2 if block_size == 128 else 4),
            num_stages=1,
        )
        ctx.save_for_backward(
            eta, delta, nu_log, theta_log, segment_pos, h0_r, h0_i,
            out_r, out_i,
        )
        ctx.block_size = int(block_size)
        ctx.reset_first = bool(reset_first)
        ctx.has_segments = has_segments
        ctx.has_h0 = has_h0
        return out_r, out_i, last_r, last_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i, grad_last_r, grad_last_i):
        (eta, delta, nu_log, theta_log, segment_pos, h0_r, h0_i,
         out_r, out_i) = ctx.saved_tensors
        batch, length, modes = out_r.shape
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        if grad_last_r is None:
            grad_last_r = torch.zeros((batch, modes), device=out_r.device, dtype=torch.float32)
        else:
            grad_last_r = grad_last_r.contiguous().float()
        if grad_last_i is None:
            grad_last_i = torch.zeros((batch, modes), device=out_r.device, dtype=torch.float32)
        else:
            grad_last_i = grad_last_i.contiguous().float()
        block_size = ctx.block_size
        mode_blocks = triton.cdiv(modes, block_size)
        partial_shape = (batch * length, mode_blocks)
        partial_eta = torch.empty(partial_shape, device=out_r.device, dtype=torch.float32)
        partial_delta = torch.empty_like(partial_eta)
        grad_eta, grad_delta = torch.empty_like(eta), torch.empty_like(delta)
        grad_nu_log = torch.zeros_like(nu_log)
        grad_theta_log = torch.zeros_like(theta_log)
        grad_br, grad_bi = torch.empty_like(grad_out_r), torch.empty_like(grad_out_i)
        grad_h0_r = torch.empty((batch, modes), device=out_r.device, dtype=torch.float32)
        grad_h0_i = torch.empty_like(grad_h0_r)
        _samu_tiled_serial_backward[(mode_blocks, batch)](
            eta, delta, nu_log, theta_log, segment_pos, h0_r, h0_i,
            out_r, out_i, grad_out_r, grad_out_i, grad_last_r, grad_last_i,
            partial_eta, partial_delta, grad_nu_log, grad_theta_log,
            grad_br, grad_bi, grad_h0_r, grad_h0_i,
            length=length, modes=modes, mode_blocks=mode_blocks,
            HAS_SEGMENTS=ctx.has_segments, RESET_FIRST=ctx.reset_first,
            HAS_H0=ctx.has_h0, BLOCK=block_size,
            num_warps=1 if block_size <= 64 else (2 if block_size == 128 else 4),
            num_stages=1,
        )
        reduce_block = triton.next_power_of_2(mode_blocks)
        _reduce_shared_control_gradients[(batch * length,)](
            partial_eta, partial_delta, grad_eta, grad_delta,
            rows=batch * length, mode_blocks=mode_blocks,
            REDUCE_BLOCK=reduce_block, num_warps=1,
        )
        return (
            grad_eta, grad_delta, grad_nu_log, grad_theta_log,
            grad_br, grad_bi, None,
            grad_h0_r if ctx.has_h0 else None,
            grad_h0_i if ctx.has_h0 else None,
            None, None,
        )


class _SAMUScanPrecomputedDecay(torch.autograd.Function):
    """SAMU scan that moves mode-wise exponentials off the serial critical path."""

    @staticmethod
    def forward(ctx, eta, delta, nu_log, theta_log, br, bi):
        if eta.shape != delta.shape or eta.ndim != 2:
            raise ValueError("SAMU controls must be matching [B, L] tensors")
        if br.shape != bi.shape or br.ndim != 3:
            raise ValueError("SAMU writes must be matching [B, L, M] tensors")
        batch, length, modes = br.shape
        eta, delta, nu_log, theta_log, br, bi = (
            tensor.contiguous()
            for tensor in (eta, delta, nu_log, theta_log, br, bi)
        )
        rho = torch.empty(batch, length, modes, device=br.device, dtype=torch.float32)
        decay_block, decay_blocks = _launch_shape(modes)
        _samu_precompute_decay[(decay_blocks, batch * length)](
            eta, nu_log, rho, rows=batch * length, modes=modes,
            BLOCK=decay_block,
            num_warps=4 if decay_block >= 64 else 2,
        )
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        last_r = torch.empty(batch, modes, device=br.device, dtype=torch.float32)
        last_i = torch.empty_like(last_r)
        block = triton.next_power_of_2(modes)
        _samu_forward_precomputed[(batch,)](
            rho, delta, theta_log, br, bi, out_r, out_i, last_r, last_i,
            length=length, modes=modes, BLOCK=block,
            num_warps=8 if block >= 256 else 4,
        )
        ctx.save_for_backward(eta, delta, nu_log, theta_log, rho, out_r, out_i)
        return out_r, out_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i):
        eta, delta, nu_log, theta_log, rho, out_r, out_i = ctx.saved_tensors
        batch, length, modes = out_r.shape
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        grad_eta, grad_delta = torch.empty_like(eta), torch.empty_like(delta)
        grad_nu_log, grad_theta_log = torch.zeros_like(nu_log), torch.zeros_like(theta_log)
        grad_br, grad_bi = torch.empty_like(grad_out_r), torch.empty_like(grad_out_i)
        block = triton.next_power_of_2(modes)
        _samu_backward_precomputed[(batch,)](
            eta, delta, nu_log, theta_log, rho, out_r, out_i,
            grad_out_r, grad_out_i, grad_eta, grad_delta,
            grad_nu_log, grad_theta_log, grad_br, grad_bi,
            length=length, modes=modes, BLOCK=block,
            num_warps=8 if block >= 256 else 4,
        )
        return grad_eta, grad_delta, grad_nu_log, grad_theta_log, grad_br, grad_bi


class _SAMUChunkScan(torch.autograd.Function):
    """Exact two-level scan with parallel forward and reverse chunk replay."""

    @staticmethod
    def forward(ctx, eta, delta, nu_log, theta_log, br, bi, raw_x,
                chunk_size: int,
                hierarchical_prefix: bool, precompute_shared: bool,
                compressed_transition: bool, atomic_shared: bool,
                two_stage_spectral: bool, compact_control_cache: bool,
                precompute_spectral: bool, fused_write: bool,
                mode_block_size: int, backward_chunk_group: int,
                fused_output_relu: bool, prefix_group_size: int,
                phase_direction, radial_direction,
                phase_amplitude, radial_amplitude,
                phase_coordinate, radial_raw,
                fused_controller_backward: bool):
        if eta.shape != delta.shape:
            raise ValueError("SAMU control or write shapes disagree")
        batch, length = eta.shape
        modes = nu_log.numel()
        if fused_write:
            if raw_x.shape != (batch, length, 2 * modes):
                raise ValueError("fused SAMU write expects raw_x [B,L,2M]")
        elif br.shape != (batch, length, modes) or bi.shape != br.shape:
            raise ValueError("SAMU control or write shapes disagree")
        if chunk_size not in (16, 32, 64) or length % chunk_size:
            raise ValueError("chunk size must be 16, 32 or 64 and divide length")
        eta, delta, nu_log, theta_log, br, bi, raw_x = (
            tensor.contiguous()
            for tensor in (eta, delta, nu_log, theta_log, br, bi, raw_x)
        )
        if fused_write:
            # O(M), rather than O(BLM): retain the canonical special-function
            # evaluation and only fuse the state-sized normalization multiply.
            write_nu = torch.exp(nu_log)
            write_gamma = (
                torch.sqrt(1.0 - torch.exp(-2.0 * write_nu)) + 1.0e-8
            )
        else:
            write_gamma = nu_log.new_empty((0,), dtype=torch.float32)
        chunks = length // chunk_size
        if mode_block_size not in (32, 64, 128, 256):
            raise ValueError("mode block size must be 32, 64, 128 or 256")
        if backward_chunk_group not in (1, 2):
            raise ValueError("backward chunk group must be 1 or 2")
        if chunks % backward_chunk_group:
            raise ValueError("backward chunk group must divide the chunk count")
        if backward_chunk_group > 1 and two_stage_spectral:
            raise ValueError("grouped replay and two-stage spectral are exclusive")
        if backward_chunk_group > 1 and fused_output_relu:
            raise ValueError("grouped replay and fused output are exclusive")
        if abs(prefix_group_size) not in (0, 32, 64, 128):
            raise ValueError("prefix group size must be 0, 32, 64 or 128")
        if prefix_group_size and not hierarchical_prefix:
            raise ValueError("a grouped prefix requires hierarchical_prefix")
        # A negative group size is an internal dispatch encoding for the
        # very-long correctness path: preserve serial forward parenthesization
        # while using the conjugate grouped hierarchy in backward.
        serial_forward_prefix = prefix_group_size < 0
        forward_hierarchical_prefix = (
            bool(hierarchical_prefix) and not serial_forward_prefix
        )
        forward_prefix_group_size = max(prefix_group_size, 0)
        backward_prefix_group_size = abs(prefix_group_size)
        if fused_controller_backward:
            if not fused_write:
                raise ValueError("fused controller backward requires fused write")
            if phase_direction.shape != (2 * modes + 1,):
                raise ValueError("phase controller direction has wrong shape")
            if radial_direction.shape != phase_direction.shape:
                raise ValueError("radial controller direction has wrong shape")
            if phase_coordinate.shape != eta.shape or radial_raw.shape != eta.shape:
                raise ValueError("controller coordinate cache has wrong shape")
        block = min(mode_block_size, triton.next_power_of_2(modes))
        mode_blocks = triton.cdiv(modes, block)
        if precompute_spectral:
            spectrum = tuple(
                torch.empty_like(nu_log, dtype=torch.float32) for _ in range(4)
            )
            spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin = spectrum
            spectral_block = min(256, triton.next_power_of_2(modes))
            _samu_precompute_static_spectrum[(triton.cdiv(modes, spectral_block),)](
                nu_log, theta_log, *spectrum,
                modes=modes, BLOCK=spectral_block, num_warps=4,
            )
        else:
            spectrum_nu = spectrum_theta = nu_log.new_empty((0,), dtype=torch.float32)
            spectrum_cos = spectrum_sin = spectrum_nu
        shape = (batch, chunks, modes)
        qr, qi, chunk_in_r, chunk_in_i = (
            torch.empty(shape, device=br.device, dtype=torch.float32)
            for _ in range(4))
        if compressed_transition:
            compressed_g = torch.empty(
                (batch, chunks), device=br.device, dtype=torch.float32)
            compressed_d = torch.empty_like(compressed_g)
        else:
            compressed_g = compressed_d = br.new_empty((0,), dtype=torch.float32)
        if compressed_transition and not hierarchical_prefix:
            pr = pi = br.new_empty((0,), dtype=torch.float32)
        else:
            pr = torch.empty(shape, device=br.device, dtype=torch.float32)
            pi = torch.empty_like(pr)
        grid = (mode_blocks, batch * chunks)
        if precompute_shared:
            exp_eta_shared = torch.empty_like(eta, dtype=torch.float32)
            cos_delta_shared = torch.empty_like(delta, dtype=torch.float32)
            sin_delta_shared = torch.empty_like(delta, dtype=torch.float32)
            shared_rows = batch * length
            _samu_precompute_shared_controls[(triton.cdiv(shared_rows, 256),)](
                eta, delta, exp_eta_shared, cos_delta_shared, sin_delta_shared,
                rows=shared_rows, BLOCK=256, num_warps=4,
            )
        else:
            exp_eta_shared = eta.new_empty((0,), dtype=torch.float32)
            cos_delta_shared = delta.new_empty((0,), dtype=torch.float32)
            sin_delta_shared = delta.new_empty((0,), dtype=torch.float32)
        launch = dict(
            length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
            PRECOMPUTED_SHARED=bool(precompute_shared),
            PRECOMPUTED_SPECTRAL=bool(precompute_spectral),
            COMPRESSED_TRANSITION=bool(compressed_transition),
            FUSED_WRITE=bool(fused_write),
            ROUND_WRITE_BF16=bool(fused_write and raw_x.dtype == torch.bfloat16),
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _samu_chunk_summary[grid](
            eta, delta, nu_log, theta_log, br, bi, raw_x, write_gamma,
            exp_eta_shared, cos_delta_shared, sin_delta_shared,
            spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
            compressed_g, compressed_d,
            pr, pi, qr, qi, **launch
        )
        if compressed_transition and hierarchical_prefix:
            _samu_reconstruct_chunk_transition[grid](
                compressed_g, compressed_d, nu_log, theta_log, pr, pi,
                spectrum_nu, spectrum_theta,
                modes=modes, chunks=chunks, CHUNK=chunk_size,
                REVERSE=False, PRECOMPUTED_SPECTRAL=bool(precompute_spectral),
                BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        if compressed_transition and not hierarchical_prefix:
            _samu_compressed_chunk_prefix[(mode_blocks, batch)](
                compressed_g, compressed_d, nu_log, theta_log, qr, qi,
                spectrum_nu, spectrum_theta,
                chunk_in_r, chunk_in_i,
                modes=modes, chunks=chunks, CHUNK=chunk_size,
                REVERSE=False, PRECOMPUTED_SPECTRAL=bool(precompute_spectral),
                BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        elif (forward_hierarchical_prefix and forward_prefix_group_size
              and chunks > 1):
            groups = triton.cdiv(chunks, forward_prefix_group_size)
            group_shape = (batch, groups, modes)
            group_pr, group_pi, group_qr, group_qi = (
                torch.empty(group_shape, device=pr.device, dtype=torch.float32)
                for _ in range(4)
            )
            group_boundary_r = torch.empty_like(group_pr)
            group_boundary_i = torch.empty_like(group_pr)
            group_grid = (mode_blocks, batch * groups)
            _complex_affine_group_local[group_grid](
                pr, pi, qr, qi, group_pr, group_pi, group_qr, group_qi,
                modes=modes, chunks=chunks, groups=groups,
                GROUP=forward_prefix_group_size, REVERSE=False, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
            _complex_affine_group_outer[(mode_blocks, batch)](
                group_pr, group_pi, group_qr, group_qi,
                group_boundary_r, group_boundary_i,
                modes=modes, groups=groups, REVERSE=False, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
            _complex_affine_group_correct[grid](
                pr, pi, qr, qi, group_boundary_r, group_boundary_i,
                chunk_in_r, chunk_in_i,
                modes=modes, chunks=chunks, groups=groups,
                GROUP=forward_prefix_group_size, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        elif forward_hierarchical_prefix and chunks > 1:
            temporary = tuple(torch.empty_like(pr) for _ in range(4))
            source = (pr, pi, qr, qi)
            destination = temporary
            stride = 1
            while stride < chunks:
                _complex_affine_prefix_stage[grid](
                    *source, *destination,
                    modes=modes, chunks=chunks, STRIDE=stride,
                    REVERSE=False, BLOCK=block,
                    num_warps=4 if block >= 64 else 2,
                )
                source, destination = destination, source
                stride *= 2
            _complex_affine_prefix_extract[grid](
                source[2], source[3], chunk_in_r, chunk_in_i,
                modes=modes, chunks=chunks, REVERSE=False, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        else:
            _samu_chunk_prefix[(mode_blocks, batch)](
                pr, pi, qr, qi, chunk_in_r, chunk_in_i,
                modes=modes, chunks=chunks, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        output_reference = raw_x if fused_write else br
        out_shape = (batch, length, modes)
        out_r = torch.empty(out_shape, device=output_reference.device,
                            dtype=output_reference.dtype)
        out_i = torch.empty_like(out_r)
        if fused_output_relu:
            visible_out = torch.empty(
                (batch, length, 2 * modes), device=output_reference.device,
                dtype=output_reference.dtype,
            )
        else:
            visible_out = output_reference.new_empty((0,))
        _samu_chunk_replay[grid](
            eta, delta, nu_log, theta_log, br, bi, raw_x, write_gamma,
            exp_eta_shared, cos_delta_shared, sin_delta_shared,
            spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
            chunk_in_r, chunk_in_i, out_r, out_i, visible_out,
            FUSED_OUTPUT_RELU=bool(fused_output_relu), **launch
        )
        # E/C/S fully determine every transition used by backward.  Avoid
        # retaining duplicate eta/delta buffers on this path; their gradients
        # are still returned with the original [B,L] shape to the controller.
        compact_controls = precompute_shared and compact_control_cache
        saved_eta = eta.new_empty((0,)) if compact_controls else eta
        saved_delta = delta.new_empty((0,)) if compact_controls else delta
        ctx.save_for_backward(
            saved_eta, saved_delta, nu_log, theta_log, out_r, out_i,
            exp_eta_shared, cos_delta_shared, sin_delta_shared,
            spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
            compressed_g, compressed_d, raw_x, write_gamma,
            phase_direction, radial_direction,
            phase_amplitude, radial_amplitude,
            phase_coordinate, radial_raw,
        )
        ctx.control_shape = (batch, length)
        ctx.chunk_size = chunk_size
        ctx.hierarchical_prefix = bool(hierarchical_prefix)
        ctx.precompute_shared = bool(precompute_shared)
        ctx.compressed_transition = bool(compressed_transition)
        ctx.atomic_shared = bool(atomic_shared)
        ctx.two_stage_spectral = bool(two_stage_spectral)
        ctx.precompute_spectral = bool(precompute_spectral)
        ctx.fused_write = bool(fused_write)
        ctx.round_write_bf16 = bool(fused_write and raw_x.dtype == torch.bfloat16)
        ctx.mode_block_size = int(mode_block_size)
        ctx.backward_chunk_group = int(backward_chunk_group)
        ctx.fused_output_relu = bool(fused_output_relu)
        ctx.prefix_group_size = int(backward_prefix_group_size)
        ctx.fused_controller_backward = bool(fused_controller_backward)
        return visible_out if fused_output_relu else (out_r, out_i)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (eta, delta, nu_log, theta_log, out_r, out_i,
         exp_eta_shared, cos_delta_shared,
         sin_delta_shared, spectrum_nu, spectrum_theta,
         spectrum_cos, spectrum_sin, compressed_g,
         compressed_d, raw_x, write_gamma,
         phase_direction, radial_direction,
         phase_amplitude, radial_amplitude,
         phase_coordinate, radial_raw) = ctx.saved_tensors
        batch, length, modes = out_r.shape
        chunk_size = ctx.chunk_size
        chunks = length // chunk_size
        block = min(ctx.mode_block_size, triton.next_power_of_2(modes))
        mode_blocks = triton.cdiv(modes, block)
        if ctx.fused_output_relu:
            grad_visible = grad_outputs[0].contiguous()
            grad_out_r = grad_out_i = out_r.new_empty((0,))
        else:
            grad_out_r = grad_outputs[0].contiguous()
            grad_out_i = grad_outputs[1].contiguous()
            grad_visible = out_r.new_empty((0,))
        shape = (batch, chunks, modes)
        qr, qi, future_r, future_i = (
            torch.empty(shape, device=out_r.device, dtype=torch.float32)
            for _ in range(4))
        if ctx.compressed_transition and not ctx.hierarchical_prefix:
            pr = pi = out_r.new_empty((0,), dtype=torch.float32)
        else:
            pr = torch.empty(shape, device=out_r.device, dtype=torch.float32)
            pi = torch.empty_like(pr)
        grid = (mode_blocks, batch * chunks)
        launch = dict(
            length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
            PRECOMPUTED_SHARED=ctx.precompute_shared,
            PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
            COMPRESSED_TRANSITION=ctx.compressed_transition,
            FUSED_WRITE=ctx.fused_write,
            ROUND_WRITE_BF16=ctx.round_write_bf16,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _samu_reverse_chunk_summary[grid](
            eta, delta, nu_log, theta_log,
            exp_eta_shared, cos_delta_shared, sin_delta_shared,
            spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
            out_r, out_i, grad_out_r, grad_out_i, grad_visible,
            pr, pi, qr, qi,
            length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
            PRECOMPUTED_SHARED=ctx.precompute_shared,
            PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
            COMPRESSED_TRANSITION=ctx.compressed_transition,
            FUSED_OUTPUT_RELU=ctx.fused_output_relu,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        if ctx.compressed_transition and ctx.hierarchical_prefix:
            _samu_reconstruct_chunk_transition[grid](
                compressed_g, compressed_d, nu_log, theta_log, pr, pi,
                spectrum_nu, spectrum_theta,
                modes=modes, chunks=chunks, CHUNK=chunk_size,
                REVERSE=True, PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
                BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        if ctx.compressed_transition and not ctx.hierarchical_prefix:
            _samu_compressed_chunk_prefix[(mode_blocks, batch)](
                compressed_g, compressed_d, nu_log, theta_log, qr, qi,
                spectrum_nu, spectrum_theta,
                future_r, future_i,
                modes=modes, chunks=chunks, CHUNK=chunk_size,
                REVERSE=True, PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
                BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        elif ctx.hierarchical_prefix and ctx.prefix_group_size and chunks > 1:
            groups = triton.cdiv(chunks, ctx.prefix_group_size)
            group_shape = (batch, groups, modes)
            group_pr, group_pi, group_qr, group_qi = (
                torch.empty(group_shape, device=pr.device, dtype=torch.float32)
                for _ in range(4)
            )
            group_boundary_r = torch.empty_like(group_pr)
            group_boundary_i = torch.empty_like(group_pr)
            group_grid = (mode_blocks, batch * groups)
            _complex_affine_group_local[group_grid](
                pr, pi, qr, qi, group_pr, group_pi, group_qr, group_qi,
                modes=modes, chunks=chunks, groups=groups,
                GROUP=ctx.prefix_group_size, REVERSE=True, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
            _complex_affine_group_outer[(mode_blocks, batch)](
                group_pr, group_pi, group_qr, group_qi,
                group_boundary_r, group_boundary_i,
                modes=modes, groups=groups, REVERSE=True, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
            _complex_affine_group_correct[grid](
                pr, pi, qr, qi, group_boundary_r, group_boundary_i,
                future_r, future_i,
                modes=modes, chunks=chunks, groups=groups,
                GROUP=ctx.prefix_group_size, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        elif ctx.hierarchical_prefix and chunks > 1:
            temporary = tuple(torch.empty_like(pr) for _ in range(4))
            source = (pr, pi, qr, qi)
            destination = temporary
            stride = 1
            while stride < chunks:
                _complex_affine_prefix_stage[grid](
                    *source, *destination,
                    modes=modes, chunks=chunks, STRIDE=stride,
                    REVERSE=True, BLOCK=block,
                    num_warps=4 if block >= 64 else 2,
                )
                source, destination = destination, source
                stride *= 2
            _complex_affine_prefix_extract[grid](
                source[2], source[3], future_r, future_i,
                modes=modes, chunks=chunks, REVERSE=True, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        else:
            _samu_reverse_chunk_prefix[(mode_blocks, batch)](
                pr, pi, qr, qi, future_r, future_i,
                modes=modes, chunks=chunks, BLOCK=block,
                num_warps=4 if block >= 64 else 2,
            )
        if ctx.atomic_shared:
            partial_eta = partial_delta = out_r.new_empty((0,), dtype=torch.float32)
            grad_eta = torch.zeros(ctx.control_shape, device=out_r.device, dtype=torch.float32)
            grad_delta = torch.zeros_like(grad_eta)
        else:
            partial_shape = (batch * length, mode_blocks)
            partial_eta = torch.empty(partial_shape, device=out_r.device, dtype=torch.float32)
            partial_delta = torch.empty_like(partial_eta)
            grad_eta = torch.empty(ctx.control_shape, device=out_r.device, dtype=torch.float32)
            grad_delta = torch.empty_like(grad_eta)
        if ctx.two_stage_spectral:
            # qr/qi are dead once the reverse outer-prefix boundary has been
            # produced. Reuse their storage for per-chunk spectral partials.
            partial_nu_log, partial_theta_log = qr, qi
            grad_nu_log = grad_theta_log = out_r.new_empty(
                (0,), dtype=torch.float32
            )
        else:
            partial_nu_log = partial_theta_log = out_r.new_empty(
                (0,), dtype=torch.float32
            )
            grad_nu_log = torch.zeros_like(nu_log)
            grad_theta_log = torch.zeros_like(theta_log)
        if ctx.fused_write:
            grad_br = grad_bi = out_r.new_empty((0,))
            grad_raw_x = torch.empty_like(raw_x)
        else:
            grad_br = torch.empty_like(out_r)
            grad_bi = torch.empty_like(out_i)
            grad_raw_x = raw_x.new_empty((0,))
        if ctx.backward_chunk_group == 1:
            _samu_reverse_chunk_replay[grid](
                eta, delta, nu_log, theta_log, raw_x, write_gamma,
                exp_eta_shared, cos_delta_shared, sin_delta_shared,
                spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
                out_r, out_i,
                grad_out_r, grad_out_i, grad_visible, future_r, future_i,
                partial_eta, partial_delta, grad_eta, grad_delta,
                grad_nu_log, grad_theta_log, partial_nu_log, partial_theta_log,
                grad_br, grad_bi, grad_raw_x, mode_blocks=mode_blocks,
                ATOMIC_SHARED=ctx.atomic_shared,
                TWO_STAGE_SPECTRAL=ctx.two_stage_spectral,
                FUSED_OUTPUT_RELU=ctx.fused_output_relu,
                length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
                PRECOMPUTED_SHARED=ctx.precompute_shared,
                PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
                COMPRESSED_TRANSITION=ctx.compressed_transition,
                FUSED_WRITE=ctx.fused_write,
                ROUND_WRITE_BF16=ctx.round_write_bf16,
                BLOCK=block, num_warps=4 if block >= 64 else 2,
            )
        else:
            chunk_groups = chunks // ctx.backward_chunk_group
            replay_grid = (mode_blocks, batch * chunk_groups)
            _samu_reverse_chunk_replay_grouped[replay_grid](
                eta, delta, nu_log, theta_log, raw_x, write_gamma,
                exp_eta_shared, cos_delta_shared, sin_delta_shared,
                spectrum_nu, spectrum_theta, spectrum_cos, spectrum_sin,
                out_r, out_i, grad_out_r, grad_out_i, future_r, future_i,
                partial_eta, partial_delta, grad_eta, grad_delta,
                grad_nu_log, grad_theta_log, grad_br, grad_bi, grad_raw_x,
                length=length, modes=modes, chunks=chunks,
                chunk_groups=chunk_groups, mode_blocks=mode_blocks,
                CHUNK=chunk_size,
                CHUNKS_PER_PROGRAM=ctx.backward_chunk_group,
                PRECOMPUTED_SHARED=ctx.precompute_shared,
                PRECOMPUTED_SPECTRAL=ctx.precompute_spectral,
                ATOMIC_SHARED=ctx.atomic_shared,
                FUSED_WRITE=ctx.fused_write,
                ROUND_WRITE_BF16=ctx.round_write_bf16,
                BLOCK=block, num_warps=4 if block >= 64 else 2,
            )
        if not ctx.atomic_shared:
            reduce_block = triton.next_power_of_2(mode_blocks)
            _reduce_shared_control_gradients[(batch * length,)](
                partial_eta, partial_delta, grad_eta, grad_delta,
                rows=batch * length, mode_blocks=mode_blocks,
                REDUCE_BLOCK=reduce_block, num_warps=1,
            )
        if ctx.two_stage_spectral:
            grad_nu_log = partial_nu_log.sum(dim=(0, 1))
            grad_theta_log = partial_theta_log.sum(dim=(0, 1))
        grad_phase_direction = grad_radial_direction = None
        grad_phase_amplitude = grad_radial_amplitude = None
        if ctx.fused_controller_backward:
            rows = batch * length
            width = 2 * modes
            row_chunk = 128
            row_chunks = triton.cdiv(rows, row_chunk)
            augmented_width = width + 1
            direction_partial = torch.empty(
                (row_chunks, 2, augmented_width),
                device=raw_x.device, dtype=torch.float32,
            )
            amplitude_partial = torch.empty(
                (row_chunks, 2), device=raw_x.device, dtype=torch.float32
            )
            combined_grad_x = torch.empty_like(raw_x)
            phase_norm = torch.linalg.vector_norm(phase_direction.float()).clamp_min(1.0e-12)
            radial_norm = torch.linalg.vector_norm(radial_direction.float()).clamp_min(1.0e-12)
            normalized_phase = (phase_direction.float() / phase_norm).contiguous()
            normalized_radial = (radial_direction.float() / radial_norm).contiguous()
            controller_block = 128
            controller_width_blocks = triton.cdiv(augmented_width, controller_block)
            _samu_fused_controller_backward[(controller_width_blocks, row_chunks)](
                raw_x, grad_raw_x, grad_eta, grad_delta,
                phase_coordinate, radial_raw,
                normalized_phase, normalized_radial,
                phase_amplitude, radial_amplitude,
                combined_grad_x, direction_partial, amplitude_partial,
                rows=rows, width=width, row_chunks=row_chunks,
                ROW_CHUNK=row_chunk, SCALE=1.0 / (modes ** 0.5),
                ROUND_INPUT_BF16=raw_x.dtype == torch.bfloat16,
                BLOCK=controller_block, num_warps=4,
            )
            grad_normalized = direction_partial.sum(dim=0)
            phase_dot = torch.sum(grad_normalized[0] * normalized_phase)
            radial_dot = torch.sum(grad_normalized[1] * normalized_radial)
            grad_phase_direction = (
                grad_normalized[0] - normalized_phase * phase_dot
            ) / phase_norm
            grad_radial_direction = (
                grad_normalized[1] - normalized_radial * radial_dot
            ) / radial_norm
            amplitude_gradients = amplitude_partial.sum(dim=0)
            grad_phase_amplitude = amplitude_gradients[0]
            grad_radial_amplitude = amplitude_gradients[1]
            grad_raw_x = combined_grad_x
        return (
            None if ctx.fused_controller_backward else grad_eta,
            None if ctx.fused_controller_backward else grad_delta,
            grad_nu_log, grad_theta_log,
            None if ctx.fused_write else grad_br,
            None if ctx.fused_write else grad_bi,
            grad_raw_x if ctx.fused_write else None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None,
            grad_phase_direction, grad_radial_direction,
            grad_phase_amplitude, grad_radial_amplitude,
            None, None, None,
        )


def real_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _RealScan.apply(a, b)


def real_chunk_scan(
    a: torch.Tensor, b: torch.Tensor, chunk_size: int = 16
) -> torch.Tensor:
    return _RealChunkScan.apply(a, b, chunk_size)


def complex_scan(ar: torch.Tensor, ai: torch.Tensor,
                 br: torch.Tensor, bi: torch.Tensor):
    return _ComplexScan.apply(ar, ai, br, bi)


def samu_scan(eta: torch.Tensor, delta: torch.Tensor,
              nu_log: torch.Tensor, theta_log: torch.Tensor,
              br: torch.Tensor, bi: torch.Tensor):
    return _SAMUScan.apply(eta, delta, nu_log, theta_log, br, bi)


def samu_tiled_serial_scan(
    eta: torch.Tensor,
    delta: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
    *,
    segment_pos: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    block_size: int = 128,
    reset_first: bool = False,
):
    """Run the wide fused serial SAMU scan and return outputs plus FP32 state."""
    empty_segment = eta.new_empty((0,), dtype=torch.int32)
    empty_state = br.new_empty((0,), dtype=torch.float32)
    if initial_state is None:
        h0_r, h0_i = empty_state, empty_state
    else:
        h0_r, h0_i = initial_state
    return _SAMUTiledSerialScan.apply(
        eta, delta, nu_log, theta_log, br, bi,
        empty_segment if segment_pos is None else segment_pos,
        h0_r, h0_i, int(block_size), bool(reset_first),
    )


def samu_scan_precomputed_decay(
    eta: torch.Tensor,
    delta: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
):
    return _SAMUScanPrecomputedDecay.apply(
        eta, delta, nu_log, theta_log, br, bi
    )


def samu_chunk_scan(
    eta: torch.Tensor,
    delta: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
    chunk_size: int = 32,
    hierarchical_prefix: bool = False,
    precompute_shared: bool = False,
    compressed_transition: bool = False,
    atomic_shared: bool = False,
    two_stage_spectral: bool = False,
    compact_control_cache: bool = False,
    precompute_spectral: bool = False,
    mode_block_size: int = 128,
    raw_x: torch.Tensor | None = None,
    fused_write: bool = False,
    backward_chunk_group: int = 1,
    fused_output_relu: bool = False,
    prefix_group_size: int = 0,
    controller_backward_cache: tuple[torch.Tensor, ...] | None = None,
):
    if fused_write:
        if raw_x is None:
            raise ValueError("raw_x is required by the fused-write SAMU path")
    else:
        raw_x = br.new_empty((0,))
    if controller_backward_cache is None:
        empty = raw_x.new_empty((0,), dtype=torch.float32)
        phase_direction = radial_direction = empty
        phase_amplitude = radial_amplitude = empty
        phase_coordinate = radial_raw = empty
        fused_controller_backward = False
    else:
        (phase_direction, radial_direction,
         phase_amplitude, radial_amplitude,
         phase_coordinate, radial_raw) = controller_backward_cache
        fused_controller_backward = True
    return _SAMUChunkScan.apply(
        eta, delta, nu_log, theta_log, br, bi, raw_x, chunk_size,
        hierarchical_prefix, precompute_shared, compressed_transition,
        atomic_shared, two_stage_spectral, compact_control_cache,
        precompute_spectral, fused_write, mode_block_size, backward_chunk_group,
        fused_output_relu, prefix_group_size,
        phase_direction, radial_direction,
        phase_amplitude, radial_amplitude,
        phase_coordinate, radial_raw, fused_controller_backward,
    )


def samu_training_scan_dispatch(
    eta: torch.Tensor,
    delta: torch.Tensor,
    nu_log: torch.Tensor,
    theta_log: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
    *,
    chunk_size: int = 16,
    hierarchical_prefix: bool = False,
    precompute_shared: bool = False,
    compressed_transition: bool = False,
    atomic_shared: bool = False,
    two_stage_spectral: bool = False,
    compact_control_cache: bool = False,
    precompute_spectral: bool = False,
    raw_x: torch.Tensor | None = None,
    fused_write: bool = False,
    backward_chunk_group: int = 1,
    fused_output_relu: bool = False,
    prefix_group_size: int = 0,
    controller_backward_cache: tuple[torch.Tensor, ...] | None = None,
    segment_pos: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    reset_first: bool = False,
    block_size: int = 128,
):
    """Correctness-first dispatch for the canonical training recurrence.

    The chunk backend currently has a deliberately narrow fast-path contract:
    zero initial state, no packed/internal reset and a complete final chunk.
    Shapes outside that contract fall back to the exact reset-aware tiled
    serial backend.  In particular, compressed G/D is never allowed to treat
    padding as real tokens or silently drop reset semantics.
    """
    has_internal_reset = segment_pos is not None
    has_initial_state = initial_state is not None
    has_partial_chunk = eta.shape[1] % chunk_size != 0
    if fused_write and raw_x is None:
        raise ValueError("raw_x is required by the fused-write SAMU path")
    if has_internal_reset or has_initial_state or has_partial_chunk:
        if controller_backward_cache is not None:
            raise ValueError(
                "fused controller backward does not support fallback semantics"
            )
        if fused_write:
            nu = torch.exp(nu_log)
            gamma = torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8
            modes = nu_log.numel()
            br = (raw_x[..., :modes].float() * gamma).to(raw_x.dtype)
            bi = (raw_x[..., modes:].float() * gamma).to(raw_x.dtype)
        out_r, out_i, _, _ = samu_tiled_serial_scan(
            eta, delta, nu_log, theta_log, br, bi,
            segment_pos=segment_pos,
            initial_state=initial_state,
            block_size=block_size,
            reset_first=reset_first,
        )
        if fused_output_relu:
            return torch.cat((out_r, out_i), dim=-1).relu()
        return out_r, out_i
    return samu_chunk_scan(
        eta, delta, nu_log, theta_log, br, bi,
        chunk_size, hierarchical_prefix, precompute_shared,
        compressed_transition, atomic_shared, two_stage_spectral,
        compact_control_cache, precompute_spectral,
        mode_block_size=block_size, raw_x=raw_x, fused_write=fused_write,
        backward_chunk_group=backward_chunk_group,
        fused_output_relu=fused_output_relu,
        prefix_group_size=prefix_group_size,
        controller_backward_cache=controller_backward_cache,
    )


@torch.no_grad()
def real_scan_with_last(a: torch.Tensor, b: torch.Tensor):
    """Inference scan returning the FP32 final state without BF16 round-trip."""
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("real scan expects matching [B, L, D] tensors")
    a, b = a.contiguous(), b.contiguous()
    batch, length, width = a.shape
    out = torch.empty_like(b)
    last = torch.empty(batch, width, device=a.device, dtype=torch.float32)
    block, blocks = _launch_shape(width)
    _real_forward[(blocks, batch)](
        a, b, out, last, length=length, width=width, BLOCK=block,
        num_warps=4 if block >= 64 else 2,
    )
    return out, last


@torch.no_grad()
def complex_scan_with_last(ar: torch.Tensor, ai: torch.Tensor,
                           br: torch.Tensor, bi: torch.Tensor):
    """Inference scan returning both FP32 final-state components."""
    shape = ar.shape
    if any(t.shape != shape for t in (ai, br, bi)) or ar.ndim != 3:
        raise ValueError("complex scan expects matching [B, L, M] tensors")
    ar, ai, br, bi = (t.contiguous() for t in (ar, ai, br, bi))
    batch, length, modes = shape
    out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
    last_r = torch.empty(batch, modes, device=ar.device, dtype=torch.float32)
    last_i = torch.empty_like(last_r)
    block, blocks = _launch_shape(modes)
    _complex_forward[(blocks, batch)](
        ar, ai, br, bi, out_r, out_i, last_r, last_i,
        length=length, modes=modes, BLOCK=block,
        num_warps=4 if block >= 64 else 2,
    )
    return out_r, out_i, last_r, last_i


@torch.no_grad()
def samu_scan_with_last(eta: torch.Tensor, delta: torch.Tensor,
                        nu_log: torch.Tensor, theta_log: torch.Tensor,
                        br: torch.Tensor, bi: torch.Tensor):
    if eta.shape != delta.shape or br.shape != bi.shape:
        raise ValueError("SAMU control or write shapes disagree")
    eta, delta, nu_log, theta_log, br, bi = (
        tensor.contiguous() for tensor in (eta, delta, nu_log, theta_log, br, bi)
    )
    batch, length, modes = br.shape
    out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
    last_r = torch.empty(batch, modes, device=br.device, dtype=torch.float32)
    last_i = torch.empty_like(last_r)
    block = triton.next_power_of_2(modes)
    _samu_forward[(batch,)](
        eta, delta, nu_log, theta_log, br, bi,
        out_r, out_i, last_r, last_i,
        length=length, modes=modes, BLOCK=block,
        num_warps=8 if block >= 256 else 4,
    )
    return out_r, out_i, last_r, last_i


@torch.no_grad()
def samu_decode(x: torch.Tensor, phase_direction: torch.Tensor,
                radial_direction: torch.Tensor, phase_amplitude: torch.Tensor,
                radial_amplitude: torch.Tensor, nu_log: torch.Tensor,
                theta_log: torch.Tensor, state_r: torch.Tensor,
                state_i: torch.Tensor):
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("SAMU decode expects [B, 2M]")
    x = x.contiguous()
    batch, width = x.shape
    modes = width // 2
    if state_r.shape != (batch, modes) or state_i.shape != state_r.shape:
        raise ValueError("SAMU decode state shape mismatch")
    out = torch.empty_like(x)
    last_r = torch.empty_like(state_r, dtype=torch.float32)
    last_i = torch.empty_like(last_r)
    block = triton.next_power_of_2(width)
    _samu_decode[(batch,)](
        x, phase_direction, radial_direction,
        phase_amplitude, radial_amplitude, nu_log, theta_log,
        state_r, state_i, out, last_r, last_i,
        width=width, modes=modes, SCALE=1.0 / (modes ** 0.5), BLOCK=block,
        num_warps=8 if block >= 256 else 4,
    )
    return out, (last_r, last_i)


@torch.no_grad()
def samu_decode_blocked(
    x: torch.Tensor,
    normalized_phase_direction: torch.Tensor,
    normalized_radial_direction: torch.Tensor,
    phase_scale: torch.Tensor,
    radial_scale: torch.Tensor,
    nu: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    gamma: torch.Tensor,
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    block_m: int = 32,
):
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("blocked SAMU decode expects [B, 2M]")
    x = x.contiguous()
    batch, width = x.shape
    modes = width // 2
    out = torch.empty_like(x)
    last_r = torch.empty_like(state_r, dtype=torch.float32)
    last_i = torch.empty_like(state_i, dtype=torch.float32)
    block_d = triton.next_power_of_2(width)
    _samu_decode_blocked[(triton.cdiv(modes, block_m), batch)](
        x, normalized_phase_direction, normalized_radial_direction,
        phase_scale, radial_scale, nu, cos_theta, sin_theta, gamma,
        state_r, state_i, out, last_r, last_i,
        width=width, modes=modes, BLOCK_D=block_d, BLOCK_M=block_m,
        num_warps=8 if block_d >= 256 else 4,
    )
    return out, (last_r, last_i)
