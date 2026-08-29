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
def _samu_chunk_summary(eta, delta, nu_log, theta_log, br, bi,
                        pr, pi, qr, qi,
                        length: tl.constexpr, modes: tl.constexpr,
                        chunks: tl.constexpr, CHUNK: tl.constexpr,
                        BLOCK: tl.constexpr):
    """Summarize each independent time chunk as z -> P*z + Q."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    px = tl.full((BLOCK,), 1.0, tl.float32)
    py = tl.zeros((BLOCK,), tl.float32)
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        radius = tl.exp(-nu * tl.exp(radial))
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        next_px = ar * px - ai * py
        next_py = ai * px + ar * py
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        px, py, x, y = next_px, next_py, next_x, next_y
    summary_offset = chunk_program * modes + lane
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
def _samu_chunk_replay(eta, delta, nu_log, theta_log, br, bi,
                       chunk_in_r, chunk_in_i, out_r, out_i,
                       length: tl.constexpr, modes: tl.constexpr,
                       chunks: tl.constexpr, CHUNK: tl.constexpr,
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
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        radius = tl.exp(-nu * tl.exp(radial))
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y
        tl.store(out_r + position, x, mask=mask)
        tl.store(out_i + position, y, mask=mask)


@triton.jit
def _samu_reverse_chunk_summary(eta, delta, nu_log, theta_log,
                                grad_out_r, grad_out_i, pr, pi, qr, qi,
                                length: tl.constexpr, modes: tl.constexpr,
                                chunks: tl.constexpr, CHUNK: tl.constexpr,
                                BLOCK: tl.constexpr):
    """Summarize the reverse adjoint map of each time chunk."""
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
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
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        radius = tl.exp(-nu * tl.exp(radial))
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        # The reverse transition is the complex conjugate of the forward one.
        ar, ai = radius * cosine, -radius * sine
        position = token_offset * modes + lane
        direct_r = tl.load(grad_out_r + position, mask=mask, other=0.0).to(tl.float32)
        direct_i = tl.load(grad_out_i + position, mask=mask, other=0.0).to(tl.float32)
        write_r = ar * direct_r - ai * direct_i
        write_i = ai * direct_r + ar * direct_i
        next_px = ar * px - ai * py
        next_py = ai * px + ar * py
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        px, py, x, y = next_px, next_py, next_x, next_y
    summary_offset = chunk_program * modes + lane
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
def _samu_reverse_chunk_replay(eta, delta, nu_log, theta_log,
                               out_r, out_i, grad_out_r, grad_out_i,
                               future_r, future_i, partial_eta, partial_delta,
                               grad_nu_log, grad_theta_log, grad_br, grad_bi,
                               length: tl.constexpr, modes: tl.constexpr,
                               chunks: tl.constexpr, mode_blocks: tl.constexpr,
                               CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    summary_offset = chunk_program * modes + lane
    adj_r = tl.load(future_r + summary_offset, mask=mask, other=0.0)
    adj_i = tl.load(future_i + summary_offset, mask=mask, other=0.0)
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_theta = tl.zeros((BLOCK,), tl.float32)
    for reverse_offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_offset)
        token_offset = batch * length + t
        position = token_offset * modes + lane
        adj_r += tl.load(grad_out_r + position, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + position, mask=mask, other=0.0).to(tl.float32)
        previous_position = (token_offset - 1) * modes + lane
        previous_r = tl.load(
            out_r + previous_position, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        previous_i = tl.load(
            out_i + previous_position, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        exp_radial = tl.exp(radial)
        log_radius = -nu * exp_radial
        radius = tl.exp(log_radius)
        cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        radial_gradient = log_radius * (grad_ar * ar + grad_ai * ai)
        phase_gradient = -grad_ar * ai + grad_ai * ar
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
        tl.store(grad_br + position, adj_r, mask=mask)
        tl.store(grad_bi + position, adj_i, mask=mask)
        previous_adj_r = ar * adj_r + ai * adj_i
        previous_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = previous_adj_r, previous_adj_i
    tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
    tl.atomic_add(grad_theta_log + lane, total_theta, mask=mask)


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
    gamma = tl.sqrt(tl.maximum(1.0 - tl.exp(-2.0 * nu), 1e-8))
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
    def forward(ctx, eta, delta, nu_log, theta_log, br, bi, chunk_size: int):
        if eta.shape != delta.shape or br.shape != bi.shape:
            raise ValueError("SAMU control or write shapes disagree")
        batch, length, modes = br.shape
        if chunk_size not in (16, 32, 64) or length % chunk_size:
            raise ValueError("chunk size must be 16, 32 or 64 and divide length")
        eta, delta, nu_log, theta_log, br, bi = (
            tensor.contiguous()
            for tensor in (eta, delta, nu_log, theta_log, br, bi)
        )
        chunks = length // chunk_size
        block, mode_blocks = _launch_shape(modes)
        shape = (batch, chunks, modes)
        pr, pi, qr, qi, chunk_in_r, chunk_in_i = (
            torch.empty(shape, device=br.device, dtype=torch.float32)
            for _ in range(6)
        )
        grid = (mode_blocks, batch * chunks)
        launch = dict(
            length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _samu_chunk_summary[grid](
            eta, delta, nu_log, theta_log, br, bi, pr, pi, qr, qi, **launch
        )
        _samu_chunk_prefix[(mode_blocks, batch)](
            pr, pi, qr, qi, chunk_in_r, chunk_in_i,
            modes=modes, chunks=chunks, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        _samu_chunk_replay[grid](
            eta, delta, nu_log, theta_log, br, bi,
            chunk_in_r, chunk_in_i, out_r, out_i, **launch
        )
        ctx.save_for_backward(eta, delta, nu_log, theta_log, out_r, out_i)
        ctx.chunk_size = chunk_size
        return out_r, out_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i):
        eta, delta, nu_log, theta_log, out_r, out_i = ctx.saved_tensors
        batch, length, modes = out_r.shape
        chunk_size = ctx.chunk_size
        chunks = length // chunk_size
        block, mode_blocks = _launch_shape(modes)
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        shape = (batch, chunks, modes)
        pr, pi, qr, qi, future_r, future_i = (
            torch.empty(shape, device=out_r.device, dtype=torch.float32)
            for _ in range(6)
        )
        grid = (mode_blocks, batch * chunks)
        launch = dict(
            length=length, modes=modes, chunks=chunks, CHUNK=chunk_size,
            BLOCK=block, num_warps=4 if block >= 64 else 2,
        )
        _samu_reverse_chunk_summary[grid](
            eta, delta, nu_log, theta_log, grad_out_r, grad_out_i,
            pr, pi, qr, qi, **launch
        )
        _samu_reverse_chunk_prefix[(mode_blocks, batch)](
            pr, pi, qr, qi, future_r, future_i,
            modes=modes, chunks=chunks, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        partial_shape = (batch * length, mode_blocks)
        partial_eta = torch.empty(partial_shape, device=out_r.device, dtype=torch.float32)
        partial_delta = torch.empty_like(partial_eta)
        grad_eta, grad_delta = torch.empty_like(eta), torch.empty_like(delta)
        grad_nu_log, grad_theta_log = torch.zeros_like(nu_log), torch.zeros_like(theta_log)
        grad_br, grad_bi = torch.empty_like(grad_out_r), torch.empty_like(grad_out_i)
        _samu_reverse_chunk_replay[grid](
            eta, delta, nu_log, theta_log, out_r, out_i,
            grad_out_r, grad_out_i, future_r, future_i,
            partial_eta, partial_delta, grad_nu_log, grad_theta_log,
            grad_br, grad_bi, mode_blocks=mode_blocks, **launch
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
):
    return _SAMUChunkScan.apply(
        eta, delta, nu_log, theta_log, br, bi, chunk_size
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
