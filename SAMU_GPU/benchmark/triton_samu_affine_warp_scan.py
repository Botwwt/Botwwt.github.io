"""SAMU-specific complex affine tile scan with on-the-fly transitions.

The kernels consume shared per-token ``E=exp(eta), C=cos(delta), S=sin(delta)``
and static per-mode spectral values.  They never write a logical
``lambda_[r/i][B,L,M]`` tensor.  Each time tile is scanned with Triton's
associative scan, chunk summaries are prefixed across tiles, and a second
local scan applies the chunk boundary.  The first backward implementation is
an exact reverse/conjugate tiled scan; a parallel reverse tile scan is kept as
a separate optimization gate rather than conflating it with forward results.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _complex_affine_combine(
    left_pr, left_pi, left_qr, left_qi,
    right_pr, right_pi, right_qr, right_qi,
):
    """Return ``right o left``; composition is associative, not commutative."""
    product_r = right_pr * left_pr - right_pi * left_pi
    product_i = right_pi * left_pr + right_pr * left_pi
    write_r = right_pr * left_qr - right_pi * left_qi + right_qr
    write_i = right_pi * left_qr + right_pr * left_qi + right_qi
    return product_r, product_i, write_r, write_i


@triton.jit
def _local_affine_summary(
    e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
    summary_pr, summary_pi, summary_qr, summary_qi,
    length: tl.constexpr, modes: tl.constexpr, chunks: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    STEPS: tl.constexpr, MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    step = tl.arange(0, STEPS)[:, None]
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)[None, :]
    token = chunk * STEPS + step
    valid_token = token < length
    valid_mode = lane < modes
    mask = valid_token & valid_mode
    token_offset = batch * length + token
    mode = lane
    shared_e = tl.load(e + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    shared_c = tl.load(c + token_offset, mask=valid_token, other=1.0).to(tl.float32)
    shared_s = tl.load(s + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    mode_nu = tl.load(nu + mode, mask=valid_mode, other=0.0).to(tl.float32)
    mode_c = tl.load(cos_theta + mode, mask=valid_mode, other=1.0).to(tl.float32)
    mode_s = tl.load(sin_theta + mode, mask=valid_mode, other=0.0).to(tl.float32)
    radius = tl.exp(-mode_nu * shared_e)
    transition_r = radius * (mode_c * shared_c - mode_s * shared_s)
    transition_i = radius * (mode_s * shared_c + mode_c * shared_s)
    if HAS_SEGMENTS:
        reset = valid_token & (
            tl.load(segment_pos + token_offset, mask=valid_token, other=1) == 0
        )
    elif RESET_FIRST:
        reset = valid_token & (token == 0)
    else:
        reset = False
    transition_r = tl.where(reset, 0.0, transition_r)
    transition_i = tl.where(reset, 0.0, transition_i)
    transition_r = tl.where(mask, transition_r, 1.0)
    transition_i = tl.where(mask, transition_i, 0.0)
    position = token_offset * modes + lane
    write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
    write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
    prefix_pr, prefix_pi, prefix_qr, prefix_qi = tl.associative_scan(
        (transition_r, transition_i, write_r, write_i),
        axis=0,
        combine_fn=_complex_affine_combine,
    )
    last = step == STEPS - 1
    result_pr = tl.sum(tl.where(last, prefix_pr, 0.0), axis=0)
    result_pi = tl.sum(tl.where(last, prefix_pi, 0.0), axis=0)
    result_qr = tl.sum(tl.where(last, prefix_qr, 0.0), axis=0)
    result_qi = tl.sum(tl.where(last, prefix_qi, 0.0), axis=0)
    summary_offset = chunk_program * modes + (
        mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    )
    summary_mask = summary_offset - chunk_program * modes < modes
    tl.store(summary_pr + summary_offset, result_pr, mask=summary_mask)
    tl.store(summary_pi + summary_offset, result_pi, mask=summary_mask)
    tl.store(summary_qr + summary_offset, result_qr, mask=summary_mask)
    tl.store(summary_qi + summary_offset, result_qi, mask=summary_mask)


@triton.jit
def _serial_outer_prefix(
    summary_pr, summary_pi, summary_qr, summary_qi,
    h0_r, h0_i, boundary_r, boundary_i, last_r, last_i,
    modes: tl.constexpr, chunks: tl.constexpr, HAS_H0: tl.constexpr,
    MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    mask = lane < modes
    if HAS_H0:
        state_r = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
        state_i = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        state_r = tl.zeros((MODE_BLOCK,), tl.float32)
        state_i = tl.zeros((MODE_BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(boundary_r + offset, state_r, mask=mask)
        tl.store(boundary_i + offset, state_i, mask=mask)
        pr = tl.load(summary_pr + offset, mask=mask, other=1.0)
        pi = tl.load(summary_pi + offset, mask=mask, other=0.0)
        qr = tl.load(summary_qr + offset, mask=mask, other=0.0)
        qi = tl.load(summary_qi + offset, mask=mask, other=0.0)
        next_r = pr * state_r - pi * state_i + qr
        next_i = pi * state_r + pr * state_i + qi
        state_r, state_i = next_r, next_i
    tl.store(last_r + batch * modes + lane, state_r, mask=mask)
    tl.store(last_i + batch * modes + lane, state_i, mask=mask)


@triton.jit
def _local_affine_replay(
    e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
    boundary_r, boundary_i, out_r, out_i,
    length: tl.constexpr, modes: tl.constexpr, chunks: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    STEPS: tl.constexpr, MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    step = tl.arange(0, STEPS)[:, None]
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)[None, :]
    token = chunk * STEPS + step
    valid_token = token < length
    valid_mode = lane < modes
    mask = valid_token & valid_mode
    token_offset = batch * length + token
    shared_e = tl.load(e + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    shared_c = tl.load(c + token_offset, mask=valid_token, other=1.0).to(tl.float32)
    shared_s = tl.load(s + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    mode_nu = tl.load(nu + lane, mask=valid_mode, other=0.0).to(tl.float32)
    mode_c = tl.load(cos_theta + lane, mask=valid_mode, other=1.0).to(tl.float32)
    mode_s = tl.load(sin_theta + lane, mask=valid_mode, other=0.0).to(tl.float32)
    radius = tl.exp(-mode_nu * shared_e)
    transition_r = radius * (mode_c * shared_c - mode_s * shared_s)
    transition_i = radius * (mode_s * shared_c + mode_c * shared_s)
    if HAS_SEGMENTS:
        reset = valid_token & (
            tl.load(segment_pos + token_offset, mask=valid_token, other=1) == 0
        )
    elif RESET_FIRST:
        reset = valid_token & (token == 0)
    else:
        reset = False
    transition_r = tl.where(reset, 0.0, transition_r)
    transition_i = tl.where(reset, 0.0, transition_i)
    transition_r = tl.where(mask, transition_r, 1.0)
    transition_i = tl.where(mask, transition_i, 0.0)
    position = token_offset * modes + lane
    write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
    write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
    prefix_pr, prefix_pi, prefix_qr, prefix_qi = tl.associative_scan(
        (transition_r, transition_i, write_r, write_i),
        axis=0,
        combine_fn=_complex_affine_combine,
    )
    summary_lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    summary_mask = summary_lane < modes
    summary_offset = chunk_program * modes + summary_lane
    initial_r = tl.load(boundary_r + summary_offset, mask=summary_mask, other=0.0)
    initial_i = tl.load(boundary_i + summary_offset, mask=summary_mask, other=0.0)
    state_r = prefix_pr * initial_r[None, :] - prefix_pi * initial_i[None, :] + prefix_qr
    state_i = prefix_pi * initial_r[None, :] + prefix_pr * initial_i[None, :] + prefix_qi
    tl.store(out_r + position, state_r, mask=mask)
    tl.store(out_i + position, state_i, mask=mask)


@triton.jit
def _reduce_three_shared(
    partial_e, partial_c, partial_s, grad_e, grad_c, grad_s,
    rows: tl.constexpr, mode_blocks: tl.constexpr, REDUCE: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, REDUCE)
    mask = lane < mode_blocks
    offset = row * mode_blocks + lane
    tl.store(grad_e + row, tl.sum(tl.load(partial_e + offset, mask=mask, other=0.0), axis=0))
    tl.store(grad_c + row, tl.sum(tl.load(partial_c + offset, mask=mask, other=0.0), axis=0))
    tl.store(grad_s + row, tl.sum(tl.load(partial_s + offset, mask=mask, other=0.0), axis=0))


@triton.jit
def _reverse_conjugate_serial_backward(
    e, c, s, nu, cos_theta, sin_theta, segment_pos, h0_r, h0_i,
    out_r, out_i, grad_out_r, grad_out_i, grad_last_r, grad_last_i,
    partial_e, partial_c, partial_s, grad_nu, grad_cos_theta, grad_sin_theta,
    grad_br, grad_bi, grad_h0_r, grad_h0_i,
    length: tl.constexpr, modes: tl.constexpr, mode_blocks: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    HAS_H0: tl.constexpr, MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    mask = lane < modes
    mode_nu = tl.load(nu + lane, mask=mask, other=0.0).to(tl.float32)
    mode_c = tl.load(cos_theta + lane, mask=mask, other=1.0).to(tl.float32)
    mode_s = tl.load(sin_theta + lane, mask=mask, other=0.0).to(tl.float32)
    adj_r = tl.load(grad_last_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    adj_i = tl.load(grad_last_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    total_nu = tl.zeros((MODE_BLOCK,), tl.float32)
    total_ct = tl.zeros((MODE_BLOCK,), tl.float32)
    total_st = tl.zeros((MODE_BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        token_offset = batch * length + t
        position = token_offset * modes + lane
        adj_r += tl.load(grad_out_r + position, mask=mask, other=0.0).to(tl.float32)
        adj_i += tl.load(grad_out_i + position, mask=mask, other=0.0).to(tl.float32)
        previous_position = (token_offset - 1) * modes + lane
        previous_r = tl.load(out_r + previous_position, mask=mask & (t > 0), other=0.0).to(tl.float32)
        previous_i = tl.load(out_i + previous_position, mask=mask & (t > 0), other=0.0).to(tl.float32)
        if HAS_H0:
            initial_r = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
            initial_i = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
            previous_r = tl.where(t == 0, initial_r, previous_r)
            previous_i = tl.where(t == 0, initial_i, previous_i)
        shared_e = tl.load(e + token_offset).to(tl.float32)
        shared_c = tl.load(c + token_offset).to(tl.float32)
        shared_s = tl.load(s + token_offset).to(tl.float32)
        radius = tl.exp(-mode_nu * shared_e)
        cosine = mode_c * shared_c - mode_s * shared_s
        sine = mode_s * shared_c + mode_c * shared_s
        ar, ai = radius * cosine, radius * sine
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + token_offset) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        grad_ar = adj_r * previous_r + adj_i * previous_i
        grad_ai = -adj_r * previous_i + adj_i * previous_r
        grad_radius = grad_ar * cosine + grad_ai * sine
        grad_cosine = grad_ar * radius
        grad_sine = grad_ai * radius
        grad_e_value = -mode_nu * radius * grad_radius
        grad_c_value = mode_c * grad_cosine + mode_s * grad_sine
        grad_s_value = -mode_s * grad_cosine + mode_c * grad_sine
        grad_nu_value = -shared_e * radius * grad_radius
        grad_ct_value = shared_c * grad_cosine + shared_s * grad_sine
        grad_st_value = -shared_s * grad_cosine + shared_c * grad_sine
        grad_e_value = tl.where(reset, 0.0, grad_e_value)
        grad_c_value = tl.where(reset, 0.0, grad_c_value)
        grad_s_value = tl.where(reset, 0.0, grad_s_value)
        grad_nu_value = tl.where(reset, 0.0, grad_nu_value)
        grad_ct_value = tl.where(reset, 0.0, grad_ct_value)
        grad_st_value = tl.where(reset, 0.0, grad_st_value)
        partial_offset = token_offset * mode_blocks + mode_block
        tl.store(partial_e + partial_offset, tl.sum(tl.where(mask, grad_e_value, 0.0), axis=0))
        tl.store(partial_c + partial_offset, tl.sum(tl.where(mask, grad_c_value, 0.0), axis=0))
        tl.store(partial_s + partial_offset, tl.sum(tl.where(mask, grad_s_value, 0.0), axis=0))
        total_nu += grad_nu_value
        total_ct += grad_ct_value
        total_st += grad_st_value
        tl.store(grad_br + position, adj_r, mask=mask)
        tl.store(grad_bi + position, adj_i, mask=mask)
        ar = tl.where(reset, 0.0, ar)
        ai = tl.where(reset, 0.0, ai)
        next_adj_r = ar * adj_r + ai * adj_i
        next_adj_i = -ai * adj_r + ar * adj_i
        adj_r, adj_i = next_adj_r, next_adj_i
    tl.atomic_add(grad_nu + lane, total_nu, mask=mask)
    tl.atomic_add(grad_cos_theta + lane, total_ct, mask=mask)
    tl.atomic_add(grad_sin_theta + lane, total_st, mask=mask)
    tl.store(grad_h0_r + batch * modes + lane, adj_r, mask=mask)
    tl.store(grad_h0_i + batch * modes + lane, adj_i, mask=mask)


class _SAMUAffineTileScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, c, s, nu, cos_theta, sin_theta, br, bi,
                segment_pos, h0_r, h0_i, steps: int, mode_block: int,
                num_warps: int, reset_first: bool):
        batch, length, modes = br.shape
        if e.shape != (batch, length) or c.shape != e.shape or s.shape != e.shape:
            raise ValueError("E/C/S must have shape [B,L]")
        if bi.shape != br.shape or any(t.shape != (modes,) for t in (nu, cos_theta, sin_theta)):
            raise ValueError("SAMU write or spectrum shape mismatch")
        if steps not in (8, 16, 32) or mode_block not in (16, 32, 64):
            raise ValueError("unsupported tile configuration")
        if num_warps not in (4, 8, 16, 32):
            raise ValueError("num_warps must be 4, 8, 16 or 32")
        has_segments = segment_pos.numel() != 0
        has_h0 = h0_r.numel() != 0
        chunks = triton.cdiv(length, steps)
        mode_blocks = triton.cdiv(modes, mode_block)
        shape = (batch, chunks, modes)
        summaries = tuple(torch.empty(shape, device=br.device, dtype=torch.float32)
                          for _ in range(4))
        boundary_r = torch.empty(shape, device=br.device, dtype=torch.float32)
        boundary_i = torch.empty_like(boundary_r)
        out_r, out_i = torch.empty_like(br), torch.empty_like(bi)
        last_r = torch.empty((batch, modes), device=br.device, dtype=torch.float32)
        last_i = torch.empty_like(last_r)
        grid = (mode_blocks, batch * chunks)
        warps = num_warps
        _local_affine_summary[grid](
            e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
            *summaries, length=length, modes=modes, chunks=chunks,
            HAS_SEGMENTS=has_segments, RESET_FIRST=bool(reset_first),
            STEPS=steps, MODE_BLOCK=mode_block, num_warps=warps,
        )
        _serial_outer_prefix[(mode_blocks, batch)](
            *summaries, h0_r, h0_i, boundary_r, boundary_i, last_r, last_i,
            modes=modes, chunks=chunks, HAS_H0=has_h0,
            MODE_BLOCK=mode_block, num_warps=warps,
        )
        _local_affine_replay[grid](
            e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
            boundary_r, boundary_i, out_r, out_i,
            length=length, modes=modes, chunks=chunks,
            HAS_SEGMENTS=has_segments, RESET_FIRST=bool(reset_first),
            STEPS=steps, MODE_BLOCK=mode_block, num_warps=warps,
        )
        ctx.save_for_backward(e, c, s, nu, cos_theta, sin_theta,
                              segment_pos, h0_r, h0_i, out_r, out_i)
        ctx.mode_block = mode_block
        ctx.num_warps = num_warps
        ctx.has_segments = has_segments
        ctx.has_h0 = has_h0
        ctx.reset_first = bool(reset_first)
        return out_r, out_i, last_r, last_i

    @staticmethod
    def backward(ctx, grad_out_r, grad_out_i, grad_last_r, grad_last_i):
        (e, c, s, nu, cos_theta, sin_theta, segment_pos,
         h0_r, h0_i, out_r, out_i) = ctx.saved_tensors
        batch, length, modes = out_r.shape
        # Forward associative tiles prefer narrow mode blocks to limit the
        # four-component prefix register footprint.  The serial reverse scan
        # has no such time-tile tensor and uses a wide mode tile to avoid an
        # artificial explosion in programs, atomics and partial reductions.
        mode_block = min(128, triton.next_power_of_2(modes))
        mode_blocks = triton.cdiv(modes, mode_block)
        if grad_last_r is None:
            grad_last_r = torch.zeros((batch, modes), device=out_r.device, dtype=torch.float32)
        if grad_last_i is None:
            grad_last_i = torch.zeros((batch, modes), device=out_r.device, dtype=torch.float32)
        grad_out_r, grad_out_i = grad_out_r.contiguous(), grad_out_i.contiguous()
        partial_shape = (batch * length, mode_blocks)
        partial_e = torch.empty(partial_shape, device=out_r.device, dtype=torch.float32)
        partial_c = torch.empty_like(partial_e)
        partial_s = torch.empty_like(partial_e)
        grad_e, grad_c, grad_s = torch.empty_like(e), torch.empty_like(c), torch.empty_like(s)
        grad_nu = torch.zeros_like(nu)
        grad_ct = torch.zeros_like(cos_theta)
        grad_st = torch.zeros_like(sin_theta)
        grad_br, grad_bi = torch.empty_like(out_r), torch.empty_like(out_i)
        grad_h0_r = torch.empty((batch, modes), device=out_r.device, dtype=torch.float32)
        grad_h0_i = torch.empty_like(grad_h0_r)
        warps = 2 if mode_block == 128 else 4
        _reverse_conjugate_serial_backward[(mode_blocks, batch)](
            e, c, s, nu, cos_theta, sin_theta, segment_pos, h0_r, h0_i,
            out_r, out_i, grad_out_r, grad_out_i, grad_last_r, grad_last_i,
            partial_e, partial_c, partial_s, grad_nu, grad_ct, grad_st,
            grad_br, grad_bi, grad_h0_r, grad_h0_i,
            length=length, modes=modes, mode_blocks=mode_blocks,
            HAS_SEGMENTS=ctx.has_segments, RESET_FIRST=ctx.reset_first,
            HAS_H0=ctx.has_h0, MODE_BLOCK=mode_block, num_warps=warps,
        )
        reduce = triton.next_power_of_2(mode_blocks)
        _reduce_three_shared[(batch * length,)](
            partial_e, partial_c, partial_s, grad_e, grad_c, grad_s,
            rows=batch * length, mode_blocks=mode_blocks, REDUCE=reduce,
            num_warps=1,
        )
        return (grad_e, grad_c, grad_s, grad_nu, grad_ct, grad_st,
                grad_br, grad_bi, None,
                grad_h0_r if ctx.has_h0 else None,
                grad_h0_i if ctx.has_h0 else None,
                None, None, None, None)


def samu_affine_tile_scan(
    eta: torch.Tensor, delta: torch.Tensor,
    nu_log: torch.Tensor, theta_log: torch.Tensor,
    br: torch.Tensor, bi: torch.Tensor, *,
    segment_pos: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    steps: int = 16, mode_block: int = 32,
    num_warps: int = 4,
    reset_first: bool = False,
):
    """Run the on-the-fly complex affine tile scan with full autograd."""
    e = torch.exp(eta)
    c, s = torch.cos(delta), torch.sin(delta)
    nu = torch.exp(nu_log)
    theta = torch.exp(theta_log)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    empty_segment = eta.new_empty((0,), dtype=torch.int32)
    empty_state = br.new_empty((0,), dtype=torch.float32)
    if initial_state is None:
        h0_r = h0_i = empty_state
    else:
        h0_r, h0_i = initial_state
    return _SAMUAffineTileScan.apply(
        e, c, s, nu, cos_theta, sin_theta, br, bi,
        empty_segment if segment_pos is None else segment_pos,
        h0_r, h0_i, int(steps), int(mode_block), int(num_warps),
        bool(reset_first),
    )
