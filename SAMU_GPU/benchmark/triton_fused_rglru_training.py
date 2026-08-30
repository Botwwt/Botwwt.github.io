"""Fused differentiable RG-LRU preparation and exact scans for NVIDIA GPUs.

This module follows the public RecurrentGemma RG-LRU equations while moving
the pointwise gate activation, transition construction, normalized write and
scan into the same Triton programs.  It provides both a state-stationary serial
path and a three-stage exact chunk path.  The latter adds temporal parallelism
without materializing per-token ``a`` or ``write`` tensors.

The serial fusion and activation-recomputation strategy is prior art from
fattorib/hawk-pytorch.  Reset-aware exact chunk summaries, partial final chunks,
initial-state gradients and the fused chunk backward are implemented here for
the H800 comparison.  "No materialized transition" below means no logical
global ``[B,L,D]`` transition allocation; it is not a claim of zero HBM traffic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    # RG-LRU's initialized a_param is in a benign range.  Keep the expression
    # aligned with the public fused kernel instead of introducing an approximation.
    return tl.log(1.0 + tl.exp(x))


@triton.jit
def _step_terms(x, gate_x_logit, gate_a_logit, a_param, reset,
                ROUND_BF16: tl.constexpr):
    gate_x = tl.sigmoid(gate_x_logit)
    gate_a = tl.sigmoid(gate_a_logit)
    log_a = -8.0 * gate_a * _softplus(a_param)
    a_square = tl.exp(2.0 * log_a)
    transition = tl.exp(log_a)
    multiplier = tl.sqrt(tl.maximum(1.0 - a_square, 0.0))
    transition = tl.where(reset, 0.0, transition)
    multiplier = tl.where(reset, 1.0, multiplier)
    write = x * gate_x * multiplier
    if ROUND_BF16:
        # Match the production eager-BF16 activation boundary before the FP32 scan.
        transition = transition.to(tl.bfloat16).to(tl.float32)
        write = write.to(tl.bfloat16).to(tl.float32)
    return transition, write


@triton.jit
def _serial_forward(x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0,
                    out, last, length: tl.constexpr, width: tl.constexpr,
                    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
                    HAS_H0: tl.constexpr, ROUND_BF16: tl.constexpr,
                    BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    param = tl.load(a_param + lane, mask=mask, other=0.0).to(tl.float32)
    if HAS_H0:
        state = tl.load(h0 + batch * width + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        offset = (batch * length + t) * width + lane
        token = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        gx = tl.load(gate_x_logit + offset, mask=mask, other=0.0).to(tl.float32)
        ga = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + batch * length + t) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition, write = _step_terms(token, gx, ga, param, reset, ROUND_BF16)
        state = transition * state + write
        tl.store(out + offset, state, mask=mask)
    tl.store(last + batch * width + lane, state, mask=mask)


@triton.jit
def _serial_backward(x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                     h0, out, grad_out, grad_last, grad_x, grad_gate_x,
                     grad_gate_a, partial_a_param, grad_h0,
                     length: tl.constexpr, width: tl.constexpr,
                     HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
                     HAS_H0: tl.constexpr, ROUND_BF16: tl.constexpr,
                     BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    param = tl.load(a_param + lane, mask=mask, other=0.0).to(tl.float32)
    softplus_param = _softplus(param)
    sigmoid_param = tl.sigmoid(param)
    adjoint = tl.load(grad_last + batch * width + lane, mask=mask, other=0.0).to(tl.float32)
    param_gradient = tl.zeros((BLOCK,), tl.float32)
    for reverse_t in tl.range(0, length, 1, num_stages=1):
        t = length - 1 - reverse_t
        offset = (batch * length + t) * width + lane
        adjoint += tl.load(grad_out + offset, mask=mask, other=0.0).to(tl.float32)
        token = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        gx_logit = tl.load(gate_x_logit + offset, mask=mask, other=0.0).to(tl.float32)
        ga_logit = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        gate_x = tl.sigmoid(gx_logit)
        gate_a = tl.sigmoid(ga_logit)
        log_a = -8.0 * gate_a * softplus_param
        a_square = tl.exp(2.0 * log_a)
        transition_unrounded = tl.exp(log_a)
        multiplier = tl.sqrt(tl.maximum(1.0 - a_square, 0.0))
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + batch * length + t) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition = tl.where(reset, 0.0, transition_unrounded)
        multiplier = tl.where(reset, 1.0, multiplier)
        if ROUND_BF16:
            transition = transition.to(tl.bfloat16).to(tl.float32)
        previous_offset = (batch * length + t - 1) * width + lane
        previous = tl.load(
            out + previous_offset, mask=mask & (t > 0), other=0.0
        ).to(tl.float32)
        if HAS_H0:
            initial = tl.load(h0 + batch * width + lane, mask=mask, other=0.0).to(tl.float32)
            previous = tl.where(t == 0, initial, previous)
        grad_write = adjoint
        token_gradient = grad_write * gate_x * multiplier
        gate_x_gradient = (
            grad_write * token * multiplier * gate_x * (1.0 - gate_x)
        )
        grad_transition = adjoint * previous
        sqrt_gradient = 1.0 / tl.sqrt(tl.maximum(4.0 * (1.0 - a_square), 1.0e-6))
        grad_log_a = grad_transition * transition_unrounded
        grad_log_a += (
            grad_write * token * gate_x * sqrt_gradient * (-2.0 * a_square)
        )
        grad_log_a = tl.where(reset, 0.0, grad_log_a)
        gate_a_gradient = (
            grad_log_a * (-8.0 * softplus_param) * gate_a * (1.0 - gate_a)
        )
        param_gradient += grad_log_a * (-8.0 * gate_a) * sigmoid_param
        tl.store(grad_x + offset, token_gradient, mask=mask)
        tl.store(grad_gate_x + offset, gate_x_gradient, mask=mask)
        tl.store(grad_gate_a + offset, gate_a_gradient, mask=mask)
        adjoint *= transition
    tl.store(partial_a_param + batch * width + lane, param_gradient, mask=mask)
    tl.store(grad_h0 + batch * width + lane, adjoint, mask=mask)


@triton.jit
def _chunk_summary(x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                   summary_a, summary_b, length, width: tl.constexpr,
                   chunks: tl.constexpr, HAS_SEGMENTS: tl.constexpr,
                   RESET_FIRST: tl.constexpr, ROUND_BF16: tl.constexpr,
                   CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    param = tl.load(a_param + lane, mask=lane_mask, other=0.0).to(tl.float32)
    product = tl.full((BLOCK,), 1.0, tl.float32)
    state = tl.zeros((BLOCK,), tl.float32)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        valid = t < length
        offset = (batch * length + t) * width + lane
        mask = lane_mask & valid
        token = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        gx = tl.load(gate_x_logit + offset, mask=mask, other=0.0).to(tl.float32)
        ga = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_SEGMENTS:
            reset = valid & (tl.load(segment_pos + batch * length + t, mask=valid, other=1) == 0)
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition, write = _step_terms(token, gx, ga, param, reset, ROUND_BF16)
        transition = tl.where(valid, transition, 1.0)
        write = tl.where(valid, write, 0.0)
        product = transition * product
        state = transition * state + write
    summary = chunk_program * width + lane
    tl.store(summary_a + summary, product, mask=lane_mask)
    tl.store(summary_b + summary, state, mask=lane_mask)


@triton.jit
def _chunk_prefix(summary_a, summary_b, h0, chunk_input, last,
                  width: tl.constexpr, chunks: tl.constexpr,
                  HAS_H0: tl.constexpr, BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    if HAS_H0:
        state = tl.load(h0 + batch * width + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * width + lane
        tl.store(chunk_input + offset, state, mask=mask)
        product = tl.load(summary_a + offset, mask=mask, other=1.0)
        write = tl.load(summary_b + offset, mask=mask, other=0.0)
        state = product * state + write
    tl.store(last + batch * width + lane, state, mask=mask)


@triton.jit
def _affine_prefix_stage(input_a, input_b, output_a, output_b,
                         width: tl.constexpr, chunks: tl.constexpr,
                         STRIDE: tl.constexpr, REVERSE: tl.constexpr,
                         BLOCK: tl.constexpr):
    """One Hillis--Steele stage over exact affine chunk transforms.

    The pair ``(a, b)`` represents ``state -> a * state + b``. Forward
    traversal reads the transform ``STRIDE`` chunks to the left; reverse
    traversal reads it to the right. Ping-pong buffers make every stage
    race-free. This trades O(log C) launches and extra summary traffic for
    temporal parallelism across C chunks.
    """
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    offset = (batch * chunks + chunk) * width + lane
    current_a = tl.load(input_a + offset, mask=lane_mask, other=1.0)
    current_b = tl.load(input_b + offset, mask=lane_mask, other=0.0)
    if REVERSE:
        has_previous = chunk + STRIDE < chunks
        previous_chunk = chunk + STRIDE
    else:
        has_previous = chunk >= STRIDE
        previous_chunk = chunk - STRIDE
    previous_offset = (batch * chunks + previous_chunk) * width + lane
    previous_a = tl.load(
        input_a + previous_offset,
        mask=lane_mask & has_previous,
        other=1.0,
    )
    previous_b = tl.load(
        input_b + previous_offset,
        mask=lane_mask & has_previous,
        other=0.0,
    )
    # current o previous: current_a * (previous_a * x + previous_b)
    combined_a = current_a * previous_a
    combined_b = current_a * previous_b + current_b
    tl.store(output_a + offset, combined_a, mask=lane_mask)
    tl.store(output_b + offset, combined_b, mask=lane_mask)


@triton.jit
def _affine_prefix_extract(prefix_a, prefix_b, initial, boundary, final,
                           width: tl.constexpr, chunks: tl.constexpr,
                           REVERSE: tl.constexpr, HAS_INITIAL: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Convert an inclusive affine prefix into chunk boundary states."""
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    if HAS_INITIAL:
        initial_state = tl.load(
            initial + batch * width + lane, mask=lane_mask, other=0.0
        ).to(tl.float32)
    else:
        initial_state = tl.zeros((BLOCK,), tl.float32)
    if REVERSE:
        first = chunk == chunks - 1
        previous_chunk = chunk + 1
    else:
        first = chunk == 0
        previous_chunk = chunk - 1
    previous_offset = (batch * chunks + previous_chunk) * width + lane
    previous_a = tl.load(
        prefix_a + previous_offset, mask=lane_mask & ~first, other=1.0
    )
    previous_b = tl.load(
        prefix_b + previous_offset, mask=lane_mask & ~first, other=0.0
    )
    state = tl.where(
        first, initial_state, previous_a * initial_state + previous_b
    )
    offset = (batch * chunks + chunk) * width + lane
    tl.store(boundary + offset, state, mask=lane_mask)
    terminal_chunk = 0 if REVERSE else chunks - 1
    terminal = chunk == terminal_chunk
    terminal_a = tl.load(prefix_a + offset, mask=lane_mask & terminal, other=1.0)
    terminal_b = tl.load(prefix_b + offset, mask=lane_mask & terminal, other=0.0)
    terminal_state = terminal_a * initial_state + terminal_b
    tl.store(
        final + batch * width + lane,
        terminal_state,
        mask=lane_mask & terminal,
    )


@triton.jit
def _affine_group_local(
    summary_a, summary_b, group_a, group_b,
    width: tl.constexpr, chunks: tl.constexpr, groups: tl.constexpr,
    GROUP: tl.constexpr, REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):
    """Overwrite chunk summaries with local exclusive affine prefixes."""
    width_block = tl.program_id(0)
    group_program = tl.program_id(1)
    batch = group_program // groups
    group = group_program - batch * groups
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    accumulated_a = tl.full((BLOCK,), 1.0, tl.float32)
    accumulated_b = tl.zeros((BLOCK,), tl.float32)
    for logical_chunk in tl.range(0, GROUP, 1, num_stages=1):
        if REVERSE:
            local_chunk = GROUP - 1 - logical_chunk
        else:
            local_chunk = logical_chunk
        chunk = group * GROUP + local_chunk
        valid = chunk < chunks
        offset = (batch * chunks + chunk) * width + lane
        current_a = tl.load(
            summary_a + offset, mask=lane_mask & valid, other=1.0
        )
        current_b = tl.load(
            summary_b + offset, mask=lane_mask & valid, other=0.0
        )
        tl.store(summary_a + offset, accumulated_a, mask=lane_mask & valid)
        tl.store(summary_b + offset, accumulated_b, mask=lane_mask & valid)
        next_a = current_a * accumulated_a
        next_b = current_a * accumulated_b + current_b
        accumulated_a = tl.where(valid, next_a, accumulated_a)
        accumulated_b = tl.where(valid, next_b, accumulated_b)
    group_offset = group_program * width + lane
    tl.store(group_a + group_offset, accumulated_a, mask=lane_mask)
    tl.store(group_b + group_offset, accumulated_b, mask=lane_mask)


@triton.jit
def _affine_group_outer(
    group_a, group_b, initial, group_boundary, final,
    width: tl.constexpr, groups: tl.constexpr,
    HAS_INITIAL: tl.constexpr, REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Serial prefix over the small sequence of affine group summaries."""
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    if HAS_INITIAL:
        state = tl.load(
            initial + batch * width + lane, mask=mask, other=0.0
        ).to(tl.float32)
    else:
        state = tl.zeros((BLOCK,), tl.float32)
    for logical_group in tl.range(0, groups, 1, num_stages=1):
        if REVERSE:
            group = groups - 1 - logical_group
        else:
            group = logical_group
        offset = (batch * groups + group) * width + lane
        tl.store(group_boundary + offset, state, mask=mask)
        transform_a = tl.load(group_a + offset, mask=mask, other=1.0)
        transform_b = tl.load(group_b + offset, mask=mask, other=0.0)
        state = transform_a * state + transform_b
    tl.store(final + batch * width + lane, state, mask=mask)


@triton.jit
def _affine_group_correct(
    local_a, local_b, group_boundary, boundary,
    width: tl.constexpr, chunks: tl.constexpr,
    groups: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Apply each local exclusive prefix to its outer group boundary."""
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    group = chunk // GROUP
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    local_offset = chunk_program * width + lane
    group_offset = (batch * groups + group) * width + lane
    transform_a = tl.load(local_a + local_offset, mask=mask, other=1.0)
    transform_b = tl.load(local_b + local_offset, mask=mask, other=0.0)
    outer = tl.load(group_boundary + group_offset, mask=mask, other=0.0)
    tl.store(boundary + local_offset, transform_a * outer + transform_b,
             mask=mask)


@triton.jit
def _chunk_replay(x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                  chunk_input, out, length, width: tl.constexpr,
                  chunks: tl.constexpr, HAS_SEGMENTS: tl.constexpr,
                  RESET_FIRST: tl.constexpr, ROUND_BF16: tl.constexpr,
                  CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    state = tl.load(chunk_input + chunk_program * width + lane, mask=lane_mask, other=0.0)
    param = tl.load(a_param + lane, mask=lane_mask, other=0.0).to(tl.float32)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        valid = t < length
        offset = (batch * length + t) * width + lane
        mask = lane_mask & valid
        token = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        gx = tl.load(gate_x_logit + offset, mask=mask, other=0.0).to(tl.float32)
        ga = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_SEGMENTS:
            reset = valid & (tl.load(segment_pos + batch * length + t, mask=valid, other=1) == 0)
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition, write = _step_terms(token, gx, ga, param, reset, ROUND_BF16)
        state = tl.where(valid, transition * state + write, state)
        tl.store(out + offset, state, mask=mask)


@triton.jit
def _reverse_chunk_summary(gate_a_logit, a_param, segment_pos, grad_out,
                           summary_a, summary_b, length, width: tl.constexpr,
                           chunks: tl.constexpr, HAS_SEGMENTS: tl.constexpr,
                           RESET_FIRST: tl.constexpr, ROUND_BF16: tl.constexpr,
                           CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    param = tl.load(a_param + lane, mask=lane_mask, other=0.0).to(tl.float32)
    product = tl.full((BLOCK,), 1.0, tl.float32)
    adjoint = tl.zeros((BLOCK,), tl.float32)
    for reverse_local in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_local)
        valid = t < length
        offset = (batch * length + t) * width + lane
        mask = lane_mask & valid
        ga_logit = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        gate_a = tl.sigmoid(ga_logit)
        transition = tl.exp(-8.0 * gate_a * _softplus(param))
        if HAS_SEGMENTS:
            reset = valid & (tl.load(segment_pos + batch * length + t, mask=valid, other=1) == 0)
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition = tl.where(reset, 0.0, transition)
        if ROUND_BF16:
            transition = transition.to(tl.bfloat16).to(tl.float32)
        transition = tl.where(valid, transition, 1.0)
        direct = tl.load(grad_out + offset, mask=mask, other=0.0).to(tl.float32)
        product = transition * product
        adjoint = transition * adjoint + transition * direct
    summary = chunk_program * width + lane
    tl.store(summary_a + summary, product, mask=lane_mask)
    tl.store(summary_b + summary, adjoint, mask=lane_mask)


@triton.jit
def _reverse_chunk_prefix(summary_a, summary_b, grad_last, future, grad_h0,
                          width: tl.constexpr, chunks: tl.constexpr,
                          BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    adjoint = tl.load(grad_last + batch * width + lane, mask=mask, other=0.0).to(tl.float32)
    for reverse_chunk in tl.range(0, chunks, 1, num_stages=1):
        chunk = chunks - 1 - reverse_chunk
        offset = (batch * chunks + chunk) * width + lane
        tl.store(future + offset, adjoint, mask=mask)
        product = tl.load(summary_a + offset, mask=mask, other=1.0)
        write = tl.load(summary_b + offset, mask=mask, other=0.0)
        adjoint = product * adjoint + write
    tl.store(grad_h0 + batch * width + lane, adjoint, mask=mask)


@triton.jit
def _reverse_chunk_replay(x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                          h0, out, grad_out, future, grad_x, grad_gate_x,
                          grad_gate_a, partial_a_param, length,
                          width: tl.constexpr, chunks: tl.constexpr,
                          HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
                          HAS_H0: tl.constexpr, ROUND_BF16: tl.constexpr,
                          CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    param = tl.load(a_param + lane, mask=lane_mask, other=0.0).to(tl.float32)
    softplus_param = _softplus(param)
    sigmoid_param = tl.sigmoid(param)
    adjoint = tl.load(future + chunk_program * width + lane, mask=lane_mask, other=0.0)
    param_gradient = tl.zeros((BLOCK,), tl.float32)
    for reverse_local in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + (CHUNK - 1 - reverse_local)
        valid = t < length
        offset = (batch * length + t) * width + lane
        mask = lane_mask & valid
        adjoint += tl.load(grad_out + offset, mask=mask, other=0.0).to(tl.float32)
        token = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        gx_logit = tl.load(gate_x_logit + offset, mask=mask, other=0.0).to(tl.float32)
        ga_logit = tl.load(gate_a_logit + offset, mask=mask, other=0.0).to(tl.float32)
        gate_x = tl.sigmoid(gx_logit)
        gate_a = tl.sigmoid(ga_logit)
        log_a = -8.0 * gate_a * softplus_param
        a_square = tl.exp(2.0 * log_a)
        transition_unrounded = tl.exp(log_a)
        multiplier = tl.sqrt(tl.maximum(1.0 - a_square, 0.0))
        if HAS_SEGMENTS:
            reset = valid & (tl.load(segment_pos + batch * length + t, mask=valid, other=1) == 0)
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        transition = tl.where(reset, 0.0, transition_unrounded)
        multiplier = tl.where(reset, 1.0, multiplier)
        if ROUND_BF16:
            transition = transition.to(tl.bfloat16).to(tl.float32)
        previous = tl.load(
            out + (batch * length + t - 1) * width + lane,
            mask=mask & (t > 0), other=0.0,
        ).to(tl.float32)
        if HAS_H0:
            initial = tl.load(h0 + batch * width + lane, mask=lane_mask, other=0.0).to(tl.float32)
            previous = tl.where(t == 0, initial, previous)
        grad_write = adjoint
        token_gradient = grad_write * gate_x * multiplier
        gate_x_gradient = grad_write * token * multiplier * gate_x * (1.0 - gate_x)
        grad_transition = adjoint * previous
        sqrt_gradient = 1.0 / tl.sqrt(tl.maximum(4.0 * (1.0 - a_square), 1.0e-6))
        grad_log_a = grad_transition * transition_unrounded
        grad_log_a += grad_write * token * gate_x * sqrt_gradient * (-2.0 * a_square)
        grad_log_a = tl.where(reset, 0.0, grad_log_a)
        gate_a_gradient = grad_log_a * (-8.0 * softplus_param) * gate_a * (1.0 - gate_a)
        param_gradient += tl.where(valid, grad_log_a * (-8.0 * gate_a) * sigmoid_param, 0.0)
        tl.store(grad_x + offset, token_gradient, mask=mask)
        tl.store(grad_gate_x + offset, gate_x_gradient, mask=mask)
        tl.store(grad_gate_a + offset, gate_a_gradient, mask=mask)
        adjoint = tl.where(valid, adjoint * transition, adjoint)
    partial_offset = chunk_program * width + lane
    tl.store(partial_a_param + partial_offset, param_gradient, mask=lane_mask)


def _launch_shape(width: int, block_size: int) -> tuple[int, int]:
    if block_size not in (32, 64, 128, 256):
        raise ValueError("block_size must be 32, 64, 128 or 256")
    return block_size, triton.cdiv(width, block_size)


def _warps(block: int) -> int:
    return 1 if block <= 64 else (2 if block == 128 else 4)


class _FusedRGLRU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0,
                chunk_size: int, block_size: int, reset_first: bool,
                hierarchical_prefix: bool, prefix_group_size: int):
        if x.shape != gate_x_logit.shape or x.shape != gate_a_logit.shape or x.ndim != 3:
            raise ValueError("x and RG-LRU gate logits must be matching [B,L,D] tensors")
        batch, length, width = x.shape
        if a_param.shape != (width,):
            raise ValueError("a_param must have shape [D]")
        has_segments = segment_pos.numel() != 0
        has_h0 = h0.numel() != 0
        if has_segments and segment_pos.shape != (batch, length):
            raise ValueError("segment_pos must have shape [B,L]")
        if has_h0 and h0.shape != (batch, width):
            raise ValueError("h0 must have shape [B,D]")
        if prefix_group_size not in (0, 32, 64, 128):
            raise ValueError("prefix_group_size must be 0, 32, 64 or 128")
        if prefix_group_size and not hierarchical_prefix:
            raise ValueError("grouped prefix requires hierarchical_prefix")
        tensors = [x, gate_x_logit, gate_a_logit, a_param]
        x, gate_x_logit, gate_a_logit, a_param = (t.contiguous() for t in tensors)
        segment_pos = segment_pos.contiguous()
        h0 = h0.contiguous()
        out = torch.empty_like(x)
        last = torch.empty((batch, width), device=x.device, dtype=torch.float32)
        block, blocks = _launch_shape(width, block_size)
        common = dict(
            length=length, width=width, HAS_SEGMENTS=has_segments,
            RESET_FIRST=bool(reset_first), ROUND_BF16=x.dtype == torch.bfloat16,
            BLOCK=block, num_warps=_warps(block), num_stages=1,
        )
        if chunk_size == 0:
            _serial_forward[(blocks, batch)](
                x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0,
                out, last, HAS_H0=has_h0, **common,
            )
        else:
            if chunk_size not in (8, 16, 32, 64):
                raise ValueError("chunk_size must be 0, 8, 16, 32 or 64")
            chunks = triton.cdiv(length, chunk_size)
            shape = (batch, chunks, width)
            summary_a = torch.empty(shape, device=x.device, dtype=torch.float32)
            summary_b = torch.empty_like(summary_a)
            chunk_input = torch.empty_like(summary_a)
            grid = (blocks, batch * chunks)
            chunk_common = dict(
                length=length, width=width, chunks=chunks,
                HAS_SEGMENTS=has_segments, RESET_FIRST=bool(reset_first),
                ROUND_BF16=x.dtype == torch.bfloat16, CHUNK=chunk_size,
                BLOCK=block, num_warps=_warps(block), num_stages=1,
            )
            _chunk_summary[grid](
                x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                summary_a, summary_b, **chunk_common,
            )
            if hierarchical_prefix and prefix_group_size and chunks > 1:
                groups = triton.cdiv(chunks, prefix_group_size)
                group_shape = (batch, groups, width)
                group_a = torch.empty(
                    group_shape, device=x.device, dtype=torch.float32
                )
                group_b = torch.empty_like(group_a)
                group_boundary = torch.empty_like(group_a)
                group_grid = (blocks, batch * groups)
                _affine_group_local[group_grid](
                    summary_a, summary_b, group_a, group_b,
                    width=width, chunks=chunks, groups=groups,
                    GROUP=prefix_group_size, REVERSE=False, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
                _affine_group_outer[(blocks, batch)](
                    group_a, group_b, h0, group_boundary, last,
                    width=width, groups=groups, HAS_INITIAL=has_h0,
                    REVERSE=False, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
                _affine_group_correct[grid](
                    summary_a, summary_b, group_boundary, chunk_input,
                    width=width, chunks=chunks, groups=groups,
                    GROUP=prefix_group_size, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            elif hierarchical_prefix and chunks > 1:
                temporary_a = torch.empty_like(summary_a)
                temporary_b = torch.empty_like(summary_b)
                source_a, source_b = summary_a, summary_b
                destination_a, destination_b = temporary_a, temporary_b
                stride = 1
                while stride < chunks:
                    _affine_prefix_stage[grid](
                        source_a, source_b, destination_a, destination_b,
                        width=width, chunks=chunks, STRIDE=stride,
                        REVERSE=False, BLOCK=block,
                        num_warps=_warps(block), num_stages=1,
                    )
                    source_a, destination_a = destination_a, source_a
                    source_b, destination_b = destination_b, source_b
                    stride *= 2
                _affine_prefix_extract[grid](
                    source_a, source_b, h0, chunk_input, last,
                    width=width, chunks=chunks, REVERSE=False,
                    HAS_INITIAL=has_h0, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            else:
                _chunk_prefix[(blocks, batch)](
                    summary_a, summary_b, h0, chunk_input, last,
                    width=width, chunks=chunks, HAS_H0=has_h0, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            _chunk_replay[grid](
                x, gate_x_logit, gate_a_logit, a_param, segment_pos,
                chunk_input, out, **chunk_common,
            )
        ctx.save_for_backward(x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0, out)
        ctx.chunk_size = int(chunk_size)
        ctx.block_size = int(block_size)
        ctx.reset_first = bool(reset_first)
        ctx.has_segments = has_segments
        ctx.has_h0 = has_h0
        ctx.hierarchical_prefix = bool(hierarchical_prefix)
        ctx.prefix_group_size = int(prefix_group_size)
        return out, last

    @staticmethod
    def backward(ctx, grad_out, grad_last):
        x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0, out = ctx.saved_tensors
        batch, length, width = x.shape
        grad_out = grad_out.contiguous()
        if grad_last is None:
            grad_last = torch.zeros((batch, width), device=x.device, dtype=torch.float32)
        else:
            grad_last = grad_last.contiguous().float()
        grad_x = torch.empty_like(x)
        grad_gate_x = torch.empty_like(gate_x_logit)
        grad_gate_a = torch.empty_like(gate_a_logit)
        grad_h0 = torch.empty((batch, width), device=x.device, dtype=torch.float32)
        block, blocks = _launch_shape(width, ctx.block_size)
        if ctx.chunk_size == 0:
            partial = torch.empty((batch, width), device=x.device, dtype=torch.float32)
            _serial_backward[(blocks, batch)](
                x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0, out,
                grad_out, grad_last, grad_x, grad_gate_x, grad_gate_a,
                partial, grad_h0, length=length, width=width,
                HAS_SEGMENTS=ctx.has_segments, RESET_FIRST=ctx.reset_first,
                HAS_H0=ctx.has_h0, ROUND_BF16=x.dtype == torch.bfloat16,
                BLOCK=block, num_warps=_warps(block), num_stages=1,
            )
        else:
            chunks = triton.cdiv(length, ctx.chunk_size)
            shape = (batch, chunks, width)
            summary_a = torch.empty(shape, device=x.device, dtype=torch.float32)
            summary_b = torch.empty_like(summary_a)
            future = torch.empty_like(summary_a)
            partial = torch.empty_like(summary_a)
            grid = (blocks, batch * chunks)
            common = dict(
                length=length, width=width, chunks=chunks,
                HAS_SEGMENTS=ctx.has_segments, RESET_FIRST=ctx.reset_first,
                ROUND_BF16=x.dtype == torch.bfloat16, CHUNK=ctx.chunk_size,
                BLOCK=block, num_warps=_warps(block), num_stages=1,
            )
            _reverse_chunk_summary[grid](
                gate_a_logit, a_param, segment_pos, grad_out,
                summary_a, summary_b, **common,
            )
            if (ctx.hierarchical_prefix and ctx.prefix_group_size
                    and chunks > 1):
                groups = triton.cdiv(chunks, ctx.prefix_group_size)
                group_shape = (batch, groups, width)
                group_a = torch.empty(
                    group_shape, device=x.device, dtype=torch.float32
                )
                group_b = torch.empty_like(group_a)
                group_boundary = torch.empty_like(group_a)
                group_grid = (blocks, batch * groups)
                _affine_group_local[group_grid](
                    summary_a, summary_b, group_a, group_b,
                    width=width, chunks=chunks, groups=groups,
                    GROUP=ctx.prefix_group_size, REVERSE=True, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
                _affine_group_outer[(blocks, batch)](
                    group_a, group_b, grad_last, group_boundary, grad_h0,
                    width=width, groups=groups, HAS_INITIAL=True,
                    REVERSE=True, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
                _affine_group_correct[grid](
                    summary_a, summary_b, group_boundary, future,
                    width=width, chunks=chunks, groups=groups,
                    GROUP=ctx.prefix_group_size, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            elif ctx.hierarchical_prefix and chunks > 1:
                temporary_a = torch.empty_like(summary_a)
                temporary_b = torch.empty_like(summary_b)
                source_a, source_b = summary_a, summary_b
                destination_a, destination_b = temporary_a, temporary_b
                stride = 1
                while stride < chunks:
                    _affine_prefix_stage[grid](
                        source_a, source_b, destination_a, destination_b,
                        width=width, chunks=chunks, STRIDE=stride,
                        REVERSE=True, BLOCK=block,
                        num_warps=_warps(block), num_stages=1,
                    )
                    source_a, destination_a = destination_a, source_a
                    source_b, destination_b = destination_b, source_b
                    stride *= 2
                _affine_prefix_extract[grid](
                    source_a, source_b, grad_last, future, grad_h0,
                    width=width, chunks=chunks, REVERSE=True,
                    HAS_INITIAL=True, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            else:
                _reverse_chunk_prefix[(blocks, batch)](
                    summary_a, summary_b, grad_last, future, grad_h0,
                    width=width, chunks=chunks, BLOCK=block,
                    num_warps=_warps(block), num_stages=1,
                )
            _reverse_chunk_replay[grid](
                x, gate_x_logit, gate_a_logit, a_param, segment_pos, h0, out,
                grad_out, future, grad_x, grad_gate_x, grad_gate_a, partial,
                HAS_H0=ctx.has_h0, **common,
            )
        grad_a_param = partial.sum(dim=tuple(range(partial.ndim - 1)))
        if not ctx.has_h0:
            grad_h0_result = None
        else:
            grad_h0_result = grad_h0
        return (
            grad_x, grad_gate_x, grad_gate_a, grad_a_param,
            None, grad_h0_result, None, None, None, None, None,
        )


def fused_rglru_scan(
    x: torch.Tensor,
    gate_x_logit: torch.Tensor,
    gate_a_logit: torch.Tensor,
    a_param: torch.Tensor,
    *,
    segment_pos: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    chunk_size: int = 0,
    block_size: int = 64,
    reset_first: bool = True,
    hierarchical_prefix: bool = False,
    prefix_group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused RG-LRU preparation+scan and return all states and FP32 last state.

    ``segment_pos`` enables arbitrary packed-sequence resets.  When omitted,
    ``reset_first=True`` is the fast path for one independent sequence per batch
    item.  A generic explicit segment tensor is used by correctness tests.
    """
    empty_segment = x.new_empty((0,), dtype=torch.int32)
    empty_h0 = x.new_empty((0,), dtype=torch.float32)
    return _FusedRGLRU.apply(
        x, gate_x_logit, gate_a_logit, a_param,
        empty_segment if segment_pos is None else segment_pos,
        empty_h0 if h0 is None else h0,
        int(chunk_size), int(block_size), bool(reset_first),
        bool(hierarchical_prefix), int(prefix_group_size),
    )
