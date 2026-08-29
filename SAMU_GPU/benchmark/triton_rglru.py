"""Inference Triton kernels for the official Griffin/RecurrentGemma RG-LRU.

The equations match ``recurrentgemma.torch.layers.RGLRU``:

    i_t = sigmoid(W_x x_t + b_x)
    r_t = sigmoid(W_a x_t + b_a)
    log(a_t) = -8 r_t softplus(a_param)
    h_t = a_t h_{t-1} + sqrt(1 - a_t**2) (i_t x_t)

At a document boundary (``segment_pos == 0``), the official implementation
sets the recurrent coefficient to zero *and* replaces the square-root input
multiplier with one.  The recurrent accumulator/cache is FP32; activations,
gate projections, and returned sequence values are BF16.

Prefill uses one block-diagonal batched GEMM for both gates, followed by either
a serial scan kernel or a three-kernel chunk summary/prefix/replay scan.  Decode
fuses the two block-diagonal matrix-vector products, gates, reset, and recurrent
update in one Triton launch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@dataclass
class PackedRGLRU:
    """Exact packed view of an official RecurrentGemma ``RGLRU`` layer."""

    gate_weight: torch.Tensor
    gate_bias: torch.Tensor
    softplus_a: torch.Tensor
    width: int
    num_heads: int
    head_dim: int


def pack_rglru(model) -> PackedRGLRU:
    """Pack official parameters without adding dense off-block weights.

    ``gate_weight[h, :, :head_dim]`` is W_x and the second half is W_a.  Thus
    the two official block-diagonal projections become one strided batched GEMM
    with exactly the same non-zero multiply count.
    """

    input_weight = model.input_gate.w.detach()
    recurrent_weight = model.a_gate.w.detach()
    input_bias = model.input_gate.b.detach()
    recurrent_bias = model.a_gate.b.detach()
    num_heads, head_dim, _ = input_weight.shape
    return PackedRGLRU(
        gate_weight=torch.cat((input_weight, recurrent_weight), dim=-1).contiguous(),
        gate_bias=torch.cat((input_bias, recurrent_bias), dim=-1).contiguous(),
        # This is a static parameter transform.  Keeping the official BF16 result
        # also preserves the public implementation's inference precision policy.
        softplus_a=F.softplus(model.a_param.detach()).contiguous(),
        width=model.width,
        num_heads=num_heads,
        head_dim=head_dim,
    )


def _project_gates(x: torch.Tensor, packed: PackedRGLRU) -> torch.Tensor:
    """One block-diagonal BMM for both official gates; output is [H, BL, 2Dh]."""

    batch, length, width = x.shape
    if width != packed.width:
        raise ValueError(f"expected width {packed.width}, got {width}")
    blocks = x.reshape(batch * length, packed.num_heads, packed.head_dim)
    return torch.bmm(blocks.permute(1, 0, 2), packed.gate_weight)


@triton.jit
def _bf16_round(value):
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _official_step(x_value, projected_x, projected_a, bias_x, bias_a, softplus_a,
                   is_reset):
    """Official eager-BF16 pointwise order, represented in FP32 registers."""

    # Official BlockDiagonalLinear returns a BF16 activation before RGLRU adds
    # its BF16 bias.  Prefill already arrives here from a BF16 torch.bmm, while
    # fused decode accumulates the dot product in FP32 registers.  Round both
    # paths at the same projection boundary so decode cannot silently gain a
    # different numerical policy.
    projected_x = _bf16_round(projected_x.to(tl.float32))
    projected_a = _bf16_round(projected_a.to(tl.float32))
    raw_x = _bf16_round(projected_x + bias_x.to(tl.float32))
    raw_a = _bf16_round(projected_a + bias_a.to(tl.float32))
    gate_x = _bf16_round(tl.sigmoid(raw_x))
    gate_a = _bf16_round(tl.sigmoid(raw_a))
    # Source expression is: -8.0 * gate_a * softplus(a_param).
    scaled_gate = _bf16_round(-8.0 * gate_a)
    log_a = _bf16_round(scaled_gate * softplus_a.to(tl.float32))
    a = _bf16_round(tl.exp(log_a))
    twice_log_a = _bf16_round(2.0 * log_a)
    a_square = _bf16_round(tl.exp(twice_log_a))
    one_minus = _bf16_round(1.0 - a_square)
    multiplier = _bf16_round(tl.sqrt(tl.maximum(one_minus, 0.0)))
    multiplier = tl.where(is_reset, 1.0, multiplier)
    gated_x = _bf16_round(x_value.to(tl.float32) * gate_x)
    normalized_x = _bf16_round(gated_x * multiplier)
    effective_a = tl.where(is_reset, 0.0, a)
    return effective_a, normalized_x


@triton.jit
def _serial_scan_kernel(x, projected, bias, softplus_a, segment_pos, h0, out,
                        last_h, length, rows, WIDTH: tl.constexpr,
                        HEAD_DIM: tl.constexpr, HAS_H0: tl.constexpr,
                        BLOCK: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_DIM
    channel = head * HEAD_DIM + lane
    state = tl.load(h0 + batch * WIDTH + channel, mask=mask, other=0.0) if HAS_H0 else tl.zeros((BLOCK,), tl.float32)
    bias_base = head * (2 * HEAD_DIM)
    bx = tl.load(bias + bias_base + lane, mask=mask, other=0.0)
    ba = tl.load(bias + bias_base + HEAD_DIM + lane, mask=mask, other=0.0)
    spa = tl.load(softplus_a + channel, mask=mask, other=0.0)
    for t in tl.range(0, length, 1, num_stages=1):
        row = batch * length + t
        projected_base = (head * rows + row) * (2 * HEAD_DIM)
        px = tl.load(projected + projected_base + lane, mask=mask, other=0.0)
        pa = tl.load(projected + projected_base + HEAD_DIM + lane, mask=mask, other=0.0)
        xv = tl.load(x + row * WIDTH + channel, mask=mask, other=0.0)
        reset = tl.load(segment_pos + row) == 0
        a, write = _official_step(xv, px, pa, bx, ba, spa, reset)
        state = a * state + write
        tl.store(out + row * WIDTH + channel, state, mask=mask)
    tl.store(last_h + batch * WIDTH + channel, state, mask=mask)


@triton.jit
def _chunk_summary_kernel(x, projected, bias, softplus_a, segment_pos, summary_a,
                          summary_b, length, rows, chunks, WIDTH: tl.constexpr,
                          HEAD_DIM: tl.constexpr, CHUNK: tl.constexpr,
                          BLOCK: tl.constexpr):
    head = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_DIM
    channel = head * HEAD_DIM + lane
    bias_base = head * (2 * HEAD_DIM)
    bx = tl.load(bias + bias_base + lane, mask=mask, other=0.0)
    ba = tl.load(bias + bias_base + HEAD_DIM + lane, mask=mask, other=0.0)
    spa = tl.load(softplus_a + channel, mask=mask, other=0.0)
    affine_a = tl.full((BLOCK,), 1.0, tl.float32)
    affine_b = tl.zeros((BLOCK,), tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        valid = t < length
        row = batch * length + t
        projected_base = (head * rows + row) * (2 * HEAD_DIM)
        px = tl.load(projected + projected_base + lane, mask=mask & valid, other=0.0)
        pa = tl.load(projected + projected_base + HEAD_DIM + lane, mask=mask & valid, other=0.0)
        xv = tl.load(x + row * WIDTH + channel, mask=mask & valid, other=0.0)
        reset = (tl.load(segment_pos + row, mask=valid, other=1) == 0) & valid
        step_a, write = _official_step(xv, px, pa, bx, ba, spa, reset)
        step_a = tl.where(valid, step_a, 1.0)
        write = tl.where(valid, write, 0.0)
        affine_a = step_a * affine_a
        affine_b = step_a * affine_b + write
    summary_offset = (chunk_program * WIDTH) + channel
    tl.store(summary_a + summary_offset, affine_a, mask=mask)
    tl.store(summary_b + summary_offset, affine_b, mask=mask)


@triton.jit
def _chunk_prefix_kernel(summary_a, summary_b, h0, chunk_input, last_h,
                         chunks, WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr,
                         HAS_H0: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_DIM
    channel = head * HEAD_DIM + lane
    state = tl.load(h0 + batch * WIDTH + channel, mask=mask, other=0.0) if HAS_H0 else tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * WIDTH + channel
        tl.store(chunk_input + offset, state, mask=mask)
        a = tl.load(summary_a + offset, mask=mask, other=1.0)
        b = tl.load(summary_b + offset, mask=mask, other=0.0)
        state = a * state + b
    tl.store(last_h + batch * WIDTH + channel, state, mask=mask)


@triton.jit
def _chunk_replay_kernel(x, projected, bias, softplus_a, segment_pos,
                         chunk_input, out, length, rows, chunks,
                         WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr,
                         CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = tl.arange(0, BLOCK)
    mask = lane < HEAD_DIM
    channel = head * HEAD_DIM + lane
    state_offset = chunk_program * WIDTH + channel
    state = tl.load(chunk_input + state_offset, mask=mask, other=0.0)
    bias_base = head * (2 * HEAD_DIM)
    bx = tl.load(bias + bias_base + lane, mask=mask, other=0.0)
    ba = tl.load(bias + bias_base + HEAD_DIM + lane, mask=mask, other=0.0)
    spa = tl.load(softplus_a + channel, mask=mask, other=0.0)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        valid = t < length
        row = batch * length + t
        projected_base = (head * rows + row) * (2 * HEAD_DIM)
        px = tl.load(projected + projected_base + lane, mask=mask & valid, other=0.0)
        pa = tl.load(projected + projected_base + HEAD_DIM + lane, mask=mask & valid, other=0.0)
        xv = tl.load(x + row * WIDTH + channel, mask=mask & valid, other=0.0)
        reset = (tl.load(segment_pos + row, mask=valid, other=1) == 0) & valid
        a, write = _official_step(xv, px, pa, bx, ba, spa, reset)
        state = a * state + write
        tl.store(out + row * WIDTH + channel, state, mask=mask & valid)


@triton.jit
def _decode_kernel(x, weight, bias, softplus_a, segment_pos, h0, out, last_h,
                   WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr,
                   BLOCK_I: tl.constexpr, BLOCK_O: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    i = tl.arange(0, BLOCK_I)
    j = tl.arange(0, BLOCK_O)
    imask, jmask = i < HEAD_DIM, j < HEAD_DIM
    channel_i = head * HEAD_DIM + i
    channel_j = head * HEAD_DIM + j
    xv = tl.load(x + batch * WIDTH + channel_i, mask=imask, other=0.0)
    weight_base = head * HEAD_DIM * (2 * HEAD_DIM)
    offsets_x = weight_base + i[:, None] * (2 * HEAD_DIM) + j[None, :]
    offsets_a = offsets_x + HEAD_DIM
    product_mask = imask[:, None] & jmask[None, :]
    wx = tl.load(weight + offsets_x, mask=product_mask, other=0.0)
    projected_x = tl.sum(xv[:, None].to(tl.float32) * wx.to(tl.float32), axis=0)
    wa = tl.load(weight + offsets_a, mask=product_mask, other=0.0)
    projected_a = tl.sum(xv[:, None].to(tl.float32) * wa.to(tl.float32), axis=0)
    bias_base = head * (2 * HEAD_DIM)
    bx = tl.load(bias + bias_base + j, mask=jmask, other=0.0)
    ba = tl.load(bias + bias_base + HEAD_DIM + j, mask=jmask, other=0.0)
    spa = tl.load(softplus_a + channel_j, mask=jmask, other=0.0)
    token_value = tl.load(x + batch * WIDTH + channel_j, mask=jmask, other=0.0)
    reset = tl.load(segment_pos + batch) == 0
    a, write = _official_step(token_value, projected_x, projected_a, bx, ba, spa, reset)
    state = tl.load(h0 + batch * WIDTH + channel_j, mask=jmask, other=0.0)
    state = a * state + write
    tl.store(out + batch * WIDTH + channel_j, state, mask=jmask)
    tl.store(last_h + batch * WIDTH + channel_j, state, mask=jmask)


def _validate_inputs(x: torch.Tensor, segment_pos: torch.Tensor,
                     packed: PackedRGLRU) -> None:
    if x.ndim != 3 or x.shape[-1] != packed.width:
        raise ValueError(f"x must have shape [B, L, {packed.width}]")
    if segment_pos.shape != x.shape[:2]:
        raise ValueError("segment_pos must have shape [B, L]")
    if x.dtype != torch.bfloat16:
        raise ValueError("the audited optimized path currently requires BF16 input")


def rglru_triton_serial(x: torch.Tensor, segment_pos: torch.Tensor,
                        packed: PackedRGLRU, cache: torch.Tensor | None = None,
                        *, num_warps: int | None = None):
    _validate_inputs(x, segment_pos, packed)
    batch, length, _ = x.shape
    projected = _project_gates(x, packed)
    out = torch.empty_like(x)
    last_h = torch.empty(batch, packed.width, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(packed.head_dim)
    placeholder = cache if cache is not None else last_h
    launch_warps = num_warps or (8 if block >= 256 else 4 if block >= 64 else 2)
    _serial_scan_kernel[(packed.num_heads, batch)](
        x, projected, packed.gate_bias, packed.softplus_a, segment_pos,
        placeholder, out, last_h, length, batch * length,
        WIDTH=packed.width, HEAD_DIM=packed.head_dim, HAS_H0=cache is not None,
        BLOCK=block, num_warps=launch_warps,
    )
    return out, last_h


def rglru_triton_chunked(x: torch.Tensor, segment_pos: torch.Tensor,
                         packed: PackedRGLRU, chunk_size: int,
                         cache: torch.Tensor | None = None, *,
                         num_warps: int | None = None):
    _validate_inputs(x, segment_pos, packed)
    batch, length, _ = x.shape
    chunks = triton.cdiv(length, chunk_size)
    projected = _project_gates(x, packed)
    summary_shape = (batch, chunks, packed.width)
    summary_a = torch.empty(summary_shape, device=x.device, dtype=torch.float32)
    summary_b = torch.empty_like(summary_a)
    chunk_input = torch.empty_like(summary_a)
    out = torch.empty_like(x)
    last_h = torch.empty(batch, packed.width, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(packed.head_dim)
    warps = num_warps or (4 if block >= 64 else 2)
    grid = (packed.num_heads, batch * chunks)
    common = dict(length=length, rows=batch * length, chunks=chunks,
                  WIDTH=packed.width, HEAD_DIM=packed.head_dim,
                  CHUNK=chunk_size, BLOCK=block, num_warps=warps)
    _chunk_summary_kernel[grid](
        x, projected, packed.gate_bias, packed.softplus_a, segment_pos,
        summary_a, summary_b, **common,
    )
    placeholder = cache if cache is not None else last_h
    _chunk_prefix_kernel[(packed.num_heads, batch)](
        summary_a, summary_b, placeholder, chunk_input, last_h, chunks,
        WIDTH=packed.width, HEAD_DIM=packed.head_dim, HAS_H0=cache is not None,
        BLOCK=block, num_warps=warps,
    )
    _chunk_replay_kernel[grid](
        x, projected, packed.gate_bias, packed.softplus_a, segment_pos,
        chunk_input, out, **common,
    )
    return out, last_h


def rglru_triton_auto(x: torch.Tensor, segment_pos: torch.Tensor,
                      packed: PackedRGLRU, cache: torch.Tensor | None = None):
    """Length-aware default; architecture-specific sweeps are recorded separately."""

    length = x.shape[1]
    if length <= 256:
        return rglru_triton_chunked(
            x, segment_pos, packed, 8, cache, num_warps=4
        )
    if length <= 1024:
        return rglru_triton_chunked(
            x, segment_pos, packed, 16, cache, num_warps=4
        )
    return rglru_triton_chunked(
        x, segment_pos, packed, 32, cache,
        num_warps=4 if length <= 2048 else 2,
    )


def rglru_triton_decode(x: torch.Tensor, segment_pos: torch.Tensor,
                        packed: PackedRGLRU, cache: torch.Tensor, *,
                        num_warps: int = 2):
    """One-token fused RG-LRU update.

    ``x`` may be [B, D] or [B, 1, D]; ``segment_pos`` may be [B] or [B, 1].
    """

    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError("decode accepts exactly one token")
        x = x[:, 0]
    if segment_pos.ndim == 2:
        segment_pos = segment_pos[:, 0]
    # Slicing [B, L, D] at L=1 can retain the original sequence stride.  The
    # fused kernel deliberately uses a compact [B, D] layout.
    x = x.contiguous()
    segment_pos = segment_pos.contiguous()
    cache = cache.contiguous()
    if x.shape[-1] != packed.width or cache.shape != x.shape:
        raise ValueError("decode x/cache shape mismatch")
    batch = x.shape[0]
    out = torch.empty_like(x)
    last_h = torch.empty_like(cache, dtype=torch.float32)
    block = triton.next_power_of_2(packed.head_dim)
    _decode_kernel[(packed.num_heads, batch)](
        x, packed.gate_weight, packed.gate_bias, packed.softplus_a,
        segment_pos, cache, out, last_h, WIDTH=packed.width,
        HEAD_DIM=packed.head_dim, BLOCK_I=block, BLOCK_O=block,
        num_warps=num_warps,
    )
    return out[:, None], last_h
