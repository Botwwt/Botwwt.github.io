"""FP32-accumulating D->2 SAMU controller without a logical x.float() tensor."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _formal_coordinate_forward(
    projected, phase_amplitude, radial_amplitude,
    eta, delta, phase_coordinate, radial_raw,
    rows: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    """One-launch exact formal coordinate map over the two projections."""
    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    phase_projection = tl.load(
        projected + offset * 2, mask=mask, other=0.0
    ).to(tl.float32)
    radial_projection = tl.load(
        projected + offset * 2 + 1, mask=mask, other=0.0
    ).to(tl.float32)
    phase = libdevice.tanh(phase_projection)
    radial = libdevice.tanh(radial_projection)
    radial_coordinate = radial / (1.0 + radial * radial)
    phase_strength = libdevice.tanh(
        tl.load(phase_amplitude).to(tl.float32)
    )
    radial_strength = libdevice.tanh(
        tl.load(radial_amplitude).to(tl.float32)
    )
    tl.store(phase_coordinate + offset, phase, mask=mask)
    tl.store(radial_raw + offset, radial, mask=mask)
    tl.store(delta + offset, SCALE * phase_strength * phase, mask=mask)
    tl.store(eta + offset, SCALE * radial_strength * radial_coordinate, mask=mask)


@triton.jit
def _formal_coordinate_backward(
    phase_coordinate, radial_raw, grad_eta, grad_delta,
    phase_amplitude, radial_amplitude, grad_projected, partial_amplitude,
    rows: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    phase = tl.load(phase_coordinate + offset, mask=mask, other=0.0).to(tl.float32)
    radial = tl.load(radial_raw + offset, mask=mask, other=0.0).to(tl.float32)
    g_eta = tl.load(grad_eta + offset, mask=mask, other=0.0).to(tl.float32)
    g_delta = tl.load(grad_delta + offset, mask=mask, other=0.0).to(tl.float32)
    phase_s = libdevice.tanh(tl.load(phase_amplitude).to(tl.float32))
    radial_s = libdevice.tanh(tl.load(radial_amplitude).to(tl.float32))
    radial_square = radial * radial
    denominator = 1.0 + radial_square
    radial_coordinate = radial / denominator
    radial_derivative = (1.0 - radial_square) / (denominator * denominator)
    grad_phase = g_delta * SCALE * phase_s * (1.0 - phase * phase)
    grad_radial = (
        g_eta * SCALE * radial_s * radial_derivative * (1.0 - radial_square)
    )
    tl.store(grad_projected + offset * 2, grad_phase, mask=mask)
    tl.store(grad_projected + offset * 2 + 1, grad_radial, mask=mask)
    phase_amplitude = g_delta * SCALE * phase * (1.0 - phase_s * phase_s)
    radial_amplitude = (
        g_eta * SCALE * radial_coordinate * (1.0 - radial_s * radial_s)
    )
    tl.store(partial_amplitude + program * 2,
             tl.sum(tl.where(mask, phase_amplitude, 0.0), axis=0))
    tl.store(partial_amplitude + program * 2 + 1,
             tl.sum(tl.where(mask, radial_amplitude, 0.0), axis=0))


@triton.jit
def _formal_coordinate_reduce(
    partial_amplitude, grad_phase_amplitude, grad_radial_amplitude,
    blocks: tl.constexpr, REDUCE: tl.constexpr,
):
    offset = tl.arange(0, REDUCE)
    mask = offset < blocks
    phase = tl.load(partial_amplitude + offset * 2, mask=mask, other=0.0)
    radial = tl.load(partial_amplitude + offset * 2 + 1, mask=mask, other=0.0)
    tl.store(grad_phase_amplitude, tl.sum(phase, axis=0))
    tl.store(grad_radial_amplitude, tl.sum(radial, axis=0))


class _FormalSAMUCoordinates(torch.autograd.Function):
    """Exact formal controller nonlinearities with a compact p/r cache."""

    @staticmethod
    def forward(ctx, projected, phase_amplitude, radial_amplitude, modes: int):
        projected = projected.contiguous()
        output_shape = projected.shape[:-1]
        eta = torch.empty(output_shape, device=projected.device, dtype=torch.float32)
        delta = torch.empty_like(eta)
        phase_coordinate = torch.empty_like(eta)
        radial_raw = torch.empty_like(eta)
        scale = 1.0 / math.sqrt(modes)
        rows = eta.numel()
        block = 256
        _formal_coordinate_forward[(triton.cdiv(rows, block),)](
            projected, phase_amplitude, radial_amplitude,
            eta, delta, phase_coordinate, radial_raw,
            rows=rows, SCALE=scale, BLOCK=block, num_warps=4,
        )
        ctx.save_for_backward(
            phase_coordinate, radial_raw, phase_amplitude, radial_amplitude
        )
        ctx.scale = scale
        return eta, delta

    @staticmethod
    def backward(ctx, grad_eta, grad_delta):
        phase_coordinate, radial_raw, phase_amplitude, radial_amplitude = ctx.saved_tensors
        grad_eta, grad_delta = grad_eta.contiguous(), grad_delta.contiguous()
        rows = grad_eta.numel()
        block = 256
        blocks = triton.cdiv(rows, block)
        grad_projected = torch.empty(
            (*grad_eta.shape, 2), device=grad_eta.device, dtype=torch.float32
        )
        partial_amplitude = torch.empty(
            (blocks, 2), device=grad_eta.device, dtype=torch.float32
        )
        _formal_coordinate_backward[(blocks,)](
            phase_coordinate, radial_raw, grad_eta, grad_delta,
            phase_amplitude, radial_amplitude, grad_projected, partial_amplitude,
            rows=rows, SCALE=ctx.scale, BLOCK=block, num_warps=4,
        )
        grad_phase_amplitude = torch.empty_like(phase_amplitude)
        grad_radial_amplitude = torch.empty_like(radial_amplitude)
        reduce = triton.next_power_of_2(blocks)
        _formal_coordinate_reduce[(1,)](
            partial_amplitude, grad_phase_amplitude, grad_radial_amplitude,
            blocks=blocks, REDUCE=reduce,
            num_warps=1 if reduce <= 32 else 4,
        )
        return grad_projected, grad_phase_amplitude, grad_radial_amplitude, None


def formal_samu_coordinates(projected, phase_amplitude, radial_amplitude, modes):
    return _FormalSAMUCoordinates.apply(
        projected, phase_amplitude, radial_amplitude, int(modes)
    )


def formal_samu_coordinates_forward_cache(
    projected, phase_amplitude, radial_amplitude, modes,
):
    """Forward-only formal coordinates plus the exact compact p/r cache.

    This is used when the recurrent autograd function owns the low-rank
    controller backward.  No controller graph or FP32 copy of the recurrent
    input is retained, while the formal normalized-direction and bounded
    radial equations stay unchanged.
    """
    projected = projected.contiguous()
    output_shape = projected.shape[:-1]
    eta = torch.empty(output_shape, device=projected.device, dtype=torch.float32)
    delta = torch.empty_like(eta)
    phase_coordinate = torch.empty_like(eta)
    radial_raw = torch.empty_like(eta)
    rows = eta.numel()
    block = 256
    _formal_coordinate_forward[(triton.cdiv(rows, block),)](
        projected, phase_amplitude, radial_amplitude,
        eta, delta, phase_coordinate, radial_raw,
        rows=rows, SCALE=1.0 / math.sqrt(int(modes)),
        BLOCK=block, num_warps=4,
    )
    return eta, delta, phase_coordinate, radial_raw


class _FormalSAMUControllerRecompute(torch.autograd.Function):
    """Exact FP32 controller forward; recompute projection and p/r in backward."""

    @staticmethod
    def forward(ctx, x, phase_direction, radial_direction,
                phase_amplitude, radial_amplitude, modes: int):
        directions = F.normalize(
            torch.stack((phase_direction, radial_direction)).float(), dim=1
        )
        projected = F.linear(
            x.float(), directions[:, :-1], directions[:, -1]
        )
        phase_coordinate = torch.tanh(projected[..., 0])
        radial_raw = torch.tanh(projected[..., 1])
        radial_coordinate = radial_raw / (1.0 + radial_raw.square())
        scale = 1.0 / math.sqrt(modes)
        delta = scale * torch.tanh(phase_amplitude) * phase_coordinate
        eta = scale * torch.tanh(radial_amplitude) * radial_coordinate
        ctx.save_for_backward(
            x, phase_direction, radial_direction,
            phase_amplitude, radial_amplitude,
        )
        ctx.modes = int(modes)
        return eta, delta

    @staticmethod
    def backward(ctx, grad_eta, grad_delta):
        (x, phase_direction, radial_direction,
         phase_amplitude, radial_amplitude) = ctx.saved_tensors
        with torch.enable_grad():
            x_replay = x.detach().float().requires_grad_(True)
            phase_replay = phase_direction.detach().requires_grad_(True)
            radial_replay = radial_direction.detach().requires_grad_(True)
            phase_amplitude_replay = phase_amplitude.detach().requires_grad_(True)
            radial_amplitude_replay = radial_amplitude.detach().requires_grad_(True)
            directions = F.normalize(
                torch.stack((phase_replay, radial_replay)).float(), dim=1
            )
            projected = F.linear(
                x_replay, directions[:, :-1], directions[:, -1]
            )
            phase_coordinate = torch.tanh(projected[..., 0])
            radial_raw = torch.tanh(projected[..., 1])
            radial_coordinate = radial_raw / (1.0 + radial_raw.square())
            scale = 1.0 / math.sqrt(ctx.modes)
            delta = scale * torch.tanh(phase_amplitude_replay) * phase_coordinate
            eta = scale * torch.tanh(radial_amplitude_replay) * radial_coordinate
            gradients = torch.autograd.grad(
                (eta, delta),
                (x_replay, phase_replay, radial_replay,
                 phase_amplitude_replay, radial_amplitude_replay),
                (grad_eta.contiguous(), grad_delta.contiguous()),
            )
        return gradients[0].to(x.dtype), *gradients[1:], None


def formal_samu_controller_recompute(
    x, phase_direction, radial_direction,
    phase_amplitude, radial_amplitude, modes,
):
    return _FormalSAMUControllerRecompute.apply(
        x, phase_direction, radial_direction,
        phase_amplitude, radial_amplitude, int(modes),
    )


@triton.jit
def _controller_partial(x, weight, partial, rows: tl.constexpr,
                        width: tl.constexpr, tiles: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lane = tile * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    value = tl.load(x + row * width + lane, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight + lane, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + width + lane, mask=mask, other=0.0).to(tl.float32)
    base = (row * tiles + tile) * 2
    tl.store(partial + base, tl.sum(value * w0, axis=0))
    tl.store(partial + base + 1, tl.sum(value * w1, axis=0))


@triton.jit
def _controller_reduce(partial, bias, output,
                       rows: tl.constexpr, tiles: tl.constexpr,
                       REDUCE: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, REDUCE)
    mask = tile < tiles
    base = row * tiles * 2
    value0 = tl.load(partial + base + tile * 2, mask=mask, other=0.0)
    value1 = tl.load(partial + base + tile * 2 + 1, mask=mask, other=0.0)
    tl.store(output + row * 2, tl.sum(value0, axis=0) + tl.load(bias))
    tl.store(output + row * 2 + 1,
             tl.sum(value1, axis=0) + tl.load(bias + 1))


@triton.jit
def _controller_grad_x(grad_output, weight, grad_x,
                       rows: tl.constexpr, width: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    width_block = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    grad0 = tl.load(grad_output + row * 2).to(tl.float32)
    grad1 = tl.load(grad_output + row * 2 + 1).to(tl.float32)
    w0 = tl.load(weight + lane, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + width + lane, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_x + row * width + lane, grad0 * w0 + grad1 * w1,
             mask=mask)


@triton.jit
def _controller_grad_weight_partial(x, grad_output, partial,
                                    rows: tl.constexpr,
                                    width: tl.constexpr,
                                    row_chunks: tl.constexpr,
                                    ROW_CHUNK: tl.constexpr,
                                    BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    row_chunk = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    accumulator0 = tl.zeros((BLOCK,), tl.float32)
    accumulator1 = tl.zeros((BLOCK,), tl.float32)
    for local_row in tl.range(0, ROW_CHUNK, 1, num_stages=1):
        row = row_chunk * ROW_CHUNK + local_row
        valid = row < rows
        value = tl.load(
            x + row * width + lane,
            mask=lane_mask & valid,
            other=0.0,
        ).to(tl.float32)
        grad0 = tl.load(
            grad_output + row * 2, mask=valid, other=0.0
        ).to(tl.float32)
        grad1 = tl.load(
            grad_output + row * 2 + 1, mask=valid, other=0.0
        ).to(tl.float32)
        accumulator0 += value * grad0
        accumulator1 += value * grad1
    base = row_chunk * 2 * width + lane
    tl.store(partial + base, accumulator0, mask=lane_mask)
    tl.store(partial + base + width, accumulator1, mask=lane_mask)


class _SharedControllerProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if x.ndim != 3 or weight.shape != (2, x.shape[-1]) or bias.shape != (2,):
            raise ValueError("controller expects x[B,L,D], weight[2,D], bias[2]")
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        rows, width = x.numel() // x.shape[-1], x.shape[-1]
        block = 256
        tiles = triton.cdiv(width, block)
        partial = torch.empty((rows, tiles, 2), device=x.device, dtype=torch.float32)
        output = torch.empty((rows, 2), device=x.device, dtype=torch.float32)
        _controller_partial[(rows, tiles)](
            x, weight, partial, rows=rows, width=width, tiles=tiles,
            BLOCK=block, num_warps=4,
        )
        reduce = triton.next_power_of_2(tiles)
        _controller_reduce[(rows,)](
            partial, bias, output, rows=rows, tiles=tiles, REDUCE=reduce,
            num_warps=1,
        )
        ctx.save_for_backward(x, weight)
        return output.view(*x.shape[:-1], 2)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, 2)
        rows, width = grad_output.shape[0], x.shape[-1]
        grad_x = torch.empty_like(x)
        block = 128
        width_blocks = triton.cdiv(width, block)
        _controller_grad_x[(rows, width_blocks)](
            grad_output, weight, grad_x,
            rows=rows, width=width, BLOCK=block, num_warps=4,
        )
        row_chunk = 128
        row_chunks = triton.cdiv(rows, row_chunk)
        partial = torch.empty(
            (row_chunks, 2, width), device=x.device, dtype=torch.float32
        )
        _controller_grad_weight_partial[(width_blocks, row_chunks)](
            x, grad_output, partial,
            rows=rows, width=width, row_chunks=row_chunks,
            ROW_CHUNK=row_chunk, BLOCK=block, num_warps=4,
        )
        grad_weight = partial.sum(dim=0)
        grad_bias = grad_output.sum(dim=0)
        return grad_x, grad_weight, grad_bias


def shared_controller_projection(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Project BF16/FP32 activations to two FP32 controls with FP32 weights."""
    return _SharedControllerProjection.apply(x, weight, bias)


class _ReferenceControllerRecompute(torch.autograd.Function):
    """Keep cuBLAS FP32 numerics while recomputing the large cast in backward."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.nn.functional.linear(x.float(), weight, bias)
        # Saving BF16 x instead of x.float() removes the persistent [B,L,D]
        # FP32 activation. Backward pays one extra cast and calls PyTorch's
        # native linear backward to preserve its reduction behavior.
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # ``aten::linear_backward`` has no direct CUDA kernel in the project's
        # PyTorch 2.1 build. Re-enter autograd on detached temporaries so the
        # same native CUDA linear backward path is selected.
        with torch.enable_grad():
            x_fp32 = x.detach().float().requires_grad_(True)
            weight_replay = weight.detach().requires_grad_(True)
            replay = torch.nn.functional.linear(x_fp32, weight_replay, None)
            grad_x_fp32, grad_weight = torch.autograd.grad(
                replay, (x_fp32, weight_replay), grad_output.contiguous()
            )
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        return grad_x_fp32.to(x.dtype), grad_weight, grad_bias


def reference_controller_projection_recompute(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _ReferenceControllerRecompute.apply(x, weight, bias)


class _ReferenceForwardTritonBackward(torch.autograd.Function):
    """Bit-identical cuBLAS forward with a BF16-reading Triton backward."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.nn.functional.linear(x.float(), weight, bias)
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, 2)
        rows, width = grad_output.shape[0], x.shape[-1]
        grad_x = torch.empty_like(x)
        block = 128
        width_blocks = triton.cdiv(width, block)
        _controller_grad_x[(rows, width_blocks)](
            grad_output, weight, grad_x,
            rows=rows, width=width, BLOCK=block, num_warps=4,
        )
        row_chunk = 128
        row_chunks = triton.cdiv(rows, row_chunk)
        partial = torch.empty(
            (row_chunks, 2, width), device=x.device, dtype=torch.float32
        )
        _controller_grad_weight_partial[(width_blocks, row_chunks)](
            x, grad_output, partial,
            rows=rows, width=width, row_chunks=row_chunks,
            ROW_CHUNK=row_chunk, BLOCK=block, num_warps=4,
        )
        return grad_x, partial.sum(dim=0), grad_output.sum(dim=0)


def reference_forward_triton_backward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _ReferenceForwardTritonBackward.apply(x, weight, bias)
