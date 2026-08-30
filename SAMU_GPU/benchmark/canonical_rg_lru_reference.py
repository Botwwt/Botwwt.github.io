"""Canonical eager PyTorch RG-LRU recurrence used as GPU-test ground truth.

The equations, reset behavior and clipped square-root derivative follow the
Google DeepMind RecurrentGemma PyTorch source pinned at
2efa84dac0e68e63547a27a18fa943c98f1c312e.  This module is deliberately eager
and serial.  It defines semantics and is never used as an optimized baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SqrtBoundDerivative(torch.autograd.Function):
    """Square root with RecurrentGemma's BF16-safe clipped derivative."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.sqrt(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return grad_output / torch.sqrt(torch.clamp(4.0 * value, min=1.0e-6))


def block_diagonal_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply official ``[..., H, W] @ [H, W, W] + [H, W]`` gates."""
    if weight.ndim != 3 or weight.shape[1] != weight.shape[2]:
        raise ValueError("weight must have shape [num_blocks, block_width, block_width]")
    blocks, block_width, _ = weight.shape
    if inputs.shape[-1] != blocks * block_width or bias.shape != (blocks, block_width):
        raise ValueError("projection shapes do not match the recurrent width")
    shaped = inputs.reshape(*inputs.shape[:-1], blocks, block_width)
    projected = torch.einsum("...hi,hij->...hj", shaped, weight) + bias
    return projected.reshape_as(inputs)


def canonical_rg_lru_from_logits(
    inputs: torch.Tensor,
    input_gate_logits: torch.Tensor,
    recurrence_gate_logits: torch.Tensor,
    a_param: torch.Tensor,
    *,
    segment_pos: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    production_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all states and the FP32 final state for canonical RG-LRU.

    Computation is FP32.  ``production_dtype=torch.bfloat16`` additionally
    models the optimized project's BF16 transition/write and output boundaries
    while retaining FP32 recurrent accumulation.
    """
    if inputs.shape != input_gate_logits.shape or inputs.shape != recurrence_gate_logits.shape:
        raise ValueError("inputs and gate logits must have matching [B,L,D] shapes")
    batch, length, width = inputs.shape
    if a_param.shape != (width,) or segment_pos.shape != (batch, length):
        raise ValueError("a_param or segment_pos has the wrong shape")
    gate_x = torch.sigmoid(input_gate_logits.float())
    gate_a = torch.sigmoid(recurrence_gate_logits.float())
    log_a = -8.0 * gate_a * F.softplus(a_param.float())
    transition = torch.exp(log_a)
    multiplier = SqrtBoundDerivative.apply(
        (1.0 - torch.exp(2.0 * log_a)).clamp_min(0.0)
    )
    reset = segment_pos.eq(0).unsqueeze(-1)
    transition = torch.where(reset, torch.zeros_like(transition), transition)
    multiplier = torch.where(reset, torch.ones_like(multiplier), multiplier)
    write = inputs.float() * gate_x * multiplier
    if production_dtype is not None:
        transition = transition.to(production_dtype).float()
        write = write.to(production_dtype).float()
    state = (
        torch.zeros((batch, width), device=inputs.device, dtype=torch.float32)
        if initial_state is None else initial_state.float()
    )
    outputs = []
    for time_index in range(length):
        state = transition[:, time_index] * state + write[:, time_index]
        outputs.append(state if production_dtype is None else state.to(production_dtype))
    return torch.stack(outputs, dim=1), state


def canonical_rg_lru(
    inputs: torch.Tensor,
    input_gate_weight: torch.Tensor,
    input_gate_bias: torch.Tensor,
    recurrence_gate_weight: torch.Tensor,
    recurrence_gate_bias: torch.Tensor,
    a_param: torch.Tensor,
    *,
    segment_pos: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    production_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical RG-LRU including both official block-diagonal projections."""
    gate_x = block_diagonal_projection(inputs, input_gate_weight, input_gate_bias)
    gate_a = block_diagonal_projection(
        inputs, recurrence_gate_weight, recurrence_gate_bias
    )
    return canonical_rg_lru_from_logits(
        inputs, gate_x, gate_a, a_param, segment_pos=segment_pos,
        initial_state=initial_state, production_dtype=production_dtype,
    )
