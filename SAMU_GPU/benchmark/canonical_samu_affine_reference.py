"""FP32 reference for ordered complex affine SAMU composition."""

from __future__ import annotations

import torch


def compose_complex_affine(left, right):
    """Return ``right o left`` for transforms ``lambda*h + b``."""
    left_lambda, left_write = left
    right_lambda, right_write = right
    return (
        right_lambda * left_lambda,
        right_lambda * left_write + right_write,
    )


def samu_sequential_from_shared(
    e: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    nu: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    write: torch.Tensor,
    *,
    segment_pos: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    reset_first: bool = False,
):
    """Canonical complex recurrence using shared E/C/S and optional resets."""
    batch, length, modes = write.shape
    state = (torch.zeros(batch, modes, device=write.device, dtype=torch.complex64)
             if initial_state is None else initial_state.to(torch.complex64))
    values = []
    for time in range(length):
        radius = torch.exp(-nu * e[:, time, None])
        transition = torch.complex(
            radius * (cos_theta * c[:, time, None] -
                      sin_theta * s[:, time, None]),
            radius * (sin_theta * c[:, time, None] +
                      cos_theta * s[:, time, None]),
        )
        if segment_pos is not None:
            reset = segment_pos[:, time].eq(0)[:, None]
        elif reset_first and time == 0:
            reset = torch.ones(batch, 1, device=write.device, dtype=torch.bool)
        else:
            reset = torch.zeros(batch, 1, device=write.device, dtype=torch.bool)
        transition = torch.where(reset, torch.zeros_like(transition), transition)
        state = transition * state + write[:, time]
        values.append(state)
    return torch.stack(values, dim=1), state


def ordered_chunk_summary(transitions: torch.Tensor, writes: torch.Tensor):
    """Reference ordered reduction for one or more independent chunks."""
    summary_lambda = torch.ones_like(transitions[..., 0, :])
    summary_write = torch.zeros_like(writes[..., 0, :])
    for time in range(transitions.shape[-2]):
        summary_lambda, summary_write = compose_complex_affine(
            (summary_lambda, summary_write),
            (transitions[..., time, :], writes[..., time, :]),
        )
    return summary_lambda, summary_write
