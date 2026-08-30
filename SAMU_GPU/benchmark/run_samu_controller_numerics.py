"""Layered numerical and performance audit of the formal SAMU controller.

This isolates the normalized D->2 projection from the recurrent scan.  It is
deliberately separate from the end-to-end mixer checks so recurrence error
amplification is never used to excuse a projection that is already inaccurate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from triton_samu_controller import (
    reference_controller_projection_recompute,
    reference_forward_triton_backward,
    shared_controller_projection,
)


def parse_shape(value: str) -> tuple[int, int, int]:
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("shape must be BxLxD")
    return tuple(int(piece) for piece in pieces)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().flatten()
    candidate = candidate.detach().float().flatten()
    difference = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    denominator = reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    if float(reference_norm) == 0.0 and float(candidate_norm) == 0.0:
        cosine = 1.0
    elif float(reference_norm) == 0.0 or float(candidate_norm) == 0.0:
        cosine = 0.0
    else:
        cosine = float(F.cosine_similarity(reference, candidate, dim=0))
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
        "cosine_similarity": cosine,
    }


def projection(name, x, weight, bias):
    if name == "fp32_reference":
        return F.linear(x.float(), weight, bias)
    if name == "triton_tree_fp32":
        return shared_controller_projection(x, weight, bias)
    if name == "cublas_bf16_fp32_accum":
        # CUDA BF16 GEMM accumulates its dot products in FP32, but the API
        # boundary rounds the two projected coordinates to BF16.
        return F.linear(x, weight.to(x.dtype), bias.to(x.dtype)).float()
    if name == "fp32_forward_recompute_backward":
        return reference_controller_projection_recompute(x, weight, bias)
    if name == "fp32_forward_triton_backward":
        return reference_forward_triton_backward(x, weight, bias)
    raise ValueError(name)


def timed(callable_) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples: list[float]) -> dict:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def run_numerics(shape, seed: int) -> list[dict]:
    batch, length, width = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x_source = torch.randn(
        batch, length, width, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    direction = torch.randn(2, width + 1, generator=generator, device="cuda")
    direction = F.normalize(direction, dim=1)
    cotangent = torch.randn(
        batch, length, 2, generator=generator, device="cuda", dtype=torch.float32
    )

    saved = {}
    rows = []
    names = (
        "fp32_reference",
        "triton_tree_fp32",
        "cublas_bf16_fp32_accum",
        "fp32_forward_recompute_backward",
        "fp32_forward_triton_backward",
    )
    for name in names:
        x = x_source.detach().clone().requires_grad_(True)
        weight = direction[:, :-1].detach().clone().requires_grad_(True)
        bias = direction[:, -1].detach().clone().requires_grad_(True)
        output = projection(name, x, weight, bias)
        (output.float() * cotangent).sum().backward()
        values = {
            "output": output.detach(),
            "grad_x": x.grad.detach(),
            "grad_weight": weight.grad.detach(),
            "grad_bias": bias.grad.detach(),
        }
        if name == "fp32_reference":
            saved = values
            comparisons = {key: {"max_abs": 0.0, "relative_l2": 0.0,
                                 "cosine_similarity": 1.0} for key in values}
        else:
            comparisons = {key: metric(saved[key], value)
                           for key, value in values.items()}
        rows.append({"candidate": name, "comparisons_to_fp32": comparisons})
        del x, weight, bias, output, values
    return rows


def run_timings(shape, seed: int, rounds: int, warmup: int,
                repetitions: int) -> list[dict]:
    batch, length, width = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        batch, length, width, generator=generator, device="cuda",
        dtype=torch.bfloat16, requires_grad=True,
    )
    direction = F.normalize(
        torch.randn(2, width + 1, generator=generator, device="cuda"), dim=1
    )
    weight = direction[:, :-1].detach().clone().requires_grad_(True)
    bias = direction[:, -1].detach().clone().requires_grad_(True)
    cotangent = torch.randn(
        batch, length, 2, generator=generator, device="cuda", dtype=torch.float32
    )
    names = (
        "fp32_reference",
        "triton_tree_fp32",
        "cublas_bf16_fp32_accum",
        "fp32_forward_recompute_backward",
        "fp32_forward_triton_backward",
    )

    def forward(name):
        return projection(name, x, weight, bias)

    def forward_backward(name):
        x.grad = weight.grad = bias.grad = None
        output = forward(name)
        (output.float() * cotangent).sum().backward()

    for name in names:
        forward_backward(name)
    raw = {name: {"forward": [], "backward": [], "forward_backward": []}
           for name in names}
    for round_index in range(rounds):
        order = list(names)
        if round_index % 2:
            order.reverse()
        for name in order:
            for _ in range(warmup):
                forward(name)
            raw[name]["forward"].extend(
                timed(lambda n=name: forward(n)) for _ in range(repetitions)
            )
            backward_samples = []
            for index in range(warmup + repetitions):
                x.grad = weight.grad = bias.grad = None
                output = forward(name)
                value = timed(
                    lambda o=output: (o.float() * cotangent).sum().backward()
                )
                if index >= warmup:
                    backward_samples.append(value)
            raw[name]["backward"].extend(backward_samples)
            for _ in range(warmup):
                forward_backward(name)
            raw[name]["forward_backward"].extend(
                timed(lambda n=name: forward_backward(n))
                for _ in range(repetitions)
            )

    rows = []
    for name in names:
        x.grad = weight.grad = bias.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        forward_backward(name)
        torch.cuda.synchronize()
        rows.append({
            "candidate": name,
            "forward": summarize(raw[name]["forward"]),
            "backward": summarize(raw[name]["backward"]),
            "forward_backward": summarize(raw[name]["forward_backward"]),
            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - allocated,
            "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - reserved,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=91827)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "protocol": {
            "scope": "controller-only normalized D-to-2 projection",
            "reference": "FP32 x cast plus FP32 cuBLAS linear",
            "input_dtype": "BF16",
            "timing": "two counterbalanced rounds; compile excluded",
            "note": "formal bounded-radial/controller nonlinearities are unchanged in mixer tests",
        },
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "shapes": [],
    }
    for index, shape in enumerate(args.shape):
        print(f"shape B={shape[0]} L={shape[1]} D={shape[2]}", flush=True)
        result["shapes"].append({
            "shape": {"batch": shape[0], "length": shape[1], "width": shape[2]},
            "numerics": run_numerics(shape, args.seed + index),
            "timings": run_timings(
                shape, args.seed + index, args.rounds, args.warmup, args.repetitions
            ),
        })
        atomic_json(args.output, result)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
