"""Calibrate fused RG-LRU training candidates on fixed H800 shapes.

This measures the complete recurrent mixer after its two official
block-diagonal projections.  It is an implementation-selection experiment,
not a public baseline comparison.  Every candidate reuses the same module,
input, loss and output gradient.  Forward/reverse candidate order and all raw
samples are recorded so the selected dispatch can be audited later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import RGLRUMixer, set_seed


DEFAULT_CANDIDATES = (
    "legacy_serial:b64",
    "legacy_chunk16:b64",
    "legacy_chunk32:b64",
    "legacy_chunk64:b64",
    "fused_serial:b32",
    "fused_serial:b64",
    "fused_serial:b128",
    "fused_serial:b256",
    "fused_chunk8:b32",
    "fused_chunk8:b64",
    "fused_chunk8:b128",
    "fused_chunk8:b256",
    "fused_chunk16:b32",
    "fused_chunk16:b64",
    "fused_chunk16:b128",
    "fused_chunk16:b256",
    "fused_chunk32:b32",
    "fused_chunk32:b64",
    "fused_chunk32:b128",
    "fused_chunk32:b256",
    "fused_chunk64:b32",
    "fused_chunk64:b64",
    "fused_chunk64:b128",
    "fused_chunk64:b256",
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def parse_shape(value: str) -> tuple[int, int, int]:
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("shape must be BxLxD")
    return tuple(int(piece) for piece in pieces)  # type: ignore[return-value]


def configure(mixer: RGLRUMixer, candidate: str) -> None:
    implementation, block_text = candidate.split(":")
    block = int(block_text.removeprefix("b"))
    if implementation == "legacy_serial":
        backend = "triton"
    elif implementation.startswith("legacy_chunk"):
        backend = implementation.removeprefix("legacy_")
    else:
        backend = implementation
    mixer.set_scan_backend(backend)
    mixer.set_fused_training_block_size(block)


def synchronize_time(callable_) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "p20_ms": ordered[max(0, int(0.2 * (len(ordered) - 1)))],
        "p80_ms": ordered[min(len(ordered) - 1, int(0.8 * (len(ordered) - 1)))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=290831)
    args = parser.parse_args()
    candidates = tuple(args.candidate) if args.candidate else DEFAULT_CANDIDATES
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = []
    for shape_index, (batch, length, width) in enumerate(args.shape):
        if width % 16:
            raise ValueError("width must be divisible by 16 official gate blocks")
        print(f"shape B={batch} L={length} D={width}", flush=True)
        mixer = RGLRUMixer(width, 16).cuda().train()
        x = torch.randn(
            batch, length, width, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        gradient = torch.randn_like(x)

        def forward(candidate: str):
            configure(mixer, candidate)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = mixer(x)
            return output

        def forward_backward(candidate: str):
            mixer.zero_grad(set_to_none=True)
            x.grad = None
            configure(mixer, candidate)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = mixer(x)
            (output.float() * gradient.float()).sum().backward()

        # Compile every path before timed interleaving. Compilation time is not
        # execution time and is deliberately excluded, while failures remain rows.
        viable, failures = [], {}
        for candidate in candidates:
            try:
                forward_backward(candidate)
                torch.cuda.synchronize()
                viable.append(candidate)
                print(f"compiled {candidate}", flush=True)
            except Exception as error:
                failures[candidate] = f"{type(error).__name__}: {error}"
                print(f"failed {candidate}: {failures[candidate]}", flush=True)
                torch.cuda.empty_cache()

        raw = {candidate: {"forward": [], "forward_backward": [], "rounds": []}
               for candidate in viable}
        for round_index in range(args.rounds):
            order = list(viable)
            if (round_index + shape_index) % 2:
                order.reverse()
            for candidate in order:
                for _ in range(args.warmup):
                    forward(candidate)
                forward_samples = [
                    synchronize_time(lambda c=candidate: forward(c))
                    for _ in range(args.repetitions)
                ]
                for _ in range(args.warmup):
                    forward_backward(candidate)
                backward_samples = [
                    synchronize_time(lambda c=candidate: forward_backward(c))
                    for _ in range(args.repetitions)
                ]
                raw[candidate]["forward"].extend(forward_samples)
                raw[candidate]["forward_backward"].extend(backward_samples)
                raw[candidate]["rounds"].append({
                    "round": round_index,
                    "order_index": order.index(candidate),
                    "forward_ms": forward_samples,
                    "forward_backward_ms": backward_samples,
                })
                print(
                    candidate,
                    f"fwd={statistics.median(forward_samples):.3f}ms",
                    f"f+b={statistics.median(backward_samples):.3f}ms",
                    flush=True,
                )

        for candidate in viable:
            configure(mixer, candidate)
            mixer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            forward_backward(candidate)
            torch.cuda.synchronize()
            peak_delta = torch.cuda.max_memory_allocated() - baseline
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "candidate": candidate,
                "forward": summarize(raw[candidate]["forward"]),
                "forward_backward": summarize(raw[candidate]["forward_backward"]),
                "rounds": raw[candidate]["rounds"],
                "peak_allocated_delta_bytes": peak_delta,
            })
        for candidate, failure in failures.items():
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "candidate": candidate,
                "status": "FAILED",
                "error": failure,
            })
        del gradient, x, mixer
        torch.cuda.empty_cache()
        atomic_json(args.output, {
            "schema_version": 1,
            "protocol": {
                "scope": "complete RG-LRU mixer including two official block-diagonal gate projections",
                "dtype": "BF16 autocast with FP32 recurrence accumulation",
                "candidate_class": "our implementation calibration; not public-baseline ranking",
                "order": "alternating forward/reverse candidate order",
                "rounds": args.rounds,
                "warmup_per_round": args.warmup,
                "timed_repetitions_per_round": args.repetitions,
                "compile_excluded": True,
                "loss": "fixed dense output cotangent",
                "input_gradient": True,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "candidates": candidates,
            "rows": rows,
        })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
