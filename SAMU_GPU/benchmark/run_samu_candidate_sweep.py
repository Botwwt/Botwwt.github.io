"""Calibrate paper-width SAMU serial/chunk training paths on H800."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import SAMUMixer, set_seed


DEFAULT_CANDIDATES = (
    "materialized_serial:b128:fp32",
    "tiled_serial:b32:fp32",
    "tiled_serial:b64:fp32",
    "tiled_serial:b128:fp32",
    "tiled_serial:b256:fp32",
    "chunk16:b128:fp32",
    "chunk32:b128:fp32",
    "chunk64:b128:fp32",
    "chunk16:b128:bf16",
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def parse_shape(value: str):
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("shape must be BxLxD")
    return tuple(int(piece) for piece in pieces)


def configure(mixer: SAMUMixer, candidate: str) -> None:
    implementation, block_text, controller_dtype = candidate.split(":")
    block = int(block_text.removeprefix("b"))
    backend = "triton" if implementation == "materialized_serial" else implementation
    mixer.set_scan_backend(backend)
    mixer.set_tiled_training_block_size(block)
    mixer.set_controller_projection_dtype(controller_dtype)


def timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=290901)
    args = parser.parse_args()
    candidates = tuple(args.candidate) if args.candidate else DEFAULT_CANDIDATES
    set_seed(args.seed)
    rows = []
    for shape_index, (batch, length, width) in enumerate(args.shape):
        print(f"shape B={batch} L={length} D={width}", flush=True)
        mixer = SAMUMixer(width).cuda().train()
        # Nonzero amplitudes make both shared controls active during calibration.
        with torch.no_grad():
            mixer.phase_amplitude.fill_(0.7)
            mixer.radial_amplitude.fill_(0.5)
        x = torch.randn(
            batch, length, width, device="cuda", dtype=torch.bfloat16,
            requires_grad=True,
        )
        gradient = torch.randn_like(x)

        def forward(candidate):
            configure(mixer, candidate)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = mixer(x)
            return output

        def forward_backward(candidate):
            mixer.zero_grad(set_to_none=True)
            x.grad = None
            output = forward(candidate)
            (output.float() * gradient.float()).sum().backward()

        viable, failures = [], {}
        for candidate in candidates:
            try:
                forward_backward(candidate); torch.cuda.synchronize()
                viable.append(candidate)
                print(f"compiled {candidate}", flush=True)
            except Exception as error:
                failures[candidate] = f"{type(error).__name__}: {error}"
                print(f"failed {candidate}: {failures[candidate]}", flush=True)
                torch.cuda.empty_cache()
        raw = {candidate: {"forward": [], "backward": [], "forward_backward": [], "rounds": []}
               for candidate in viable}
        for round_index in range(args.rounds):
            order = list(viable)
            if (round_index + shape_index) % 2:
                order.reverse()
            for order_index, candidate in enumerate(order):
                for _ in range(args.warmup): forward(candidate)
                fwd = [timed(lambda c=candidate: forward(c))
                       for _ in range(args.repetitions)]
                backward = []
                for _ in range(args.warmup + args.repetitions):
                    mixer.zero_grad(set_to_none=True)
                    x.grad = None
                    output = forward(candidate)
                    elapsed = timed(
                        lambda o=output: (o.float() * gradient.float()).sum().backward()
                    )
                    if len(backward) >= args.warmup:
                        backward.append(elapsed)
                    else:
                        backward.append(elapsed)
                backward = backward[args.warmup:]
                for _ in range(args.warmup): forward_backward(candidate)
                fb = [timed(lambda c=candidate: forward_backward(c))
                      for _ in range(args.repetitions)]
                raw[candidate]["forward"].extend(fwd)
                raw[candidate]["backward"].extend(backward)
                raw[candidate]["forward_backward"].extend(fb)
                raw[candidate]["rounds"].append({
                    "round": round_index, "order_index": order_index,
                    "forward_ms": fwd, "backward_ms": backward,
                    "forward_backward_ms": fb,
                })
                print(candidate, statistics.median(fwd), statistics.median(backward),
                      statistics.median(fb), flush=True)
        for candidate in viable:
            configure(mixer, candidate); mixer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()
            forward_backward(candidate); torch.cuda.synchronize()
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "candidate": candidate,
                "forward": summarize(raw[candidate]["forward"]),
                "backward": summarize(raw[candidate]["backward"]),
                "forward_backward": summarize(raw[candidate]["forward_backward"]),
                "rounds": raw[candidate]["rounds"],
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - baseline,
                "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - baseline_reserved,
            })
        for candidate, failure in failures.items():
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "candidate": candidate, "status": "FAILED", "error": failure,
            })
        del mixer, x, gradient
        torch.cuda.empty_cache()
        atomic_json(args.output, {
            "schema_version": 1,
            "protocol": {
                "scope": "complete SAMU mixer: two shared controls, normalized writes and recurrence",
                "input_gradient": True,
                "dtype": "BF16 autocast, FP32 recurrent carry",
                "candidate_class": "our calibration; final data are separate",
                "order": "alternating forward/reverse",
                "compile_excluded": True,
                "materialized_serial_note": "allocates logical [B,L,M] radius and phase-derived transition tensors",
                "tiled_serial_note": "no logical global [B,L,M] transition allocation; transition reconstructed near use",
            },
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name()},
            "candidates": candidates,
            "rows": rows,
        })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
