"""Run Lingua/Hippogriff's pinned original accelerated-scan call paths.

This is a scan-only protocol over pre-materialized affine coefficients.  The
Lingua wrapper and accelerated-scan extension are imported directly from their
GitHub checkouts without source changes.  Hippogriff calls the same extension
directly, so its row is a call-path reproduction, not a distinct scan kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(callable_) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
    }


def eager_scan(gates, tokens):
    state = torch.zeros_like(tokens[:, :, 0], dtype=torch.float32)
    outputs = []
    for time_index in range(tokens.shape[-1]):
        state = gates[:, :, time_index].float() * state + tokens[:, :, time_index].float()
        outputs.append(state.to(tokens.dtype))
    return torch.stack(outputs, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingua-repo", type=Path, required=True)
    parser.add_argument("--hippogriff-repo", type=Path, required=True)
    parser.add_argument("--accelerated-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    lingua = args.lingua_repo.resolve()
    hippogriff = args.hippogriff_repo.resolve()
    accelerated = args.accelerated_repo.resolve()
    sys.path[:0] = [str(lingua), str(accelerated)]
    # Original public imports.  Do not replace these with local kernels.
    from accelerated_scan.warp import scan as hippogriff_scan  # type: ignore
    from apps.fastRNN.component.compilable_scan import scan as lingua_scan  # type: ignore
    from triton_training_scan import real_chunk_scan, real_scan

    torch.manual_seed(69001)
    batch, length, width = args.batch, args.length, args.width
    # The original CUDA extension explicitly requires power-of-two L in
    # [32,65536].  Exercise its valid public contract without patching it.
    correctness_gates = torch.sigmoid(torch.randn(2, 256, 64, device="cuda")).bfloat16()
    correctness_tokens = (torch.randn_like(correctness_gates) * 0.1).bfloat16()
    correctness_gates[:, :, 0] = 0  # reset semantics are encoded in the affine coefficient
    reference_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (correctness_gates, correctness_tokens)
    ]
    reference = eager_scan(*reference_inputs)
    correctness = {}
    for name, operation in (
        ("lingua_original_wrapper", lingua_scan),
        ("hippogriff_accelerated_scan", hippogriff_scan),
    ):
        inputs = [value.detach().clone().requires_grad_(True)
                  for value in (correctness_gates, correctness_tokens)]
        output = operation(*inputs)
        cotangent = torch.randn_like(output)
        (output.float() * cotangent.float()).sum().backward()
        reference_local_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (correctness_gates, correctness_tokens)
        ]
        reference_local = eager_scan(*reference_local_inputs)
        (reference_local.float() * cotangent.float()).sum().backward()
        grad_errors = [
            float((actual.grad.float() - expected.grad.float()).abs().max())
            for actual, expected in zip(inputs, reference_local_inputs)
        ]
        correctness[name] = {
            "output_max_abs": float((output.float() - reference_local.float()).abs().max()),
            "gradient_max_abs": max(grad_errors),
            "gradient_relative": max(
                error / max(1.0, float(expected.grad.float().abs().max()))
                for error, expected in zip(grad_errors, reference_local_inputs)
            ),
        }

    gates_bld = torch.sigmoid(
        torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
    ).contiguous()
    tokens_bld = (torch.randn_like(gates_bld) * 0.1).contiguous()
    gates_bdl = gates_bld.transpose(1, 2).contiguous()
    tokens_bdl = tokens_bld.transpose(1, 2).contiguous()
    gradient_bld = torch.randn_like(tokens_bld)
    gradient_bdl = gradient_bld.transpose(1, 2).contiguous()

    functions = {
        "lingua_original_wrapper": lambda: lingua_scan(gates_bdl, tokens_bdl),
        "hippogriff_accelerated_scan": lambda: hippogriff_scan(gates_bdl, tokens_bdl),
        "ours_materialized_serial": lambda: real_scan(gates_bld, tokens_bld),
        "ours_materialized_chunk16": lambda: real_chunk_scan(gates_bld, tokens_bld, 16),
        "ours_materialized_chunk32": lambda: real_chunk_scan(gates_bld, tokens_bld, 32),
    }

    def forward_backward(name):
        output = functions[name]()
        gradient = gradient_bdl if name.startswith(("lingua", "hippogriff")) else gradient_bld
        (output.float() * gradient.float()).sum().backward()

    # Inputs are constants in the timing protocol; create grad-bearing leaves
    # once and replace captured tensors by requiring-grad views.
    gates_bld.requires_grad_(True)
    tokens_bld.requires_grad_(True)
    gates_bdl.requires_grad_(True)
    tokens_bdl.requires_grad_(True)

    failures, viable = {}, []
    for name in functions:
        try:
            forward_backward(name)
            torch.cuda.synchronize()
            viable.append(name)
            print(f"compiled {name}", flush=True)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"failed {name}: {failures[name]}", flush=True)

    def clear_grads():
        for value in (gates_bld, tokens_bld, gates_bdl, tokens_bdl):
            value.grad = None

    raw = {name: {"forward": [], "forward_backward": [], "rounds": []}
           for name in viable}
    for round_index in range(args.rounds):
        order = list(viable)
        if round_index % 2:
            order.reverse()
        for order_index, name in enumerate(order):
            for _ in range(2):
                functions[name]()
            forward_samples = [timed(functions[name]) for _ in range(args.repetitions)]
            for _ in range(2):
                clear_grads(); forward_backward(name)
            fb_samples = []
            for _ in range(args.repetitions):
                clear_grads()
                fb_samples.append(timed(lambda n=name: forward_backward(n)))
            raw[name]["forward"].extend(forward_samples)
            raw[name]["forward_backward"].extend(fb_samples)
            raw[name]["rounds"].append({
                "round": round_index, "order_index": order_index,
                "forward_ms": forward_samples,
                "forward_backward_ms": fb_samples,
            })
            print(name, statistics.median(fb_samples), flush=True)
    rows = []
    for name in viable:
        clear_grads(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        forward_backward(name); torch.cuda.synchronize()
        rows.append({
            "implementation": name,
            "forward": summarize(raw[name]["forward"]),
            "forward_backward": summarize(raw[name]["forward_backward"]),
            "rounds": raw[name]["rounds"],
            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - baseline,
        })
    for name, failure in failures.items():
        rows.append({"implementation": name, "status": "FAILED TO REPRODUCE", "error": failure})

    def commit(repo):
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    wrapper = lingua / "apps/fastRNN/component/compilable_scan.py"
    extension = accelerated / "accelerated_scan/warp.py"
    report = {
        "schema_version": 1,
        "repositories": {
            "lingua": {"url": "https://github.com/facebookresearch/lingua", "commit": commit(lingua),
                       "wrapper_sha256": sha256(wrapper)},
            "hippogriff": {"url": "https://github.com/proger/hippogriff", "commit": commit(hippogriff)},
            "accelerated_scan": {"url": "https://github.com/proger/accelerated-scan", "commit": commit(accelerated),
                                 "wrapper_sha256": sha256(extension)},
        },
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "shape": {"batch": batch, "length": length, "width": width},
        "protocol": {
            "scope": "scan only; affine gates/writes pre-materialized in each implementation's native contiguous layout",
            "dtype": "BF16 input/output with implementation-defined FP32 accumulation",
            "order": "forward/reverse",
            "compile_excluded": True,
            "comparability": "scan recurrence is comparable; Lingua/Hippogriff full RG-LRU preparation is not canonical",
            "duplicate_kernel_note": "Lingua wrapper and Hippogriff both invoke proger/accelerated-scan warp CUDA kernels",
            "original_shape_constraint": "sequence length must be a power of two in [32,65536]",
        },
        "correctness": correctness,
        "rows": rows,
    }
    atomic_json(args.output, report)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
