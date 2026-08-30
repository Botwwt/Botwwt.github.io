"""Reproduce fattorib/hawk-pytorch from its pinned, unmodified GitHub source.

The benchmark harness is ours; every Fattori operator and gate projection is
imported directly from the pinned checkout.  This script intentionally does
not copy, patch or reimplement the public baseline.
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


def summary(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fattori-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--ours-chunk", type=int, default=32)
    parser.add_argument("--ours-block", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    repo = args.fattori_repo.resolve()
    sys.path.insert(0, str(repo))
    # These imports execute the public checkout itself.
    from hawk.external import BlockDiagonalLinear  # type: ignore
    from hawk.scan_fused import fused_linear_scan  # type: ignore
    from canonical_rg_lru_reference import canonical_rg_lru_from_logits
    from triton_fused_rglru_training import fused_rglru_scan

    torch.manual_seed(68001)
    batch, length, width = args.batch, args.length, args.width
    if width % 64:
        raise ValueError("Fattori's original kernel requires width divisible by 64")
    source = repo / "hawk" / "scan_fused.py"
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    git_diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--", "hawk/scan_fused.py", "hawk/hawk.py", "hawk/external.py"],
        text=True,
    )

    # Semantic parity for the exact restricted contract supported publicly:
    # zero initial state, no reset token, L>=2 and D multiple of 64.
    correctness_length = min(max(17, length), 65)
    test_x = (torch.randn(2, correctness_length, width, device="cuda") * 0.2).bfloat16().requires_grad_(True)
    test_gx = (torch.randn_like(test_x) * 0.5).requires_grad_(True)
    test_ga = (torch.randn_like(test_x) * 0.5).requires_grad_(True)
    test_a = (torch.randn(width, device="cuda") * 0.2 + 1.0).requires_grad_(True)
    public_output = fused_linear_scan(test_x, test_gx, test_ga, test_a)
    reference_inputs = [value.detach().clone().requires_grad_(True)
                        for value in (test_x, test_gx, test_ga, test_a)]
    positions = torch.ones((2, correctness_length), device="cuda", dtype=torch.int32)
    reference_output_fp32, _ = canonical_rg_lru_from_logits(
        *reference_inputs, segment_pos=positions,
        production_dtype=None,
    )
    # Fattori stores BF16 states but retains its running carry and pointwise
    # recurrence terms in FP32.  Model that exact public precision contract.
    reference_output = reference_output_fp32.bfloat16()
    cotangent = torch.randn_like(public_output)
    (public_output.float() * cotangent.float()).sum().backward()
    (reference_output.float() * cotangent.float()).sum().backward()
    public_inputs = (test_x, test_gx, test_ga, test_a)
    gradient_errors = {
        name: float((actual.grad.float() - expected.grad.float()).abs().max())
        for name, actual, expected in zip(
            ("x", "input_gate_logits", "recurrence_gate_logits", "a_param"),
            public_inputs, reference_inputs,
        )
    }
    gradient_relative = {
        name: value / max(1.0, float(expected.grad.float().abs().max()))
        for (name, value), expected in zip(gradient_errors.items(), reference_inputs)
    }
    correctness = {
        "contract": "zero initial state, no resets, length>=2, width multiple of 64",
        "canonical_reset_semantics": "NOT SUPPORTED by the public kernel",
        "output_max_abs": float((public_output.float() - reference_output.float()).abs().max()),
        "gradient_errors": gradient_errors,
        "gradient_relative_errors": gradient_relative,
        "passed_restricted_contract": (
            float((public_output.float() - reference_output.float()).abs().max()) <= 0.02
            and max(gradient_relative.values()) <= 0.03
        ),
    }
    del test_x, test_gx, test_ga, test_a, public_output, reference_output
    torch.cuda.empty_cache()

    input_gate = BlockDiagonalLinear(width=width, num_blocks=16).cuda()
    recurrence_gate = BlockDiagonalLinear(width=width, num_blocks=16).cuda()
    a_param = torch.nn.Parameter(torch.randn(width, device="cuda") * 0.2 + 1.0)
    x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
    output_gradient = torch.randn_like(x)

    def zero_grad():
        input_gate.zero_grad(set_to_none=True)
        recurrence_gate.zero_grad(set_to_none=True)
        a_param.grad = None

    def public_forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gx = input_gate(x)
            ga = recurrence_gate(x)
            return fused_linear_scan(x, gx, ga, a_param)

    def ours_forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gx = input_gate(x)
            ga = recurrence_gate(x)
            output, _ = fused_rglru_scan(
                x, gx, ga, a_param, chunk_size=args.ours_chunk,
                block_size=args.ours_block, reset_first=False,
            )
            return output

    functions = {"fattori_original": public_forward, "ours_restricted": ours_forward}

    def forward_backward(name):
        zero_grad()
        output = functions[name]()
        (output.float() * output_gradient.float()).sum().backward()

    failures = {}
    viable = []
    for name in functions:
        try:
            forward_backward(name)
            torch.cuda.synchronize()
            viable.append(name)
            print(f"compiled {name}", flush=True)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"failed {name}: {failures[name]}", flush=True)
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
                forward_backward(name)
            fb_samples = [
                timed(lambda n=name: forward_backward(n))
                for _ in range(args.repetitions)
            ]
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
        zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        forward_backward(name)
        torch.cuda.synchronize()
        rows.append({
            "implementation": name,
            "forward": summary(raw[name]["forward"]),
            "forward_backward": summary(raw[name]["forward_backward"]),
            "rounds": raw[name]["rounds"],
            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - baseline,
        })
    for name, failure in failures.items():
        rows.append({"implementation": name, "status": "FAILED TO REPRODUCE", "error": failure})
    report = {
        "schema_version": 1,
        "repository": "https://github.com/fattorib/hawk-pytorch",
        "commit": commit,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "source_diff_empty": not bool(git_diff),
        "environment": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "shape": {"batch": batch, "length": length, "width": width},
        "protocol": {
            "scope": "official block-diagonal gate projections + activation/write/recurrence",
            "dtype": "BF16 autocast, FP32 recurrent carry",
            "order": "AB/BA",
            "compile_excluded": True,
            "semantic_scope": "restricted no-reset recurrence; secondary comparison only",
            "primary_canonical_ranking": "NOT COMPARABLE because public Fattori source lacks official reset/write normalization, arbitrary segment resets and initial state",
        },
        "correctness": correctness,
        "rows": rows,
    }
    atomic_json(args.output, report)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
