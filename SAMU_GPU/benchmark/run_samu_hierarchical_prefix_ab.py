"""A/B SAMU's serial chunk prefix and complex hierarchical prefix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import SAMUMixer, set_seed


def timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "minimum_ms": min(samples), "maximum_ms": max(samples),
            "mean_ms": statistics.fmean(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=75002)
    args = parser.parse_args()
    set_seed(args.seed)
    shapes = ((4, 2048, 2048), (1, 8192, 2560), (1, 32768, 1024))
    backends = ("chunk16", "hierarchical_chunk16")
    rows = []
    for batch, length, width in shapes:
        x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn_like(x)
        reference = SAMUMixer(width).cuda().train()
        modules = {}
        for backend in backends:
            module = SAMUMixer(width).cuda().train()
            module.load_state_dict(reference.state_dict())
            module.set_scan_backend(backend)
            module.set_controller_projection_dtype("fp32")
            modules[backend] = module

        def forward(name):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return modules[name](x)[0]

        def forward_backward(name):
            modules[name].zero_grad(set_to_none=True)
            output = forward(name)
            (output.float() * gradient.float()).sum().backward()

        compile_ms = {name: timed(lambda n=name: forward_backward(n)) for name in backends}
        raw = {name: {"forward": [], "forward_backward": [], "rounds": []}
               for name in backends}
        for round_index in range(args.rounds):
            order = backends if round_index % 2 == 0 else tuple(reversed(backends))
            for order_index, name in enumerate(order):
                for _ in range(2):
                    forward_backward(name)
                fwd = [timed(lambda n=name: forward(n)) for _ in range(args.repetitions)]
                fb = [timed(lambda n=name: forward_backward(n))
                      for _ in range(args.repetitions)]
                raw[name]["forward"].extend(fwd)
                raw[name]["forward_backward"].extend(fb)
                raw[name]["rounds"].append({"round": round_index,
                    "order_index": order_index, "forward_ms": fwd,
                    "forward_backward_ms": fb})
                print(batch, length, width, name,
                      statistics.median(fwd), statistics.median(fb), flush=True)
        for name in backends:
            modules[name].zero_grad(set_to_none=True)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            allocated, reserved = (torch.cuda.memory_allocated(),
                                   torch.cuda.memory_reserved())
            forward_backward(name); torch.cuda.synchronize()
            rows.append({"shape": {"batch": batch, "length": length,
                                   "width": width}, "backend": name,
                "first_compile_and_run_ms": compile_ms[name],
                "forward": summarize(raw[name]["forward"]),
                "forward_backward": summarize(raw[name]["forward_backward"]),
                "rounds": raw[name]["rounds"],
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - allocated,
                "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - reserved})
        del x, gradient, reference, modules
        torch.cuda.empty_cache()
    value = {"schema_version": 1,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "protocol": {"scope": "complete SAMU mixer with exact K=16 chunk scan",
            "comparison": "only the prefix over complex chunk summaries changes",
            "serial": "one program per batch/mode tile loops over C chunks",
            "hierarchical": "complex affine Hillis-Steele prefix with log2(C) stages",
            "compile_excluded_from_steady_state": True, "order": "AB/BA"},
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
