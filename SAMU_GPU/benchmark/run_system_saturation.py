"""Large-batch saturation stress for the shared 1B decoder shell.

This is deliberately separate from the Griffin Section-5-style full-trajectory
experiment.  It uses short probes only to locate the H800 saturation region;
these rows must not be presented as full 512--4096 token trajectories.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from run_griffin_section5 import (
    atomic_json,
    build_model,
    correctness,
    environment,
    parse_ints,
    stabilize_gpu,
    throughput_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--model-order", default="samu,rglru")
    parser.add_argument("--batch-candidates", type=parse_ints,
                        default=parse_ints("128,256,512,1024,2048,4096"))
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--expansion", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    kinds = [item for item in args.model_order.split(",") if item]
    if len(kinds) != 2 or set(kinds) != {"samu", "rglru"}:
        raise ValueError("--model-order must contain samu,rglru exactly once each")
    if args.probe_steps <= 0 or args.repeats < 2:
        raise ValueError("stress probes require positive steps and at least two repeats")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    audit = correctness(args.rglru_source, args.width, args.seed)
    warmup = stabilize_gpu()
    searches: dict[str, list[dict]] = {}

    for kind in kinds:
        print(f"building {kind} saturation model", flush=True)
        model = build_model(
            kind, args.width, args.depth, args.expansion, args.vocab,
            args.rglru_source, args.seed,
        )
        rows = []
        for batch in args.batch_candidates:
            print(f"{kind}: short saturation probe batch={batch}", flush=True)
            values = []
            try:
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.repeats):
                    row = throughput_probe(
                        model, batch, args.probe_steps,
                        args.seed + repeat * 104729,
                    )
                    values.append(row["latency_ms"])
                latency = statistics.median(values)
                rows.append({
                    "batch": batch,
                    "probe_steps": args.probe_steps,
                    "latency_ms": latency,
                    "raw_latency_ms": values,
                    "tokens_per_second": batch * args.probe_steps * 1000.0 / latency,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "status": "measured",
                    "measurement_class": "short_saturation_probe_not_full_trajectory",
                })
            except torch.cuda.OutOfMemoryError:
                rows.append({
                    "batch": batch, "probe_steps": args.probe_steps,
                    "status": "oom",
                    "measurement_class": "short_saturation_probe_not_full_trajectory",
                })
                torch.cuda.empty_cache()
            except RuntimeError as error:
                if "invalid argument" not in str(error).lower():
                    raise
                rows.append({
                    "batch": batch, "probe_steps": args.probe_steps,
                    "status": "unsupported_launch_grid", "reason": str(error),
                    "measurement_class": "short_saturation_probe_not_full_trajectory",
                })
                torch.cuda.empty_cache()
        searches[kind] = rows
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "status": "complete",
        "scope": "large-batch short-probe saturation stress; not a full-trajectory throughput result",
        "environment": environment(),
        "correctness": audit,
        "gpu_stabilization": warmup,
        "configuration": vars(args) | {
            "output": str(args.output),
            "rglru_source": str(args.rglru_source),
        },
        "throughput_batch_search": searches,
    }
    atomic_json(args.output.resolve(), result)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
