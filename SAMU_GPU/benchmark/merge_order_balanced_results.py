"""Combine counterbalanced Section-5 runs from their raw CUDA samples.

The two inputs must contain the same semantic workloads but opposite model and
prompt orders.  Aggregating samples, rather than averaging reported medians,
keeps the published percentiles auditable and removes a fixed-order thermal or
clock bias from the comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_griffin_section5 import atomic_json, percentile


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model"], row["workload"], row["batch"], row["prompt_length"],
        row["decode_length"],
    )


def combine_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            if row["workload"] != "maximum_throughput":
                grouped[row_key(row)].append(row)
    combined = []
    for key, group in grouped.items():
        if len(group) != len(runs):
            raise ValueError(
                f"semantic workload {key} appears in {len(group)}/{len(runs)} runs; "
                "counterbalanced runs must measure identical full-trajectory candidates"
            )
        template = dict(group[0])
        samples = [value for row in group for value in row["raw_samples_ms"]]
        median = statistics.median(samples)
        template.update({
            "median_ms": median,
            "p10_ms": percentile(samples, .10),
            "p95_ms": percentile(samples, .95),
            "raw_samples_ms": samples,
            "samples": len(samples),
            "tokens_per_second": (
                template["batch"] * template["decode_length"] * 1000.0 / median
            ),
            "counterbalanced_runs": len(group),
        })
        prefill = [
            row.get("prompt_prefill_ms_excluded_from_decode") for row in group
            if row.get("prompt_prefill_ms_excluded_from_decode") is not None
        ]
        if prefill:
            template["prompt_prefill_ms_excluded_from_decode"] = statistics.median(prefill)
            template["prompt_prefill_raw_ms"] = prefill
        combined.append(template)

    # Re-select the best fully measured candidate separately at each generation
    # length; short probe results never become a published throughput winner.
    for model in ("samu", "rglru"):
        lengths = sorted({
            row["decode_length"] for row in combined
            if row["model"] == model
            and row["workload"] == "maximum_throughput_candidate"
        })
        for length in lengths:
            candidates = [
                row for row in combined
                if row["model"] == model
                and row["workload"] == "maximum_throughput_candidate"
                and row["decode_length"] == length
            ]
            if candidates:
                winner = max(candidates, key=lambda row: row["tokens_per_second"])
                combined.append({**winner, "workload": "maximum_throughput"})
    return sorted(combined, key=row_key)


def combine_search(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for model in ("samu", "rglru"):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            for row in run["throughput_batch_search"][model]:
                grouped[row["batch"]].append(row)
        result[model] = []
        for batch, rows in sorted(grouped.items()):
            measured = [row for row in rows if row["status"] == "measured"]
            if measured:
                latencies = [row["latency_ms"] for row in measured]
                median = statistics.median(latencies)
                steps = measured[0]["probe_steps"]
                result[model].append({
                    "batch": batch,
                    "probe_steps": steps,
                    "latency_ms": median,
                    "raw_latency_ms": latencies,
                    "tokens_per_second": batch * steps * 1000.0 / median,
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
                    "status": "measured",
                    "counterbalanced_runs": len(measured),
                })
            else:
                result[model].append(dict(rows[-1]))
    return result


def merge_stress(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {
        "samu": defaultdict(list), "rglru": defaultdict(list),
    }
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "complete" or not run.get("correctness", {}).get("passed"):
            raise ValueError(f"stress input is incomplete or failed correctness: {path}")
        for model in grouped:
            for row in run.get("throughput_batch_search", {}).get(model, []):
                grouped[model][row["batch"]].append({**row, "source_file": str(path)})
    merged: dict[str, list[dict[str, Any]]] = {"samu": [], "rglru": []}
    for model, batches in grouped.items():
        for batch, rows in sorted(batches.items()):
            measured = [row for row in rows if row.get("status") == "measured"]
            if not measured:
                merged[model].append(dict(rows[-1]))
                continue
            raw = [
                value for row in measured
                for value in row.get("raw_latency_ms", [row["latency_ms"]])
            ]
            latency = statistics.median(raw)
            steps = measured[0]["probe_steps"]
            template = dict(measured[0])
            template.update({
                "latency_ms": latency,
                "raw_latency_ms": raw,
                "tokens_per_second": batch * steps * 1000.0 / latency,
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
                "counterbalanced_runs": len(measured),
                "source_files": [row["source_file"] for row in measured],
            })
            template.pop("source_file", None)
            merged[model].append(template)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--stress", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        raise ValueError("at least two counterbalanced --run inputs are required")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.run]
    if any(run.get("status") != "complete" for run in runs):
        raise ValueError("all counterbalanced inputs must be complete")
    if any(not run.get("correctness", {}).get("passed") for run in runs):
        raise ValueError("all counterbalanced inputs must pass fail-closed correctness")
    model_orders = {
        tuple(item for item in run["configuration"]["model_order"].split(",") if item)
        for run in runs
    }
    prompt_orders = {tuple(run["configuration"]["prompt_lengths"]) for run in runs}
    if model_orders != {("samu", "rglru"), ("rglru", "samu")}:
        raise ValueError(f"model execution order is not counterbalanced: {model_orders}")
    if prompt_orders != {(0, 4096), (4096, 0)}:
        raise ValueError(f"prompt execution order is not counterbalanced: {prompt_orders}")
    result = {
        **runs[0],
        "rows": combine_rows(runs),
        "throughput_batch_search": combine_search(runs),
        "throughput_saturation_stress": merge_stress(args.stress),
        "configuration": {"counterbalanced_runs": [run["configuration"] for run in runs]},
        "provenance": {
            "aggregation": "raw CUDA samples combined across counterbalanced model and prompt orders; medians and percentiles recomputed",
            "run_inputs": [str(path) for path in args.run],
            "stress_inputs": [str(path) for path in args.stress],
        },
    }
    atomic_json(args.output.resolve(), result)
    print(f"wrote {args.output.resolve()} with {len(result['rows'])} rows")


if __name__ == "__main__":
    main()
