"""Pool raw microbenchmark samples from counterbalanced execution orders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        raise ValueError("at least two counterbalanced runs are required")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.run]
    orders = {document.get("model_order") for document in documents}
    if orders != {"samu,rglru", "rglru,samu"}:
        raise ValueError(f"expected both execution orders, found {sorted(orders)}")
    by_id: dict[str, list[dict]] = {}
    for document in documents:
        if not document.get("equation_audit", {}).get("passed"):
            raise ValueError("refusing to merge a run that failed correctness")
        for row in document["rows"]:
            by_id.setdefault(row["id"], []).append(row)
    merged = []
    for identifier, source_rows in by_id.items():
        if len(source_rows) != len(documents):
            raise ValueError(f"{identifier} is missing from a run")
        if any(row.get("status") != "measured" for row in source_rows):
            raise ValueError(f"{identifier} contains a failed measurement")
        values = [sample for row in source_rows for sample in row["raw_samples_ms"]]
        row = {key: value for key, value in source_rows[0].items()
               if key not in {"median_ms", "p10_ms", "p90_ms", "p95_ms", "min_ms",
                              "max_ms", "mean_ms", "std_ms", "samples",
                              "raw_samples_ms", "tokens_per_second",
                              "logical_effective_gbps"}}
        median = statistics.median(values)
        tokens = row["batch"] if row["workload"] == "decode" else row["batch"] * row["length"]
        logical_bytes = row.get("logical_bytes_lower_bound")
        row.update({
            "median_ms": median,
            "p10_ms": percentile(values, .10),
            "p90_ms": percentile(values, .90),
            "p95_ms": percentile(values, .95),
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.pstdev(values),
            "samples": len(values),
            "raw_samples_ms": values,
            "tokens_per_second": tokens * 1000.0 / median,
            "logical_effective_gbps": (
                logical_bytes / median / 1e6 if logical_bytes else None
            ),
            "counterbalanced_source_medians_ms": [source["median_ms"] for source in source_rows],
        })
        merged.append(row)
    merged.sort(key=lambda row: (
        row.get("workload", ""), row.get("batch", 0), row.get("length", 0),
        row.get("track", ""), row.get("model", ""), row.get("backend", ""),
    ))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 3,
        "generated_from_raw": True,
        "title": "Order-balanced equal-kernel SAMU vs official-equation RG-LRU",
        "model_orders": sorted(orders),
        "provenance": [str(path.resolve()) for path in args.run],
        "environment": documents[0]["environment"],
        "gpu_stabilization": [document.get("gpu_stabilization") for document in documents],
        "equation_audit": documents[0]["equation_audit"],
        "rows": merged,
        "measured_count": len(merged),
        "failed_count": 0,
    }
    atomic_json(output / "summary.json", summary)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_files = set()
    for row in merged:
        filename = f"{row['id']}.json"
        expected_files.add(filename)
        atomic_json(raw_dir / filename, row)
    for path in raw_dir.glob("*.json"):
        if path.name not in expected_files:
            path.unlink()
    atomic_json(output / "environment.json", summary["environment"])
    atomic_json(output / "correctness.json", summary["equation_audit"])
    atomic_json(output / "run_manifest.json", {
        "status": "complete",
        "public_scope": "SAMU versus official-equation RG-LRU only",
        "model_orders": sorted(orders),
        "source_runs": summary["provenance"],
        "result_row_count": len(merged),
        "aggregation": "pooled raw timing samples; statistics recomputed",
    })
    keys = ["id", "status", "track", "model", "backend", "workload", "batch",
            "length", "d_model", "modes", "chunk_size", "median_ms", "p10_ms",
            "p95_ms", "tokens_per_second", "logical_bytes_lower_bound",
            "logical_effective_gbps", "expected_cuda_launches", "state_bytes_per_batch",
            "parameter_count", "implementation_class", "source_commit", "error"]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


if __name__ == "__main__":
    main()
