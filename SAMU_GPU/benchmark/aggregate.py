"""Aggregate raw benchmark JSON records into CSV and website JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "id", "status", "track", "model", "backend", "workload", "matching_protocol",
    "dtype", "batch", "length", "d_model", "modes", "real_state_scalars",
    "chunk_size", "mode_tile", "median_ms", "mean_ms", "p10_ms", "p90_ms",
    "p95_ms", "std_ms", "tokens_per_second", "peak_allocated_bytes",
    "logical_bytes", "parameter_count", "inner_iterations", "samples",
    "error_type", "error", "source_commit"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.results / "raw").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(value)
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})
    payload = {
        "schema_version": 1,
        "generated_from_raw": True,
        "row_count": len(rows),
        "rows": rows,
    }
    (args.results / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregated {len(rows)} rows")


if __name__ == "__main__":
    main()
