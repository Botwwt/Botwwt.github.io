"""A/B exact RG-LRU training scans on the Griffin length/scale matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from run_griffin_training_matrix import SCALES
from run_small_model_study import (
    StudyConfig,
    benchmark_complete_training_steps,
    build_model,
    load_character_data,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def set_backend(model, backend: str) -> None:
    for block in model.blocks:
        block.mixer.set_scan_backend(backend)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    vocab_size = len(load_character_data(args.data)["alphabet"])
    rows = []
    for index, (scale, dimensions) in enumerate(SCALES):
        config = StudyConfig(vocab_size=vocab_size, **dimensions)
        backends = ("triton", "chunk16", "chunk32")
        order = backends if index % 2 == 0 else tuple(reversed(backends))
        for backend in order:
            print(f"scale={scale} backend={backend}", flush=True)
            model, _ = build_model(config, "rglru", args.seed)
            model = model.cuda()
            set_backend(model, backend)
            measurements = benchmark_complete_training_steps(
                model, config, lengths=(256, 2048, 4096, 8192),
                tokens_per_step=8192,
            )
            rows.append({
                "scale": scale,
                "backend": backend,
                "config": asdict(config),
                "parameters": sum(p.numel() for p in model.parameters()),
                "measurements": measurements,
            })
            del model
            torch.cuda.empty_cache()
    atomic_json(args.output, {
        "schema_version": 1,
        "comparison": (
            "official-equation state-stationary serial scan versus exact "
            "differentiable 16- and 32-token real affine chunk scans"
        ),
        "rows": rows,
    })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
