"""Full-model trajectory A/B for SAMU decode mappings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from run_small_model_study import (
    CapturedDecoder,
    StudyConfig,
    load_character_data,
    load_trained,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def set_decode_backend(model, backend: str) -> None:
    for block in model.blocks:
        block.mixer.set_decode_backend(backend)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    data = load_character_data(args.data)
    config = StudyConfig(vocab_size=len(data["alphabet"]))
    rows = []
    for backend in ("fused", "packed", "blocked16", "blocked32", "blocked64"):
        model = load_trained(config, "samu", args.seed, args.checkpoint_root)
        set_decode_backend(model, backend)
        for batch in (1, 16, 64, 256):
            print(f"backend={backend} batch={batch}", flush=True)
            decoder = CapturedDecoder(model, batch)
            samples = []
            for _ in range(5):
                decoder.reset()
                samples.append(decoder.trajectory_ms(512))
            rows.append({
                "backend": backend,
                "batch_size": batch,
                "generated_tokens": 512,
                "samples_ms": samples,
                "median_trajectory_ms": sorted(samples)[len(samples) // 2],
                "median_tokens_per_second": sorted(
                    batch * 512 / (sample / 1000.0) for sample in samples
                )[len(samples) // 2],
            })
            del decoder
        del model
        torch.cuda.empty_cache()
    atomic_json(args.output, {"schema_version": 1, "rows": rows})
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
