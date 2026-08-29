"""H800 training matrix corresponding to Griffin Figure 3.

The paper holds tokens per optimizer step constant while sweeping sequence
length at three model scales.  A single H800 cannot reproduce the paper's
multi-TPU 400M/1.3B/7B jobs, so this script preserves the experimental design
with three explicitly labelled, smaller Hawk-style models.  Every timed sample
is a complete training step: forward, cross entropy, backward, gradient
clipping and AdamW update.

The architecture switch is restricted to ``SAMUMixer`` versus the official-
equation ``RGLRUMixer``.  Embedding, normalization, two recurrent input
branches, causal depthwise convolution, multiplicative join, output projection
and gated MLP are identical.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from run_small_model_study import (
    OFFICIAL_RECURRENTGEMMA_COMMIT,
    StudyConfig,
    benchmark_complete_training_steps,
    build_model,
    load_character_data,
    load_trained,
)


SCALES = (
    ("small", dict(width=256, rnn_width=384, depth=6)),
    ("medium", dict(width=384, rnn_width=512, depth=8)),
    ("large", dict(width=512, rnn_width=768, depth=12)),
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def model_row(
    config: StudyConfig,
    architecture: str,
    seed: int,
    lengths: tuple[int, ...],
) -> dict:
    model, shared = build_model(config, architecture, seed)
    model = model.cuda()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    mixer_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".mixer." in name
    )
    rows = benchmark_complete_training_steps(
        model, config, lengths=lengths, tokens_per_step=8192
    )
    del model
    torch.cuda.empty_cache()
    return {
        "architecture": architecture,
        "parameters": parameters,
        "recurrent_mixer_parameters": mixer_parameters,
        "shared_initialization": shared,
        "rows": rows,
    }


def aggregate_passes(raw_records: list[dict]) -> list[dict]:
    """Average medians from forward/reverse measurement orders."""
    aggregated = []
    for scale in ("small", "medium", "large"):
        for architecture in ("samu", "rglru"):
            candidates = [
                row for row in raw_records
                if row["scale"] == scale and row["architecture"] == architecture
            ]
            template = candidates[0]
            rows = []
            for length in (2048, 4096, 8192):
                measurements = [
                    next(item for item in candidate["rows"]
                         if item["sequence_length"] == length)
                    for candidate in candidates
                ]
                median_ms = statistics.fmean(
                    item["median_step_ms"] for item in measurements
                )
                rows.append({
                    "sequence_length": length,
                    "batch_size": 8192 // length,
                    "tokens_per_step": 8192,
                    "median_step_ms": median_ms,
                    "tokens_per_second": 8192 / (median_ms / 1000.0),
                    "peak_allocated_bytes": max(
                        item["peak_allocated_bytes"] for item in measurements
                    ),
                    "samples_ms": [
                        sample
                        for item in measurements for sample in item["samples_ms"]
                    ],
                    "order_balanced_passes": len(measurements),
                })
            aggregated.append({
                key: template[key] for key in (
                    "scale", "config", "architecture", "parameters",
                    "recurrent_mixer_parameters", "shared_initialization",
                )
            } | {"rows": rows})
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    data = load_character_data(args.data)
    vocab_size = len(data["alphabet"])
    raw_records = []
    for pass_name, reverse in (("forward", False), ("reverse", True)):
        scale_rows = tuple(reversed(SCALES)) if reverse else SCALES
        lengths = (8192, 4096, 2048) if reverse else (2048, 4096, 8192)
        for scale_index, (scale, dimensions) in enumerate(scale_rows):
            config = StudyConfig(vocab_size=vocab_size, **dimensions)
            order = ("samu", "rglru")
            if reverse ^ bool(scale_index % 2):
                order = tuple(reversed(order))
            for architecture in order:
                print(
                    f"pass={pass_name} scale={scale} architecture={architecture}",
                    flush=True,
                )
                raw_records.append({
                    "measurement_pass": pass_name,
                    "scale": scale,
                    "config": asdict(config),
                    **model_row(config, architecture, args.seed, lengths),
                })
    records = aggregate_passes(raw_records)

    # The trained quality experiment uses L=256 and B=32.  Re-measure this
    # exact operating point with the current fused kernels, because training
    # logs made before a kernel revision are not valid performance evidence.
    trained_config = StudyConfig(vocab_size=vocab_size)
    trained_point = []
    trained_raw = []
    for order in (("rglru", "samu"), ("samu", "rglru")):
        for architecture in order:
            model = load_trained(
                trained_config, architecture, args.seed, args.checkpoint_root
            )
            rows = benchmark_complete_training_steps(
                model, trained_config, lengths=(256,), tokens_per_step=8192
            )
            trained_raw.append({"architecture": architecture, **rows[0]})
            del model
            torch.cuda.empty_cache()
    for architecture in ("rglru", "samu"):
        candidates = [row for row in trained_raw if row["architecture"] == architecture]
        median_ms = statistics.fmean(row["median_step_ms"] for row in candidates)
        trained_point.append({
            "architecture": architecture,
            "sequence_length": 256,
            "batch_size": 32,
            "tokens_per_step": 8192,
            "median_step_ms": median_ms,
            "tokens_per_second": 8192 / (median_ms / 1000.0),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in candidates),
            "samples_ms": [sample for row in candidates for sample in row["samples_ms"]],
            "order_balanced_passes": len(candidates),
        })

    properties = torch.cuda.get_device_properties(0)
    atomic_json(args.output, {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "source": "Griffin Figure 3 and Section 4.3",
            "matched": [
                "three model scales",
                "sequence lengths 2048, 4096 and 8192",
                "8192 tokens per complete optimizer step",
                "BF16 autocast, cross entropy, backward, clipping and AdamW",
            ],
            "hardware_difference": (
                "one H800 PCIe instead of a multi-device TPU-v3 setup"
            ),
            "scale_difference": (
                "explicit small/medium/large models instead of 400M/1.3B/7B; "
                "the original sizes require model and ZeRO sharding unavailable "
                "on the rented single-GPU node"
            ),
        },
        "official_recurrentgemma_commit": OFFICIAL_RECURRENTGEMMA_COMMIT,
        "environment": {
            "hostname": platform.node(),
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fixed_tokens_per_step": 8192,
        "records": records,
        "measurement_passes": raw_records,
        "trained_L256_point": trained_point,
    })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
