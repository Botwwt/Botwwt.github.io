"""Full-model analogue of Griffin Appendix Figure 8(b) on one H800.

For three explicitly scaled Hawk-style models, this benchmark changes only the
recurrence scan implementation: state-stationary Triton linear scan, framework
eager linear scan, materialized BF16 associative scan, or materialized FP32
associative scan.  It measures complete optimizer steps at B=4, L=2048.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time

import torch

from run_griffin_training_matrix import SCALES
from run_small_model_study import (
    OFFICIAL_RECURRENTGEMMA_COMMIT,
    StudyConfig,
    benchmark_complete_training_steps,
    build_model,
    load_character_data,
)


BACKENDS = (
    "triton",
    "framework_eager",
    "associative_bf16",
    "associative_fp32",
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    vocab_size = len(load_character_data(args.data)["alphabet"])
    rows = []
    for scale_index, (scale, dimensions) in enumerate(SCALES):
        config = StudyConfig(vocab_size=vocab_size, **dimensions)
        for architecture in ("samu", "rglru"):
            order = BACKENDS if (scale_index + (architecture == "rglru")) % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                print(
                    f"scale={scale} architecture={architecture} backend={backend}",
                    flush=True,
                )
                model, _ = build_model(config, architecture, args.seed)
                model = model.cuda()
                set_backend(model, backend)
                base = {
                    "scale": scale,
                    "architecture": architecture,
                    "backend": backend,
                    "config": asdict(config),
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                try:
                    measured = benchmark_complete_training_steps(
                        model, config, lengths=(2048,), tokens_per_step=8192
                    )[0]
                    rows.append({**base, "status": "measured", **measured})
                except torch.cuda.OutOfMemoryError as error:
                    rows.append({**base, "status": "out_of_memory", "error": str(error)})
                finally:
                    del model
                    torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    atomic_json(args.output, {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "source": "Griffin Appendix D.2 Figure 8(b)",
            "matched": (
                "three model scales; complete training-step runtime; only scan "
                "backend changes within each architecture"
            ),
            "backend_mapping": {
                "triton": "NVIDIA analogue of the custom Pallas linear scan",
                "framework_eager": "transparent PyTorch analogue of native JAX linear scan",
                "associative_bf16": "materialized BF16 parallel prefix",
                "associative_fp32": "materialized FP32 parallel prefix",
            },
            "important_difference": (
                "H800/PyTorch/Triton and scaled models, not TPU-v3/JAX/Pallas "
                "at 400M/1.3B/7B"
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
        "rows": rows,
    })
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
