"""Appendix Figure 8(b) scan-backend ablation at paper Hawk scales.

Only the recurrence scan changes.  The custom state-stationary Triton scan is
the NVIDIA analogue of the paper's Pallas scan; the transparent PyTorch time
loop is the native-framework comparison.  A third row records the exact
chunked path used by the H800-optimized primary comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import time

import torch

from run_paper_scale_h800 import (
    PAPER_SCALES,
    exact_parameter_count,
    minimum_fp32_adam_bytes,
)
from run_small_model_study import (
    OFFICIAL_RECURRENTGEMMA_COMMIT,
    StudyConfig,
    benchmark_complete_training_steps,
    build_model,
)


BACKENDS = (
    "triton",
    "framework_eager",
    "associative_bf16",
    "associative_fp32",
    "chunk16",
    "chunk32",
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


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = []
    hbm = torch.cuda.get_device_properties(0).total_memory
    for scale_index, (scale, dimensions) in enumerate(PAPER_SCALES):
        config = StudyConfig(
            vocab_size=args.vocab_size,
            embedding_scale_by_sqrt_dim=False,
            final_w_init_variance_scale=2.0 / dimensions["depth"],
            **dimensions,
        )
        for architecture in ("samu", "rglru"):
            parameters = exact_parameter_count(config, architecture)
            minimum_bytes = minimum_fp32_adam_bytes(parameters)
            base = {
                "scale": scale,
                "architecture": architecture,
                "config": asdict(config),
                "parameters": parameters,
                "minimum_fp32_parameter_gradient_adam_bytes": minimum_bytes,
            }
            if minimum_bytes >= hbm:
                for backend in BACKENDS:
                    rows.append({
                        **base,
                        "backend": backend,
                        "status": "not_runnable_minimum_optimizer_state_exceeds_hbm",
                    })
                continue
            order = BACKENDS if (scale_index + (architecture == "rglru")) % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                print(
                    f"scale={scale} architecture={architecture} backend={backend}",
                    flush=True,
                )
                model = None
                try:
                    model, _ = build_model(config, architecture, args.seed)
                    model = model.cuda()
                    set_backend(model, backend)
                    measurement = benchmark_complete_training_steps(
                        model,
                        config,
                        lengths=(2048,),
                        tokens_per_step=8192,
                    )[0]
                    rows.append({
                        **base,
                        "backend": backend,
                        "status": "measured",
                        **measurement,
                    })
                except torch.cuda.OutOfMemoryError as error:
                    rows.append({
                        **base,
                        "backend": backend,
                        "status": "out_of_memory",
                        "error": str(error),
                    })
                finally:
                    if model is not None:
                        del model
                    release()
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "source": "Griffin Appendix D.2 Figure 8(b)",
            "matched": [
                "complete Hawk optimizer step",
                "L=2048 and B=4 (8192 tokens)",
                "paper Table 2 widths, recurrent widths and depths",
                "only the recurrence scan backend changes",
                "official 16-block RG-LRU equation",
            ],
            "backend_mapping": {
                "triton": "state-stationary NVIDIA analogue of Pallas linear scan; wide SAMU first materializes exact complex transitions because the fully fused serial program is capped at 512 modes",
                "framework_eager": "transparent PyTorch time loop analogue of native JAX scan",
                "associative_bf16": "materialized BF16 parallel prefix used in Figure 8",
                "associative_fp32": "materialized FP32 parallel prefix used in Figure 8",
                "chunk16": "H800 exact 16-token chunk path",
                "chunk32": "H800 exact 32-token chunk path",
            },
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
    }
    atomic_json(args.output, result)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
