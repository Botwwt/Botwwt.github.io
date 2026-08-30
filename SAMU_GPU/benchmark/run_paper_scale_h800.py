"""Single-H800 SAMU/RG-LRU training steps at Griffin Table 2 scales.

This extends the small scaling study to the paper's 400M and 1.3B Hawk body
dimensions.  The 7B point is rejected before allocation when the *minimum*
resident bytes for FP32 parameters, FP32 gradients and Adam moments already
exceed physical HBM.  No single-GPU result is presented as model-parallel or
ZeRO evidence.

Each measured point performs BF16-autocast forward, cross entropy, backward,
gradient clipping and AdamW.  The total token count stays fixed at 8192 while
sequence length changes from 2K to 4K to 8K, matching Griffin Figure 3's key
experimental axis.  Forward and reverse measurement orders are averaged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from run_small_model_study import (
    OFFICIAL_RECURRENTGEMMA_COMMIT,
    StudyConfig,
    benchmark_complete_training_steps,
    build_model,
    load_character_data,
)


PAPER_SCALES = (
    ("400m", dict(width=1536, rnn_width=2048, depth=12)),
    ("1.3b", dict(width=2048, rnn_width=2560, depth=24)),
    ("7b", dict(width=4096, rnn_width=5632, depth=32)),
)
LENGTHS = (2048, 4096, 8192)
TOKENS_PER_STEP = 8192


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def exact_parameter_count(config: StudyConfig, architecture: str) -> int:
    """Count this harness exactly without constructing a multi-billion model."""
    d, r, n = config.width, config.rnn_width, config.depth
    common_per_block = (
        3 * d * r
        + 9 * d * d
        + 10 * d
        + (config.conv_width + 3) * r
    )
    if architecture == "samu":
        mixer = 3 * r + 4
    else:
        mixer = 2 * r * r // config.num_gate_blocks + 3 * r
    return config.vocab_size * d + d + n * (common_per_block + mixer)


def minimum_fp32_adam_bytes(parameters: int) -> int:
    # This benchmark keeps FP32 master parameters.  Before activations, CUDA
    # workspaces or fragmentation, each parameter needs: parameter (4 B),
    # gradient (4 B), first Adam moment (4 B), second Adam moment (4 B).
    return 16 * parameters


def architecture_audit(vocab_size: int) -> list[dict]:
    """Record the official Hawk block contract and the sole SAMU change."""
    rows = []
    for scale, dimensions in PAPER_SCALES:
        d = dimensions["width"]
        r = dimensions["rnn_width"]
        depth = dimensions["depth"]
        common_block = 3 * d * r + 9 * d * d + 10 * d + 7 * r
        official_rglru_mixer = 2 * r * r // 16 + 3 * r
        canonical_samu_mixer = 3 * r + 4
        rows.append({
            "scale": scale,
            "width": d,
            "rnn_width": r,
            "depth": depth,
            "vocabulary_size": vocab_size,
            "official_hawk_contract": {
                "temporal_blocks": depth,
                "temporal_pre_norm": "RMSNorm(eps=1e-6, scale initialized to zero and applied as 1+scale)",
                "recurrent_block": "parallel D->D_RNN x/y projections; width-4 causal depthwise Conv1D then recurrence on x; GeLU on y; multiply; D_RNN->D projection",
                "channel_block": "RMSNorm then expansion-3 gated GeLU MLP with residual",
                "embedding": "tied input/output embedding; no sqrt(D) scaling for Griffin/Hawk paper presets",
                "gate_blocks": 16,
                "final_projection_initialization_variance_scale": 2.0 / depth,
            },
            "common_parameters_per_block": common_block,
            "official_rglru_mixer_parameters_per_block": official_rglru_mixer,
            "canonical_samu_mixer_parameters_per_block": canonical_samu_mixer,
            "only_changed_component": "RG-LRU mixer replaced by canonical SAMU mixer; all listed shell components are identical",
        })
    return rows


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def set_scan_backend(model, backend: str) -> None:
    for block in model.blocks:
        block.mixer.set_scan_backend(backend)


def measure_architecture(
    config: StudyConfig,
    architecture: str,
    seed: int,
    lengths: tuple[int, ...],
    backend: str,
) -> dict:
    parameters = exact_parameter_count(config, architecture)
    minimum_bytes = minimum_fp32_adam_bytes(parameters)
    total_hbm = torch.cuda.get_device_properties(0).total_memory
    base = {
        "architecture": architecture,
        "selected_scan_backend": backend,
        "parameters": parameters,
        "minimum_fp32_parameter_gradient_adam_bytes": minimum_bytes,
    }
    if minimum_bytes >= total_hbm:
        return {
            **base,
            "status": "not_runnable_minimum_optimizer_state_exceeds_hbm",
            "rows": [
                {
                    "sequence_length": length,
                    "batch_size": TOKENS_PER_STEP // length,
                    "status": "not_runnable_minimum_optimizer_state_exceeds_hbm",
                }
                for length in lengths
            ],
        }

    model = None
    try:
        model, shared = build_model(config, architecture, seed)
        observed_parameters = sum(parameter.numel() for parameter in model.parameters())
        if observed_parameters != parameters:
            raise RuntimeError(
                f"parameter count mismatch: formula={parameters}, model={observed_parameters}"
            )
        model = model.cuda()
        set_scan_backend(model, backend)
        rows = []
        for length in lengths:
            print(
                f"measure architecture={architecture} length={length}", flush=True
            )
            try:
                row = benchmark_complete_training_steps(
                    model,
                    config,
                    lengths=(length,),
                    tokens_per_step=TOKENS_PER_STEP,
                )[0]
                rows.append({"status": "measured", **row})
            except torch.cuda.OutOfMemoryError as error:
                rows.append({
                    "sequence_length": length,
                    "batch_size": TOKENS_PER_STEP // length,
                    "status": "out_of_memory",
                    "error": str(error),
                })
                release_cuda()
        return {**base, "status": "measured_or_attempted", "shared_initialization": shared, "rows": rows}
    except torch.cuda.OutOfMemoryError as error:
        return {
            **base,
            "status": "out_of_memory_during_model_or_optimizer_setup",
            "error": str(error),
            "rows": [],
        }
    finally:
        if model is not None:
            del model
        release_cuda()


def aggregate(raw: list[dict]) -> list[dict]:
    output = []
    for scale, dimensions in PAPER_SCALES:
        for architecture in ("samu", "rglru"):
            candidates = [
                item for item in raw
                if item["scale"] == scale and item["architecture"] == architecture
            ]
            if scale == "7b":
                output.append(candidates[0])
                continue
            rows = []
            for length in LENGTHS:
                measured = [
                    row
                    for candidate in candidates
                    for row in candidate.get("rows", [])
                    if row.get("sequence_length") == length
                    and row.get("status") == "measured"
                ]
                if not measured:
                    attempts = [
                        row
                        for candidate in candidates
                        for row in candidate.get("rows", [])
                        if row.get("sequence_length") == length
                    ]
                    rows.append(attempts[0] if attempts else {
                        "sequence_length": length,
                        "batch_size": TOKENS_PER_STEP // length,
                        "status": "not_measured",
                    })
                    continue
                median_ms = statistics.fmean(row["median_step_ms"] for row in measured)
                rows.append({
                    "sequence_length": length,
                    "batch_size": TOKENS_PER_STEP // length,
                    "tokens_per_step": TOKENS_PER_STEP,
                    "status": "measured",
                    "order_balanced_median_step_ms": median_ms,
                    "tokens_per_second": TOKENS_PER_STEP / (median_ms / 1000.0),
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
                    "samples_ms": [sample for row in measured for sample in row["samples_ms"]],
                    "measurement_passes": len(measured),
                })
            template = candidates[0]
            output.append({
                "scale": scale,
                # Preserve the byte-level enwik8 vocabulary used by the run.
                # A hard-coded small-corpus vocabulary here would make the
                # reported parameter count and the recorded configuration
                # disagree even though the measured model itself is correct.
                "config": template["config"],
                "architecture": architecture,
                "selected_scan_backend": template["selected_scan_backend"],
                "parameters": template["parameters"],
                "minimum_fp32_parameter_gradient_adam_bytes": template[
                    "minimum_fp32_parameter_gradient_adam_bytes"
                ],
                "status": "measured" if all(row["status"] == "measured" for row in rows) else "partial",
                "rows": rows,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samu-backend", default="chunk16")
    parser.add_argument("--rglru-backend", default="chunk16")
    parser.add_argument("--samu-1p3b-backend")
    parser.add_argument("--rglru-1p3b-backend")
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="Use the conference-scale 32K vocabulary by default; pass 0 to infer it from --data.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    vocab_size = (
        args.vocab_size
        if args.vocab_size > 0
        else len(load_character_data(args.data)["alphabet"])
    )
    def selected_backend(scale: str, architecture: str) -> str:
        base = args.samu_backend if architecture == "samu" else args.rglru_backend
        if scale != "1.3b":
            return base
        override = (
            args.samu_1p3b_backend
            if architecture == "samu"
            else args.rglru_1p3b_backend
        )
        return override or base

    raw = []
    for pass_name, reverse in (("forward", False), ("reverse", True)):
        scale_order = tuple(reversed(PAPER_SCALES[:2])) if reverse else PAPER_SCALES[:2]
        length_order = tuple(reversed(LENGTHS)) if reverse else LENGTHS
        for scale_index, (scale, dimensions) in enumerate(scale_order):
            config = StudyConfig(
                vocab_size=vocab_size,
                embedding_scale_by_sqrt_dim=False,
                final_w_init_variance_scale=2.0 / dimensions["depth"],
                **dimensions,
            )
            architecture_order = ("samu", "rglru")
            if reverse ^ bool(scale_index % 2):
                architecture_order = tuple(reversed(architecture_order))
            for architecture in architecture_order:
                print(
                    f"pass={pass_name} scale={scale} architecture={architecture}",
                    flush=True,
                )
                raw.append({
                    "measurement_pass": pass_name,
                    "scale": scale,
                    "config": asdict(config),
                    **measure_architecture(
                        config,
                        architecture,
                        args.seed,
                        length_order,
                        selected_backend(scale, architecture),
                    ),
                })

    # The 7B model cannot be allocated with this optimizer on one 80 GB GPU.
    # Record the exact lower bound instead of constructing a model that is
    # guaranteed to fail before activations exist.
    scale, dimensions = PAPER_SCALES[-1]
    config = StudyConfig(
        vocab_size=vocab_size,
        embedding_scale_by_sqrt_dim=False,
        final_w_init_variance_scale=2.0 / dimensions["depth"],
        **dimensions,
    )
    for architecture in ("samu", "rglru"):
        raw.append({
            "measurement_pass": "analytical_hbm_preflight",
            "scale": scale,
            "config": asdict(config),
            **measure_architecture(
                config,
                architecture,
                args.seed,
                LENGTHS,
                selected_backend(scale, architecture),
            ),
        })

    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "source": "Griffin Table 2, Figure 3, Section 4.3 and Appendix Figure 8(b)",
            "matched": [
                "Table 2 widths, recurrent widths and depths for 400M, 1.3B and 7B",
                "sequence lengths 2048, 4096 and 8192",
                "8192 tokens per complete optimizer step",
                "BF16 autocast forward, loss, backward, clipping and AdamW",
                "forward and reverse measurement orders",
                f"vocabulary size {vocab_size}",
                "pure-Hawk all-recurrent residual blocks",
                "parallel D-to-D_RNN branches, width-4 depthwise convolution, multiplicative join and expansion-3 gated MLP",
                "paper embedding scaling disabled and final projection initialization variance 2/N",
            ],
            "important_difference": (
                "one H800 instead of model-parallel TPU-v3 pods; the 7B lower-bound "
                "memory requirement exceeds HBM before activations"
            ),
        },
        "precision": {
            "resident_master_parameters": "FP32",
            "matrix_and_activation_compute": "BF16 autocast",
            "recurrent_accumulation": "FP32",
            "optimizer_moments": "FP32 AdamW",
        },
        "official_recurrentgemma_commit": OFFICIAL_RECURRENTGEMMA_COMMIT,
        "architecture_audit": architecture_audit(vocab_size),
        "environment": {
            "hostname": platform.node(),
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fixed_tokens_per_step": TOKENS_PER_STEP,
        "selected_scan_backends": {
            scale: {
                architecture: selected_backend(scale, architecture)
                for architecture in ("samu", "rglru")
            }
            for scale in ("400m", "1.3b", "7b")
        },
        "records": aggregate(raw),
        "measurement_passes": raw,
    }
    atomic_json(args.output, result)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
