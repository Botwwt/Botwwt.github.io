"""Griffin Section 5 on a full 1.3B-shape recurrent decoder on one H800.

The decoder uses the Table 2 width (2048), recurrent width (2560), depth
(24) and a 32K vocabulary.  Every generated token executes embedding lookup,
all recurrent blocks (normalization, two input projections, causal depthwise
convolution, recurrence, output projection and gated MLP), final
normalization, the tied vocabulary projection and greedy token selection.

SAMU and RG-LRU use the same shell and equal FP32 recurrent-cache bytes.
RG-LRU keeps the official RecurrentGemma equation and 16 block-diagonal gate
groups.  Runs are counterbalanced AB/BA.  Random parameters are sufficient
for a systems-speed experiment; this file never presents them as a quality
checkpoint.
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

import torch

from run_small_model_study import (
    CapturedDecoder,
    OFFICIAL_RECURRENTGEMMA_COMMIT,
    SmallHawkLM,
    StudyConfig,
    set_seed,
)


LATENCY_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
THROUGHPUT_LENGTHS = (512, 1024, 2048, 4096)
BATCH_CANDIDATES = (1, 4, 16, 32, 64, 96, 128, 192, 256, 384, 512)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    alpha = position - low
    return ordered[low] * (1 - alpha) + ordered[high] * alpha


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def set_decode_choice(model: SmallHawkLM, choice: str | int) -> None:
    for block in model.blocks:
        if model.architecture == "samu":
            block.mixer.set_decode_backend(str(choice))
        else:
            label = str(choice)
            if label.startswith("fused"):
                block.mixer.set_decode_backend("fused")
                block.mixer.set_decode_num_warps(int(label.removeprefix("fused")))
            elif label.startswith("bmm"):
                block.mixer.set_decode_backend("bmm")
                block.mixer.set_decode_num_warps(int(label.removeprefix("bmm")))
            else:
                block.mixer.set_decode_backend("fused")
                block.mixer.set_decode_num_warps(int(choice))


def calibrate_decode(model: SmallHawkLM, batch: int) -> dict:
    """Choose on complete 64-token trajectories, never on one kernel call."""
    choices = (
        ("fused", "packed", "blocked16", "blocked32", "blocked64")
        if model.architecture == "samu"
        else ("fused1", "fused2", "fused4", "fused8", "bmm2", "bmm4", "bmm8")
    )
    rows = []
    for choice in choices:
        try:
            set_decode_choice(model, choice)
            decoder = CapturedDecoder(model, batch)
            probe_samples = []
            for _ in range(3):
                decoder.reset()
                probe_samples.append(decoder.trajectory_ms(64))
            elapsed = statistics.median(probe_samples)
            rows.append({
                "choice": choice,
                "median_64_step_ms": elapsed,
                "raw_64_step_ms": probe_samples,
                "tokens_per_second": batch * 64 * 1000.0 / elapsed,
                "status": "measured",
            })
            del decoder
            release()
        except torch.cuda.OutOfMemoryError as error:
            rows.append({
                "choice": choice,
                "status": "out_of_memory",
                "error": str(error),
            })
            release()
    winner = min(
        (row for row in rows if row["status"] == "measured"),
        key=lambda row: row["median_64_step_ms"],
    )
    set_decode_choice(model, winner["choice"])
    return {"batch_size": batch, "selected": winner["choice"], "rows": rows}


def build(config: StudyConfig, architecture: str, seed: int) -> SmallHawkLM:
    set_seed(seed)
    model = SmallHawkLM(config, architecture)
    # Section 5 is an inference experiment: weights and activations are BF16,
    # while each mixer explicitly keeps recurrence state in FP32.
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


@torch.no_grad()
def prompt_cache(model: SmallHawkLM, tokens: torch.Tensor):
    """Prefill without materializing [B,L,V] logits."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.embedding(tokens)
        if model.config.embedding_scale_by_sqrt_dim:
            x = x * (model.config.width ** 0.5)
        caches = []
        for block in model.blocks:
            x, cache = block(x, return_cache=True)
            caches.append(cache)
    return caches


def measure_trajectory(
    decoder: CapturedDecoder,
    lengths: tuple[int, ...],
    repeats: int,
    caches=None,
    segment_pos: int = 0,
) -> list[dict]:
    rows = []
    for length in lengths:
        samples = []
        for _ in range(repeats):
            decoder.reset(caches=caches, segment_pos=segment_pos)
            samples.append(decoder.trajectory_ms(length))
        median = statistics.median(samples)
        rows.append({
            "decode_length": length,
            "median_ms": median,
            "p10_ms": percentile(samples, 0.10),
            "p95_ms": percentile(samples, 0.95),
            "raw_samples_ms": samples,
            "tokens_per_second": decoder.batch * length * 1000.0 / median,
        })
    return rows


def run_architecture(
    config: StudyConfig,
    architecture: str,
    pass_name: str,
    seed: int,
    repeats: int,
    finalists: int,
) -> dict:
    model = build(config, architecture, seed)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    state_bytes = config.depth * config.rnn_width * 4
    convolution_bytes = (
        config.depth * (config.conv_width - 1) * config.rnn_width * 2
    )
    rows = []
    calibrations = {}

    print(f"{pass_name} {architecture}: B=16 latency", flush=True)
    calibrations[16] = calibrate_decode(model, 16)
    decoder = CapturedDecoder(model, 16)
    generator = torch.Generator(device="cuda").manual_seed(seed + 4096)
    prompt = torch.randint(
        0, config.vocab_size, (16, 4096), device="cuda", generator=generator
    )
    # Compile once before the measured prefill.
    _ = prompt_cache(model, prompt[:, :32])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    caches = prompt_cache(model, prompt)
    end.record()
    end.synchronize()
    prefill_ms = float(start.elapsed_time(end))
    for prompt_length, initial_cache in ((0, None), (4096, caches)):
        for row in measure_trajectory(
            decoder,
            LATENCY_LENGTHS,
            repeats,
            caches=initial_cache,
            segment_pos=prompt_length,
        ):
            rows.append({
                **row,
                "measurement_pass": pass_name,
                "architecture": architecture,
                "workload": "continuous_decode_latency",
                "batch_size": 16,
                "prompt_length": prompt_length,
                "prompt_prefill_ms_excluded_from_decode": (
                    prefill_ms if prompt_length else 0.0
                ),
                "decode_choice": calibrations[16]["selected"],
            })
    del decoder, prompt, caches
    release()

    print(f"{pass_name} {architecture}: declared batch search", flush=True)
    search = []
    for batch in BATCH_CANDIDATES:
        try:
            if batch not in calibrations:
                calibrations[batch] = calibrate_decode(model, batch)
            else:
                set_decode_choice(model, calibrations[batch]["selected"])
            torch.cuda.reset_peak_memory_stats()
            candidate = CapturedDecoder(model, batch)
            probe_samples = []
            for _ in range(3):
                candidate.reset()
                probe_samples.append(candidate.trajectory_ms(64))
            elapsed = statistics.median(probe_samples)
            search.append({
                "batch_size": batch,
                "probe_steps": 64,
                "median_tokens_per_second": batch * 64 * 1000.0 / elapsed,
                "raw_64_step_ms": probe_samples,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "status": "measured",
                "decode_choice": calibrations[batch]["selected"],
            })
            del candidate
            release()
        except torch.cuda.OutOfMemoryError as error:
            search.append({
                "batch_size": batch,
                "probe_steps": 64,
                "status": "out_of_memory",
                "error": str(error),
            })
            release()
    selected = [
        row["batch_size"]
        for row in sorted(
            (row for row in search if row["status"] == "measured"),
            key=lambda row: row["median_tokens_per_second"],
            reverse=True,
        )[:finalists]
    ]
    for batch in sorted(selected):
        print(
            f"{pass_name} {architecture}: complete throughput B={batch}",
            flush=True,
        )
        set_decode_choice(model, calibrations[batch]["selected"])
        candidate = CapturedDecoder(model, batch)
        for row in measure_trajectory(
            candidate, THROUGHPUT_LENGTHS, repeats
        ):
            rows.append({
                **row,
                "measurement_pass": pass_name,
                "architecture": architecture,
                "workload": "maximum_throughput_candidate",
                "batch_size": batch,
                "prompt_length": 0,
                "decode_choice": calibrations[batch]["selected"],
            })
        del candidate
        release()

    result = {
        "architecture": architecture,
        "measurement_pass": pass_name,
        "parameters": parameters,
        "bf16_parameter_bytes": 2 * parameters,
        "fp32_state_bytes_per_sequence": state_bytes,
        "bf16_convolution_cache_bytes_per_sequence": convolution_bytes,
        "batch_search": search,
        "selected_finalist_batches": selected,
        "decode_calibration": list(calibrations.values()),
        "rows": rows,
    }
    del model
    release()
    return result


def aggregate(passes: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for item in passes:
        for row in item["rows"]:
            key = (
                row["architecture"], row["workload"], row["batch_size"],
                row["prompt_length"], row["decode_length"],
            )
            groups.setdefault(key, []).append(row)
    rows = []
    for key, values in groups.items():
        architecture, workload, batch, prompt, length = key
        samples = [sample for row in values for sample in row["raw_samples_ms"]]
        median = statistics.median(samples)
        rows.append({
            "architecture": architecture,
            "workload": workload,
            "batch_size": batch,
            "prompt_length": prompt,
            "decode_length": length,
            "median_ms": median,
            "p10_ms": percentile(samples, 0.10),
            "p95_ms": percentile(samples, 0.95),
            "samples": len(samples),
            "raw_samples_ms": samples,
            "tokens_per_second": batch * length * 1000.0 / median,
            "measurement_passes": len(values),
            "decode_choices": sorted({str(row["decode_choice"]) for row in values}),
        })
    # Add the best measured full-trajectory batch at each output length.
    for architecture in ("samu", "rglru"):
        for length in THROUGHPUT_LENGTHS:
            candidates = [
                row for row in rows
                if row["architecture"] == architecture
                and row["workload"] == "maximum_throughput_candidate"
                and row["decode_length"] == length
            ]
            if candidates:
                rows.append({
                    **max(candidates, key=lambda row: row["tokens_per_second"]),
                    "workload": "maximum_throughput",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeats-per-order", type=int, default=3)
    parser.add_argument("--throughput-finalists", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = StudyConfig(
        vocab_size=32000,
        width=2048,
        rnn_width=2560,
        depth=24,
        mlp_expansion=3,
        embedding_scale_by_sqrt_dim=False,
        final_w_init_variance_scale=2.0 / 24,
    )
    passes = []
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            if previous.get("status") == "running":
                passes = previous.get("measurement_passes", [])
        except (json.JSONDecodeError, OSError):
            passes = []
    completed = {
        (item["measurement_pass"], item["architecture"])
        for item in passes
    }
    for pass_name, order in (
        ("AB", ("samu", "rglru")),
        ("BA", ("rglru", "samu")),
    ):
        for architecture in order:
            if (pass_name, architecture) in completed:
                print(f"resume: keep {pass_name} {architecture}", flush=True)
                continue
            passes.append(run_architecture(
                config,
                architecture,
                pass_name,
                args.seed,
                args.repeats_per_order,
                args.throughput_finalists,
            ))
            completed.add((pass_name, architecture))
            atomic_json(args.output, {
                "schema_version": 1,
                "generated_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "status": "running",
                "config": asdict(config),
                "protocol": {
                    "latency_batch": 16,
                    "prompt_lengths": [0, 4096],
                    "decode_lengths": list(LATENCY_LENGTHS),
                    "batch_candidates": list(BATCH_CANDIDATES),
                    "throughput_lengths": list(THROUGHPUT_LENGTHS),
                    "throughput_finalists_per_pass": args.throughput_finalists,
                    "repeats_per_order": args.repeats_per_order,
                },
                "completed_pass_architectures": sorted(
                    [list(item) for item in completed]
                ),
                "measurement_passes": passes,
                "rows": aggregate(passes),
            })
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",
        "scope": "random-weight full 1.3B-shape inference systems experiment; no quality claim",
        "paper_mapping": {
            "source": "Griffin Section 5 and Appendix F",
            "matched": [
                "Table 2 width 2048, recurrent width 2560 and depth 24",
                "32K vocabulary and tied vocabulary projection",
                "batch 16 with empty and 4096-token prompts",
                "continuous decode lengths 128 through 4096",
                "declared batch search followed by complete 512 through 4096 trajectories",
                "BF16 weights and activations with equal-byte FP32 recurrence caches",
                "AB and BA measurement orders",
            ],
        },
        "fairness": {
            "shared": "decoder dimensions, vocabulary, full block shell, cache bytes, prompts, sampler, precision and H800",
            "changed_only": "SAMU mixer versus RG-LRU mixer",
            "rglru": "official RecurrentGemma equation with 16 block-diagonal gate groups",
            "samu": "canonical single shared-control SAMU; no grouped control and no special-function approximation",
        },
        "official_recurrentgemma_commit": OFFICIAL_RECURRENTGEMMA_COMMIT,
        "config": asdict(config),
        "environment": {
            "hostname": platform.node(),
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "protocol": {
            "latency_batch": 16,
            "prompt_lengths": [0, 4096],
            "decode_lengths": list(LATENCY_LENGTHS),
            "batch_candidates": list(BATCH_CANDIDATES),
            "throughput_lengths": list(THROUGHPUT_LENGTHS),
            "throughput_finalists_per_pass": args.throughput_finalists,
            "repeats_per_order": args.repeats_per_order,
        },
        "measurement_passes": passes,
        "rows": aggregate(passes),
    }
    atomic_json(args.output, result)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
