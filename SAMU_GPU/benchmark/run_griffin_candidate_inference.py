"""Griffin Section 5 inference protocol for the SAMU GPU candidate.

This is a random-weight systems proxy, not a quality result.  It compares the
official RG-LRU equation with its paper-default 16 gate blocks against the
Grouped SAMU-16 direct-control candidate.  Direct control and grouped writes
change the architecture and require retraining before any model claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kernels import load_official_rglru, make_samu_parameters
from run_equal_kernel_benchmarks import stabilize_gpu
from run_griffin_section5 import ShellLayer, make_shell, mlp, percentile, rms_norm
from triton_rglru import pack_rglru, rglru_triton_auto, rglru_triton_decode
from triton_samu_grouped import (
    grouped_samu_rmsnorm_decode,
    grouped_samu_triton_decode,
    grouped_samu_triton_prefill,
    pack_grouped_samu_parameters,
    triton_rmsnorm,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


@dataclass
class CandidateModel:
    kind: str
    width: int
    depth: int
    modes: int
    vocab: int
    shell: list[ShellLayer]
    embedding: torch.Tensor
    final_norm: torch.Tensor
    temporal: list[Any]
    parameter_count: int
    state_bytes_per_sequence: int
    decode_positions: dict[int, torch.Tensor]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_model(kind: str, width: int, depth: int, expansion: int, vocab: int,
                rglru_source: Path, seed: int,
                shared_shell=None) -> CandidateModel:
    if shared_shell is None:
        shell, embedding, final_norm, shell_count = make_shell(
            width, depth, expansion, vocab, seed
        )
    else:
        shell, embedding, final_norm, shell_count = shared_shell
    modes = width // 2
    temporal, temporal_count = [], 0
    if kind == "samu_grouped16_direct":
        for layer in range(depth):
            parameters = make_samu_parameters(
                width, modes, "cuda", torch.bfloat16, seed=seed + layer * 17
            )
            packed = pack_grouped_samu_parameters(parameters, groups=16)
            temporal.append(packed)
            temporal_count += packed.learned_parameter_count - 2 * width
    elif kind == "rglru_official16":
        RGLRU = load_official_rglru(rglru_source)
        for layer in range(depth):
            torch.manual_seed(seed + layer * 17)
            module = RGLRU(
                width=width, num_heads=16, device="cuda", dtype=torch.bfloat16
            ).eval()
            temporal_count += sum(parameter.numel() for parameter in module.parameters())
            temporal.append(pack_rglru(module))
            del module
    else:
        raise ValueError(kind)
    return CandidateModel(
        kind=kind, width=width, depth=depth, modes=modes, vocab=vocab,
        shell=shell, embedding=embedding, final_norm=final_norm,
        temporal=temporal, parameter_count=shell_count + temporal_count,
        state_bytes_per_sequence=depth * width * 4,
        decode_positions={},
    )


def zero_cache(model: CandidateModel, batch: int):
    if model.kind == "samu_grouped16_direct":
        return [(
            torch.zeros(batch, model.modes, device="cuda"),
            torch.zeros(batch, model.modes, device="cuda"),
        ) for _ in range(model.depth)]
    return [torch.zeros(batch, model.width, device="cuda") for _ in range(model.depth)]


def prefill(model: CandidateModel, token_ids: torch.Tensor):
    x = model.embedding[token_ids]
    batch, length = token_ids.shape
    positions = torch.arange(length, device="cuda", dtype=torch.long).unsqueeze(0).expand(batch, -1)
    caches = []
    for layer, temporal in zip(model.shell, model.temporal):
        normalized = rms_norm(x, layer.norm_temporal)
        if model.kind == "samu_grouped16_direct":
            mixed, cache = grouped_samu_triton_prefill(
                normalized, temporal, chunk_size=32, num_warps=2,
                direct_control=True,
            )
        else:
            mixed, cache = rglru_triton_auto(normalized, positions, temporal)
        x = (x + mixed).to(torch.bfloat16)
        x = (x + mlp(rms_norm(x, layer.norm_mlp), layer)).to(torch.bfloat16)
        caches.append(cache)
    return rms_norm(x[:, -1], model.final_norm), caches


def decode_step(model: CandidateModel, token: torch.Tensor, caches):
    x = token
    next_caches = []
    positions = model.decode_positions.get(token.shape[0])
    if positions is None:
        positions = torch.ones(token.shape[0], 1, device="cuda", dtype=torch.long)
        model.decode_positions[token.shape[0]] = positions
    for layer, temporal, cache in zip(model.shell, model.temporal, caches):
        if model.kind == "samu_grouped16_direct":
            if token.shape[0] <= 4:
                block_m, warps = 16, 4
            elif token.shape[0] <= 64:
                block_m, warps = 32, 1
            else:
                block_m, warps = 64, 2
            mixed, next_cache = grouped_samu_rmsnorm_decode(
                x, layer.norm_temporal, cache, temporal,
                block_m=block_m, num_warps=warps,
                direct_control=True,
            )
        else:
            normalized = triton_rmsnorm(x, layer.norm_temporal)
            mixed, next_cache = rglru_triton_decode(
                normalized[:, None], positions, temporal, cache, num_warps=2
            )
            mixed = mixed[:, 0]
        x = (x + mixed).to(torch.bfloat16)
        x = (x + mlp(rms_norm(x, layer.norm_mlp), layer)).to(torch.bfloat16)
        next_caches.append(next_cache)
    x = rms_norm(x, model.final_norm)
    logits = x @ model.embedding.T
    next_ids = logits.argmax(dim=-1)
    return model.embedding[next_ids], next_caches


def prepare_prompt(model: CandidateModel, batch: int, prompt_length: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + batch * 31 + prompt_length)
    if prompt_length == 0:
        ids = torch.zeros(batch, device="cuda", dtype=torch.long)
        return model.embedding[ids], zero_cache(model, batch), 0.0
    ids = torch.randint(0, model.vocab, (batch, prompt_length), device="cuda", generator=generator)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    last, caches = prefill(model, ids)
    end.record(); end.synchronize()
    logits = last @ model.embedding.T
    next_ids = logits.argmax(dim=-1)
    return model.embedding[next_ids], caches, float(start.elapsed_time(end))


def timed_trajectory(model: CandidateModel, token, caches, checkpoints, repeats):
    checkpoints = sorted(checkpoints)
    result = {checkpoint: [] for checkpoint in checkpoints}
    warm_token, warm_caches = token, caches
    for _ in range(2):
        warm_token, warm_caches = decode_step(model, warm_token, warm_caches)
    torch.cuda.synchronize()
    for _ in range(repeats):
        current_token, current_caches = token, caches
        start = torch.cuda.Event(enable_timing=True)
        ends = {checkpoint: torch.cuda.Event(enable_timing=True) for checkpoint in checkpoints}
        start.record()
        for step in range(1, checkpoints[-1] + 1):
            current_token, current_caches = decode_step(model, current_token, current_caches)
            if step in ends:
                ends[step].record()
        ends[checkpoints[-1]].synchronize()
        for checkpoint in checkpoints:
            result[checkpoint].append(float(start.elapsed_time(ends[checkpoint])))
    return result


def summarize(model, order, workload, batch, prompt, samples, prefill_ms=0.0):
    rows = []
    for decoded, values in samples.items():
        median = statistics.median(values)
        rows.append({
            "model": model.kind, "measurement_order": order,
            "workload": workload, "batch": batch,
            "prompt_length": prompt, "decode_length": decoded,
            "median_ms": median, "p10_ms": percentile(values, 0.10),
            "p95_ms": percentile(values, 0.95), "raw_samples_ms": values,
            "tokens_per_second": batch * decoded * 1000.0 / median,
            "prompt_prefill_ms_excluded_from_decode": prefill_ms,
        })
    return rows


def correctness(rglru_source: Path) -> dict:
    width, modes, length = 128, 64, 32
    parameters = make_samu_parameters(width, modes, "cuda", torch.bfloat16, seed=6001)
    packed = pack_grouped_samu_parameters(parameters, groups=16)
    torch.manual_seed(6001)
    x = torch.randn(2, length, width, device="cuda", dtype=torch.bfloat16)
    prefill_out, prefill_cache = grouped_samu_triton_prefill(
        x, packed, direct_control=True
    )
    cache = (
        torch.zeros(2, modes, device="cuda"),
        torch.zeros(2, modes, device="cuda"),
    )
    outputs = []
    for step in range(length):
        output, cache = grouped_samu_triton_decode(
            x[:, step], cache, packed, block_m=4, num_warps=1,
            direct_control=True,
        )
        outputs.append(output)
    decode_out = torch.stack(outputs, dim=1)
    RGLRU = load_official_rglru(rglru_source)
    torch.manual_seed(6001)
    official = RGLRU(width=width, num_heads=16, device="cuda", dtype=torch.bfloat16).eval()
    rg = pack_rglru(official)
    token = torch.randn(3, 1, width, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(3, width, device="cuda")
    positions = torch.tensor([[0], [1], [7]], device="cuda")
    expected, expected_cache = official(token, positions, cache=state, return_cache=True)
    actual, actual_cache = rglru_triton_decode(token, positions, rg, state, num_warps=2)
    audit = {
        "samu_prefill_vs_repeated_decode_output_max_abs": float((prefill_out - decode_out).abs().max()),
        "samu_prefill_vs_repeated_decode_cache_max_abs": float(max(
            (prefill_cache[0] - cache[0]).abs().max(),
            (prefill_cache[1] - cache[1]).abs().max(),
        )),
        "rglru_fused_vs_official_output_max_abs": float((actual - expected).abs().max()),
        "rglru_fused_vs_official_cache_max_abs": float((actual_cache - expected_cache).abs().max()),
    }
    limits = {
        "samu_prefill_vs_repeated_decode_output_max_abs": 0.02,
        "samu_prefill_vs_repeated_decode_cache_max_abs": 0.002,
        "rglru_fused_vs_official_output_max_abs": 0.002,
        "rglru_fused_vs_official_cache_max_abs": 5e-5,
    }
    audit["limits"] = limits
    audit["passed"] = all(audit[key] <= limit for key, limit in limits.items())
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--throughput-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("[1/5] checking SAMU and official RG-LRU equations", flush=True)
    audit = correctness(args.rglru_source)
    if not audit["passed"]:
        raise RuntimeError(f"correctness failed: {audit}")
    stabilization = stabilize_gpu()
    kinds = ("samu_grouped16_direct", "rglru_official16")
    print("[2/5] building the shared 1B-scale random-weight shell", flush=True)
    shared_shell = make_shell(2048, 24, 3, 32000, args.seed)
    models = {
        kind: build_model(
            kind, 2048, 24, 3, 32000, args.rglru_source, args.seed,
            shared_shell=shared_shell,
        )
        for kind in kinds
    }
    metadata = {
        kind: {
            "parameter_count": model.parameter_count,
            "parameter_bytes_bf16": model.parameter_count * 2,
            "state_bytes_per_sequence": model.state_bytes_per_sequence,
            "temporal_backend": (
                "RMSNorm+direct shared controls, grouped complex write/update"
                if kind == "samu_grouped16_direct"
                else "RMSNorm, official-equation 16-block fused RG-LRU"
            ),
        }
        for kind, model in models.items()
    }
    rows, searches = [], {kind: [] for kind in kinds}
    decode_lengths = [128, 256, 512, 1024, 2048, 4096]
    print("[3/5] running B=16 continuous-decode trajectories", flush=True)
    for prompt in (0, 4096):
        for order_name, order in (("AB", kinds), ("BA", tuple(reversed(kinds)))):
            stabilize_gpu(seconds=0.5)
            for kind in order:
                print(f"  latency prompt={prompt} order={order_name} model={kind}", flush=True)
                model = models[kind]
                token, caches, prefill_ms = prepare_prompt(model, 16, prompt, args.seed)
                samples = timed_trajectory(
                    model, token, caches, decode_lengths, args.latency_repeats
                )
                rows.extend(summarize(
                    model, order_name, "continuous_decode_latency", 16,
                    prompt, samples, prefill_ms,
                ))
                del token, caches
                torch.cuda.empty_cache()

    batch_candidates = [1, 4, 16, 32, 64, 96, 128, 192, 256]
    finalists = set()
    print("[4/5] probing the declared batch range", flush=True)
    for kind in kinds:
        model = models[kind]
        for batch in batch_candidates:
            print(f"  probe model={kind} batch={batch}", flush=True)
            try:
                torch.cuda.reset_peak_memory_stats()
                token, caches, _ = prepare_prompt(model, batch, 0, args.seed)
                sample = timed_trajectory(model, token, caches, [32], 1)[32][0]
                row = {
                    "batch": batch, "probe_steps": 32,
                    "latency_ms": sample,
                    "tokens_per_second": batch * 32 * 1000.0 / sample,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "status": "measured",
                }
                del token, caches
            except torch.cuda.OutOfMemoryError:
                row = {"batch": batch, "probe_steps": 32, "status": "oom"}
                torch.cuda.empty_cache()
            searches[kind].append(row)
        measured = sorted(
            (row for row in searches[kind] if row["status"] == "measured"),
            key=lambda row: row["tokens_per_second"], reverse=True,
        )
        finalists.update(row["batch"] for row in measured[:2])

    throughput_lengths = [512, 1024, 2048, 4096]
    print(f"[5/5] completing finalist trajectories batches={sorted(finalists)}", flush=True)
    for batch in sorted(finalists):
        for order_name, order in (("AB", kinds), ("BA", tuple(reversed(kinds)))):
            stabilize_gpu(seconds=0.5)
            for kind in order:
                print(f"  throughput batch={batch} order={order_name} model={kind}", flush=True)
                model = models[kind]
                token, caches, _ = prepare_prompt(model, batch, 0, args.seed)
                samples = timed_trajectory(
                    model, token, caches, throughput_lengths,
                    args.throughput_repeats,
                )
                rows.extend(summarize(
                    model, order_name, "bounded_maximum_throughput_candidate",
                    batch, 0, samples,
                ))
                del token, caches
                torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "title": "Griffin Section 5 protocol: SAMU GPU candidate versus official RG-LRU-16",
        "scope": "random-weight 1B-scale systems proxy; no trained-model quality claim",
        "paper_protocol": {
            "latency_batch": 16,
            "latency_repeats_per_order": args.latency_repeats,
            "prompt_lengths": [0, 4096],
            "decode_lengths": decode_lengths,
            "throughput_lengths": throughput_lengths,
            "throughput_repeats_per_order": args.throughput_repeats,
            "bounded_batch_search": batch_candidates,
            "decode_time_model": "(parameter bytes + batch * recurrent-state bytes) / effective bandwidth",
        },
        "fairness": {
            "shared": "the two candidates reference the exact same embedding, 24 decoder-shell layer and final-normalization tensor objects; width 2048, expansion 3, vocabulary 32000, BF16 parameters/activations, FP32 equal-byte recurrent cache, prompts, greedy sampler, H800 and AB/BA timing are also identical",
            "rglru": "pinned official equation and the paper's 16 block-diagonal gate groups",
            "samu": "16 grouped complex writes and direct shared control; this architecture candidate has fewer temporal parameters and requires retraining",
            "important_boundary": "both are recurrent and neither has a sequence-growing KV cache",
        },
        "environment": {
            "hostname": platform.node(), "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "rglru_commit": RGLRU_COMMIT,
        },
        "gpu_stabilization": stabilization,
        "correctness": audit,
        "models": metadata,
        "throughput_batch_search": searches,
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(f"wrote {args.output}", flush=True)
    del models, shared_shell
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
