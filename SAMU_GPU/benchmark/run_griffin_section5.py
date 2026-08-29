"""Griffin Section 5 inspired SAMU vs RG-LRU inference experiment.

This is a systems proxy, not a trained language-model quality comparison.  It
uses a shared decoder-only shell and swaps only the recurrent operator.  The
default shell has roughly 1.07B parameters, matching the scale and measurement
axes of Griffin Section 5 while keeping SAMU and RG-LRU width, parameter count,
state bytes, precision, prompt, sampler, and timing procedure equal.

Reported latency is continuous autoregressive sampling at batch 16 after an
empty or 4096-token prompt.  Reported full-trajectory throughput searches a
predeclared finite batch range on the same GPU and then measures complete
512/1024/2048/4096-token trajectories.  It is not an unrestricted maximum.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kernels import load_official_rglru, make_samu_parameters
from triton_rglru import PackedRGLRU, pack_rglru, rglru_triton_auto, rglru_triton_serial
from triton_samu import (
    PackedSamu,
    pack_samu_parameters,
    samu_packed_reference,
    samu_triton_auto,
    samu_triton_decode_split,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"


@dataclass
class ShellLayer:
    norm_temporal: torch.Tensor
    norm_mlp: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class SystemModel:
    kind: str
    width: int
    depth: int
    modes: int
    vocab: int
    shell: list[ShellLayer]
    embedding: torch.Tensor
    final_norm: torch.Tensor
    temporal: list[Any]
    source_parameters: list[Any]
    parameter_count: int
    state_bytes_per_sequence: int
    decode_positions: dict[int, torch.Tensor]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception:
        return None


def stabilize_gpu(seconds: float = 2.0) -> dict[str, Any]:
    """Bring clocks to steady state before full-trajectory timing."""

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    started = time.perf_counter()
    launches = 0
    while time.perf_counter() - started < seconds:
        for _ in range(16):
            torch.mm(a, a, out=out)
            launches += 1
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del a, out
    torch.cuda.empty_cache()
    return {
        "method": "repeated 4096x4096 BF16 GEMM followed by synchronization",
        "requested_seconds": seconds,
        "elapsed_seconds": elapsed,
        "launches": launches,
        "post_warmup_clock_power": command_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ]),
    }


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    import triton
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_and_board": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ]),
        "source_commits": {"recurrentgemma": RGLRU_COMMIT},
    }


def make_weight(shape: tuple[int, ...], generator: torch.Generator,
                scale: float) -> torch.Tensor:
    value = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    value.normal_(0.0, scale, generator=generator)
    return value


def make_shell(width: int, depth: int, expansion: int, vocab: int,
               seed: int) -> tuple[list[ShellLayer], torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shell = []
    hidden = expansion * width
    for _ in range(depth):
        shell.append(ShellLayer(
            norm_temporal=torch.ones(width, device="cuda", dtype=torch.bfloat16),
            norm_mlp=torch.ones(width, device="cuda", dtype=torch.bfloat16),
            gate=make_weight((hidden, width), generator, 1 / math.sqrt(width)),
            up=make_weight((hidden, width), generator, 1 / math.sqrt(width)),
            down=make_weight((width, hidden), generator, 1 / math.sqrt(hidden)),
        ))
    embedding = make_weight((vocab, width), generator, 1 / math.sqrt(width))
    final_norm = torch.ones(width, device="cuda", dtype=torch.bfloat16)
    count = sum(
        layer.norm_temporal.numel() + layer.norm_mlp.numel()
        + layer.gate.numel() + layer.up.numel() + layer.down.numel()
        for layer in shell
    ) + embedding.numel() + final_norm.numel()
    return shell, embedding, final_norm, count


def build_model(kind: str, width: int, depth: int, expansion: int, vocab: int,
                rglru_source: Path, seed: int = 5701) -> SystemModel:
    shell, embedding, final_norm, shell_count = make_shell(width, depth, expansion, vocab, seed)
    temporal, source_parameters = [], []
    modes = width // 2
    temporal_count = 0
    if kind == "samu":
        for layer in range(depth):
            params = make_samu_parameters(width, modes, "cuda", torch.bfloat16, seed=seed + layer * 17)
            packed = pack_samu_parameters(params)
            source_parameters.append(params)
            temporal.append(packed)
            temporal_count += sum(value.numel() for value in params.__dict__.values())
        state_bytes = depth * 2 * modes * 4
    elif kind == "rglru":
        RGLRU = load_official_rglru(rglru_source)
        for layer in range(depth):
            torch.manual_seed(seed + layer * 17)
            # Two block-diagonal heads make the two official RG-LRU gate
            # matrices contain the same number of learned weights and BF16
            # multiply-adds as SAMU's two real write matrices.  This is a
            # parameter-matched equation test, not a RecurrentGemma preset.
            module = RGLRU(width=width, num_heads=2, device="cuda", dtype=torch.bfloat16).eval()
            temporal_count += sum(parameter.numel() for parameter in module.parameters())
            temporal.append(pack_rglru(module))
            del module
        state_bytes = depth * width * 4
    else:
        raise ValueError(kind)
    return SystemModel(
        kind=kind, width=width, depth=depth, modes=modes, vocab=vocab,
        shell=shell, embedding=embedding, final_norm=final_norm,
        temporal=temporal, source_parameters=source_parameters,
        parameter_count=shell_count + temporal_count,
        state_bytes_per_sequence=state_bytes,
        decode_positions={},
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # torch.nn.functional.rms_norm is unavailable on the pinned Torch 2.1
    # environment.  Accumulate the variance in FP32, then return BF16 so both
    # recurrent variants use the same numerical path.
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    return (x.float() * inverse_rms * weight.float()).to(torch.bfloat16)


def mlp(x: torch.Tensor, layer: ShellLayer) -> torch.Tensor:
    gate = F.gelu(F.linear(x, layer.gate), approximate="tanh")
    up = F.linear(x, layer.up)
    return F.linear(gate * up, layer.down)


def zero_cache(model: SystemModel, batch: int):
    if model.kind == "samu":
        return [(
            torch.zeros(batch, model.modes, device="cuda", dtype=torch.float32),
            torch.zeros(batch, model.modes, device="cuda", dtype=torch.float32),
        ) for _ in range(model.depth)]
    return [torch.zeros(batch, model.width, device="cuda", dtype=torch.float32)
            for _ in range(model.depth)]


def prefill(model: SystemModel, token_ids: torch.Tensor):
    x = model.embedding[token_ids]
    batch, length = token_ids.shape
    positions = torch.arange(length, device="cuda", dtype=torch.long).unsqueeze(0).expand(batch, -1)
    caches = []
    for index, (layer, temporal) in enumerate(zip(model.shell, model.temporal)):
        normalized = rms_norm(x, layer.norm_temporal)
        if model.kind == "samu":
            mixed, cache = samu_triton_auto(
                normalized, model.source_parameters[index], temporal, return_cache=True
            )
            mixed = mixed.reshape(batch, length, model.width)
        else:
            mixed, cache = rglru_triton_auto(normalized, positions, temporal)
        x = (x + mixed).to(torch.bfloat16)
        x = (x + mlp(rms_norm(x, layer.norm_mlp), layer)).to(torch.bfloat16)
        caches.append(cache)
    return rms_norm(x[:, -1], model.final_norm), caches


def decode_step(model: SystemModel, token: torch.Tensor, caches):
    x = token
    next_caches = []
    # Segment positions are inference metadata. Allocate the non-reset vector
    # once during warmup, instead of adding an unrelated CUDA initialization
    # kernel to every timed decode step.
    positions = model.decode_positions.get(token.shape[0])
    if positions is None:
        positions = torch.ones(token.shape[0], 1, device="cuda", dtype=torch.long)
        model.decode_positions[token.shape[0]] = positions
    for layer, temporal, cache in zip(model.shell, model.temporal, caches):
        normalized = rms_norm(x, layer.norm_temporal)
        if model.kind == "samu":
            mixed, next_cache = samu_triton_decode_split(normalized, cache, temporal)
        else:
            mixed, next_cache = rglru_triton_serial(
                normalized[:, None], positions, temporal, cache
            )
            mixed = mixed[:, 0]
        x = (x + mixed).to(torch.bfloat16)
        x = (x + mlp(rms_norm(x, layer.norm_mlp), layer)).to(torch.bfloat16)
        next_caches.append(next_cache)
    x = rms_norm(x, model.final_norm)
    logits = x @ model.embedding.T
    next_ids = logits.argmax(dim=-1)
    return model.embedding[next_ids], next_caches


def prepare_prompt(model: SystemModel, batch: int, prompt_length: int, seed: int):
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


def warm_decode(model: SystemModel, token: torch.Tensor, caches, steps: int = 2) -> None:
    local_token, local_caches = token, caches
    for _ in range(steps):
        local_token, local_caches = decode_step(model, local_token, local_caches)
    torch.cuda.synchronize()


def timed_trajectory(model: SystemModel, token: torch.Tensor, caches,
                     checkpoints: list[int], repeats: int) -> dict[int, list[float]]:
    checkpoints = sorted(checkpoints)
    result = {checkpoint: [] for checkpoint in checkpoints}
    warm_decode(model, token, caches)
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


def summarize_latency(model: SystemModel, prompt: int, samples: dict[int, list[float]],
                      prefill_ms: float, batch: int) -> list[dict[str, Any]]:
    rows = []
    for decoded, values in samples.items():
        median = statistics.median(values)
        rows.append({
            "model": model.kind,
            "workload": "continuous_decode_latency",
            "batch": batch,
            "prompt_length": prompt,
            "decode_length": decoded,
            "median_ms": median,
            "p10_ms": percentile(values, .10),
            "p95_ms": percentile(values, .95),
            "raw_samples_ms": values,
            "tokens_per_second": batch * decoded * 1000.0 / median,
            "prompt_prefill_ms_excluded_from_decode": prefill_ms,
        })
    return rows


def throughput_probe(model: SystemModel, batch: int, steps: int, seed: int) -> dict[str, Any]:
    token, caches, _ = prepare_prompt(model, batch, 0, seed)
    samples = timed_trajectory(model, token, caches, [steps], repeats=1)[steps]
    latency = samples[0]
    return {"batch": batch, "probe_steps": steps, "latency_ms": latency,
            "tokens_per_second": batch * steps * 1000.0 / latency}


def run_model(model: SystemModel, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latency_rows = []
    for prompt in args.prompt_lengths:
        print(f"{model.kind}: latency prompt={prompt}", flush=True)
        token, caches, prefill_ms = prepare_prompt(model, args.latency_batch, prompt, args.seed)
        samples = timed_trajectory(
            model, token, caches, args.decode_lengths, args.latency_repeats
        )
        latency_rows.extend(summarize_latency(
            model, prompt, samples, prefill_ms, args.latency_batch
        ))
        del token, caches
        torch.cuda.empty_cache()

    search = []
    for batch in args.batch_candidates:
        print(f"{model.kind}: throughput probe batch={batch}", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            row = throughput_probe(model, batch, args.probe_steps, args.seed)
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["status"] = "measured"
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch, "probe_steps": args.probe_steps, "status": "oom"}
            torch.cuda.empty_cache()
        except RuntimeError as error:
            # A two-dimensional Triton grid uses batch as grid-y. CUDA limits
            # that dimension to 65,535 launches, so record the boundary rather
            # than aborting and losing all preceding search measurements.
            if "invalid argument" not in str(error).lower():
                raise
            row = {
                "batch": batch,
                "probe_steps": args.probe_steps,
                "status": "unsupported_launch_grid",
                "reason": str(error),
            }
            torch.cuda.empty_cache()
        search.append(row)
    measured = [row for row in search if row["status"] == "measured"]
    measured.sort(key=lambda row: row["tokens_per_second"], reverse=True)
    finalists = sorted({row["batch"] for row in measured[:args.throughput_finalists]})
    throughput_rows = []
    for batch in finalists:
        print(f"{model.kind}: full throughput trajectory batch={batch}", flush=True)
        token, caches, _ = prepare_prompt(model, batch, 0, args.seed)
        samples = timed_trajectory(
            model, token, caches, args.throughput_lengths, args.throughput_repeats
        )
        for decoded, values in samples.items():
            median = statistics.median(values)
            throughput_rows.append({
                "model": model.kind,
                "workload": "maximum_throughput_candidate",
                "batch": batch,
                "prompt_length": 0,
                "decode_length": decoded,
                "median_ms": median,
                "p10_ms": percentile(values, .10),
                "p95_ms": percentile(values, .95),
                "raw_samples_ms": values,
                "tokens_per_second": batch * decoded * 1000.0 / median,
            })
    winners = []
    for decoded in args.throughput_lengths:
        candidates = [row for row in throughput_rows if row["decode_length"] == decoded]
        if candidates:
            best = max(candidates, key=lambda row: row["tokens_per_second"])
            winners.append({**best, "workload": "maximum_throughput"})
    return latency_rows + throughput_rows + winners, search


def correctness(rglru_source: Path, width: int, seed: int) -> dict[str, Any]:
    """Fail-closed numerical audit of the exact full-system temporal paths."""

    modes = width // 2
    params = make_samu_parameters(width, modes, "cuda", torch.bfloat16, seed=seed + 911)
    packed = pack_samu_parameters(params)
    token = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
    state = (torch.randn(3, modes, device="cuda"), torch.randn(3, modes, device="cuda"))
    expected, expected_cache = samu_packed_reference(
        token[:, None], params, packed, initial=state, return_cache=True
    )
    actual, actual_cache = samu_triton_decode_split(token, state, packed)

    RGLRU = load_official_rglru(rglru_source)
    torch.manual_seed(seed + 1907)
    official_rg = RGLRU(
        width=width, num_heads=2, device="cuda", dtype=torch.bfloat16
    ).eval()
    packed_rg = pack_rglru(official_rg)
    rg_token = torch.randn(3, 1, width, device="cuda", dtype=torch.bfloat16)
    rg_state = torch.randn(3, width, device="cuda", dtype=torch.float32)
    rg_positions = torch.tensor([[0], [1], [7]], device="cuda", dtype=torch.long)
    rg_expected, rg_expected_cache = official_rg(
        rg_token, rg_positions, cache=rg_state, return_cache=True
    )
    rg_actual, rg_actual_cache = rglru_triton_serial(
        rg_token, rg_positions, packed_rg, rg_state
    )

    audit = {
        "audit_width": width,
        "rglru_reference": "unmodified recurrentgemma.torch.layers.RGLRU",
        "rglru_reference_commit": RGLRU_COMMIT,
        "rglru_cases": ["document reset", "ordinary position 1", "ordinary position 7"],
        "rglru_split_decode_output_max_abs": float(
            (rg_actual - rg_expected).abs().max()
        ),
        "rglru_split_decode_cache_max_abs": float(
            (rg_actual_cache - rg_expected_cache).abs().max()
        ),
        "samu_split_decode_output_max_abs": float(
            (actual - expected[:, 0].reshape(3, width)).abs().max()
        ),
        "samu_split_decode_cache_max_abs": float(max(
            (actual_cache[0] - expected_cache[0]).abs().max(),
            (actual_cache[1] - expected_cache[1]).abs().max(),
        )),
    }
    limits = {
        "rglru_split_decode_output_max_abs": 2e-3,
        "rglru_split_decode_cache_max_abs": 5e-5,
        "samu_split_decode_output_max_abs": 1e-2,
        "samu_split_decode_cache_max_abs": 1e-4,
    }
    failures = {
        name: {"measured": audit[name], "limit": limit}
        for name, limit in limits.items()
        if audit[name] > limit
    }
    audit["acceptance_limits"] = limits
    audit["passed"] = not failures
    audit["failures"] = failures
    if failures:
        raise RuntimeError(f"full-system correctness acceptance failed: {json.dumps(failures, sort_keys=True)}")
    return audit


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../benchmark_results_griffin_section5/summary.json"))
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--expansion", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--models", choices=("both", "samu", "rglru"), default="both")
    parser.add_argument(
        "--model-order", type=str, default="samu,rglru",
        help="comma-separated execution order when --models=both; use the reverse order in a counterbalanced run",
    )
    parser.add_argument("--latency-batch", type=int, default=16)
    parser.add_argument("--prompt-lengths", type=parse_ints, default=parse_ints("0,4096"))
    parser.add_argument("--decode-lengths", type=parse_ints, default=parse_ints("128,256,512,1024,2048,4096"))
    parser.add_argument("--throughput-lengths", type=parse_ints, default=parse_ints("512,1024,2048,4096"))
    parser.add_argument("--batch-candidates", type=parse_ints, default=parse_ints("1,4,16,32,64,96,128,192,256"))
    parser.add_argument("--probe-steps", type=int, default=32)
    parser.add_argument("--throughput-finalists", type=int, default=2)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--throughput-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.width % 2:
        raise ValueError("width must be even so SAMU has equal real-state bytes")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    env = environment()
    audit = correctness(args.rglru_source, args.width, args.seed)
    stabilization = stabilize_gpu()
    print("stabilization", json.dumps(stabilization), flush=True)
    if args.models == "both":
        kinds = [item for item in args.model_order.split(",") if item]
        if len(kinds) != 2 or set(kinds) != {"samu", "rglru"}:
            raise ValueError("--model-order must contain samu,rglru exactly once each")
    else:
        kinds = [args.models]
    all_rows, searches, model_metadata = [], {}, {}
    for kind in kinds:
        print(f"building {kind} model", flush=True)
        model = build_model(
            kind, args.width, args.depth, args.expansion, args.vocab,
            args.rglru_source, args.seed,
        )
        model_metadata[kind] = {
            "parameter_count": model.parameter_count,
            "parameter_bytes_bf16": model.parameter_count * 2,
            "state_bytes_per_sequence": model.state_bytes_per_sequence,
            "dynamic_control_values_per_token": 2 if kind == "samu" else 2 * args.width,
            "temporal_backend": (
                "packed BF16 GEMM + SAMU Triton recurrence"
                if kind == "samu"
                else "two-head packed BF16 block GEMM + official-equation RG-LRU Triton recurrence"
            ),
            # Projection and recurrence are two logical operators.  The actual
            # CUDA launch count is selected by cuBLAS and can include an extra
            # split-K reduction for particular batch shapes; it is profiled
            # separately instead of being asserted here.
            "decode_logical_operators_per_layer": 2,
            "rglru_num_heads": 2 if kind == "rglru" else None,
        }
        print(json.dumps(model_metadata[kind]), flush=True)
        rows, search = run_model(model, args)
        all_rows.extend(rows)
        searches[kind] = search
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "status": "complete",
        "title": "Griffin Section 5 inspired SAMU vs RG-LRU systems analysis",
        "scope": "random-weight 1B-scale inference systems proxy; not a trained checkpoint or quality comparison",
        "paper_protocol": {
            "source": "Griffin arXiv:2402.19427v1 Section 5",
            "latency_batch": args.latency_batch,
            "prompt_lengths": args.prompt_lengths,
            "decode_lengths": args.decode_lengths,
            "throughput_lengths": args.throughput_lengths,
            "decode_time_model": "(parameter bytes + batch * recurrent-cache bytes) / effective memory bandwidth",
        },
        "fairness": {
            "shared": "decoder shell, width, depth, MLP expansion, vocabulary, BF16 parameter policy, FP32 state bytes, prompts, sampler, timing, and GPU",
            "changed_only": "SAMU recurrence versus official-equation RG-LRU recurrence",
            "rglru_head_choice": "two block-diagonal heads exactly match SAMU temporal learned-parameter count and BF16 projection multiply-add count (four scalar parameters apart per layer); this is not a RecurrentGemma model preset",
            "important_boundary": "Both methods are recurrent, so neither has a sequence-growing KV cache; prompt-length invariance is expected and tested.",
        },
        "configuration": vars(args) | {"output": str(args.output), "rglru_source": str(args.rglru_source)},
        "environment": env,
        "gpu_stabilization": stabilization,
        "correctness": audit,
        "models": model_metadata,
        "throughput_batch_search": searches,
        "rows": all_rows,
    }
    atomic_json(args.output.resolve(), result)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
