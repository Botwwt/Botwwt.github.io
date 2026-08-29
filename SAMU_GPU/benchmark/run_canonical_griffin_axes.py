"""Canonical SAMU versus official RG-LRU on the Griffin Section 4/5 axes.

This benchmark deliberately separates two questions that the Griffin paper
also separates:

1. Section 4.2 / Appendix D.2: how expensive is the recurrence after the
   projection?  We use the paper's B=8, state width 1024 and L=2K..16K scan
   shape.  Both methods receive their projected dynamic values.
2. Section 5: how expensive is one recurrent layer during continuous decode?
   We use the paper's 1.3B RNN width D_RNN=2560, equal FP32 state bytes and the
   paper-default 16 RG-LRU gate groups.  Both methods include every projection
   required by their own canonical equation.

The second track is a recurrent-layer experiment, not a trained Griffin/SAMU
language model.  There is no canonical SAMU Griffin block or trained checkpoint
in this repository, so the script refuses to label this as full-model latency.
No grouped SAMU, direct-control candidate, bounded polynomial, rho Taylor path,
or compressed-transition prototype is used in either primary track.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

from kernels import load_official_rglru, make_samu_parameters
from run_equal_kernel_benchmarks import measure, stabilize_gpu
from run_training_on_device import RGLRUInputs, SamuInputs
from triton_rglru import (
    pack_rglru,
    rglru_triton_auto,
    rglru_triton_decode,
    rglru_triton_serial,
)
from triton_samu import (
    pack_samu_parameters,
    samu_packed_reference,
    samu_triton_auto,
    samu_triton_decode_split,
)


RGLRU_COMMIT = "2efa84dac0e68e63547a27a18fa943c98f1c312e"
SCAN_LENGTHS = (2048, 4096, 8192, 16384)
DECODE_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
THROUGHPUT_LENGTHS = (512, 1024, 2048, 4096)
BATCH_CANDIDATES = (1, 4, 16, 32, 64, 96, 128, 192, 256)


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
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=20
        ).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    import triton

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "driver_and_board": command_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ]),
        "rglru_commit": RGLRU_COMMIT,
        "hardware_counters": {
            "dram_bytes": None,
            "l2_bytes": None,
            "sfu_instructions": None,
            "achieved_occupancy": None,
            "reason": "ERR_NVGPUCTRPERM: host disables NVIDIA performance counters",
        },
    }


def correctness(RGLRU, width: int = 2560) -> dict[str, Any]:
    """Fail closed against both public equation implementations."""

    torch.manual_seed(8061)
    batch = 3
    modes = width // 2
    samu_parameters = make_samu_parameters(
        width, modes, "cuda", torch.bfloat16, seed=8061
    )
    samu_packed = pack_samu_parameters(
        samu_parameters,
        enable_bounded_poly=False,
        enable_rho_taylor=False,
    )
    token = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
    samu_state = (
        torch.randn(batch, modes, device="cuda", dtype=torch.float32),
        torch.randn(batch, modes, device="cuda", dtype=torch.float32),
    )
    expected, expected_cache = samu_packed_reference(
        token[:, None], samu_parameters, samu_packed,
        initial=samu_state, return_cache=True,
    )
    actual, actual_cache = samu_triton_decode_split(token, samu_state, samu_packed)

    torch.manual_seed(9061)
    official = RGLRU(
        width=width, num_heads=16, device="cuda", dtype=torch.bfloat16
    ).eval()
    rg_packed = pack_rglru(official)
    rg_token = torch.randn(batch, 1, width, device="cuda", dtype=torch.bfloat16)
    rg_state = torch.randn(batch, width, device="cuda", dtype=torch.float32)
    positions = torch.tensor([[0], [1], [9]], device="cuda", dtype=torch.long)
    rg_expected, rg_expected_cache = official(
        rg_token, positions, cache=rg_state, return_cache=True
    )
    rg_actual, rg_actual_cache = rglru_triton_decode(
        rg_token, positions, rg_packed, rg_state
    )
    rg_split, rg_split_cache = rglru_triton_serial(
        rg_token, positions, rg_packed, rg_state
    )

    audit = {
        "samu_reference": "canonical G=1 PyTorch equation, generic exp/sin/cos",
        "samu_output_max_abs": float((actual - expected[:, 0].reshape(batch, width)).abs().max()),
        "samu_cache_max_abs": float(max(
            (actual_cache[0] - expected_cache[0]).abs().max(),
            (actual_cache[1] - expected_cache[1]).abs().max(),
        )),
        "rglru_reference": "unmodified recurrentgemma.torch.layers.RGLRU",
        "rglru_reference_commit": RGLRU_COMMIT,
        "rglru_cases": ["segment reset", "ordinary position 1", "ordinary position 9"],
        "rglru_output_max_abs": float((rg_actual - rg_expected).abs().max()),
        "rglru_cache_max_abs": float((rg_actual_cache - rg_expected_cache).abs().max()),
        "rglru_split_output_max_abs": float((rg_split - rg_expected).abs().max()),
        "rglru_split_cache_max_abs": float((rg_split_cache - rg_expected_cache).abs().max()),
    }
    limits = {
        "samu_output_max_abs": 0.015625,
        "samu_cache_max_abs": 5e-4,
        "rglru_output_max_abs": 0.015625,
        "rglru_cache_max_abs": 5e-4,
        "rglru_split_output_max_abs": 0.015625,
        "rglru_split_cache_max_abs": 5e-4,
    }
    audit["limits"] = limits
    audit["passed"] = all(audit[key] <= value for key, value in limits.items())
    if not audit["passed"]:
        raise RuntimeError(f"canonical correctness audit failed: {audit}")
    del official
    return audit


def run_scan_track(RGLRU, rep_ms: int) -> dict[str, Any]:
    """The exact Griffin Figure 8(a) shape, adapted to one H800."""

    rows: list[dict[str, Any]] = []
    for length in SCAN_LENGTHS:
        samu = SamuInputs(8, length, 1024)
        rg = RGLRUInputs(8, length, 1024, RGLRU)
        # The primary methods are exact generic-math paths only.
        operations: dict[str, Callable[[], Any]] = {
            "samu_serial_exact": samu.serial(4),
            "samu_chunk32_exact": samu.chunk(
                32, 2, compressed_p=False, rho_taylor=False
            ),
            "rglru_serial_official16": rg.serial(4),
            "rglru_chunk32_official16": rg.chunk(32, 4),
        }
        for order, names in (
            ("forward", tuple(operations)),
            ("reverse", tuple(reversed(tuple(operations)))),
        ):
            stabilize_gpu(seconds=0.5)
            for name in names:
                result, compile_seconds = measure(
                    operations[name], warmup_ms=20, rep_ms=rep_ms
                )
                rows.append({
                    "method": name,
                    "batch": 8,
                    "length": length,
                    "real_state_scalars": 1024,
                    "measurement_order": order,
                    "scope": "post-projection forward recurrence",
                    "median_ms": result["median_ms"],
                    "p10_ms": result["p10_ms"],
                    "p90_ms": result["p90_ms"],
                    "p95_ms": result["p95_ms"],
                    "raw_samples_ms": result["raw_samples_ms"],
                    "compile_seconds": compile_seconds,
                })
        del samu, rg, operations
        torch.cuda.empty_cache()

    summary = []
    for method in sorted({row["method"] for row in rows}):
        for length in SCAN_LENGTHS:
            values = [
                row["median_ms"] for row in rows
                if row["method"] == method and row["length"] == length
            ]
            summary.append({
                "method": method,
                "length": length,
                "order_balanced_median_ms": statistics.mean(values),
                "orders": len(values),
            })
    return {
        "paper_mapping": {
            "source": "Griffin Section 4.2 and Appendix D.2, Figure 8(a)",
            "matched": "batch 8, 1024 real state scalars, sequence lengths 2K/4K/8K/16K",
            "changed": "one NVIDIA H800 PCIe instead of a two-chip TPU-v3 pod; SAMU versus RG-LRU instead of multiple Hawk scan backends",
            "timed_scope": "projection excluded, forward recurrence only",
        },
        "excluded_primary_variants": [
            "grouped SAMU",
            "direct-control SAMU",
            "bounded polynomial",
            "rho Taylor approximation",
            "compressed transition",
        ],
        "rows": rows,
        "summary": summary,
    }


class DecodeLayer:
    def __init__(self, kind: str, width: int, RGLRU):
        self.kind = kind
        self.width = width
        self.modes = width // 2
        if kind == "samu_canonical":
            self.source = make_samu_parameters(
                width, self.modes, "cuda", torch.bfloat16, seed=10061
            )
            self.packed = pack_samu_parameters(
                self.source,
                enable_bounded_poly=False,
                enable_rho_taylor=False,
            )
            self.learned_parameters = sum(
                tensor.numel() for tensor in self.source.__dict__.values()
            )
            self.projection_parameters = self.packed.weight.numel()
            self.launches_per_step = 2
        elif kind == "rglru_official16":
            torch.manual_seed(11061)
            self.source = RGLRU(
                width=width, num_heads=16, device="cuda", dtype=torch.bfloat16
            ).eval()
            self.packed = pack_rglru(self.source)
            self.learned_parameters = sum(
                parameter.numel() for parameter in self.source.parameters()
            )
            self.projection_parameters = self.packed.gate_weight.numel()
            self.launches_per_step = 1
        else:
            raise ValueError(kind)
        self.backend_by_batch: dict[int, str] = {}
        self.backend_calibration_ms: dict[int, dict[str, float]] = {}

    def zero_state(self, batch: int):
        if self.kind == "samu_canonical":
            return (
                torch.zeros(batch, self.modes, device="cuda", dtype=torch.float32),
                torch.zeros(batch, self.modes, device="cuda", dtype=torch.float32),
            )
        return torch.zeros(batch, self.width, device="cuda", dtype=torch.float32)

    def prefill(self, values: torch.Tensor):
        batch, length, _ = values.shape
        if self.kind == "samu_canonical":
            output, state = samu_triton_auto(
                values, self.source, self.packed,
                compressed_p=False, return_cache=True,
            )
            return output[:, -1].reshape(batch, self.width), state
        positions = torch.arange(length, device="cuda", dtype=torch.long)[None].expand(batch, -1)
        output, state = rglru_triton_auto(values, positions, self.packed)
        return output[:, -1], state

    def _rg_fused_step(self, values: torch.Tensor, state):
        positions = torch.ones(values.shape[0], device="cuda", dtype=torch.long)
        output, next_state = rglru_triton_decode(values, positions, self.packed, state)
        return output[:, 0], next_state

    def _rg_split_step(self, values: torch.Tensor, state):
        positions = torch.ones(values.shape[0], 1, device="cuda", dtype=torch.long)
        output, next_state = rglru_triton_serial(
            values[:, None], positions, self.packed, state
        )
        return output[:, 0], next_state

    def _select_rg_backend(self, values: torch.Tensor, state) -> str:
        batch = values.shape[0]
        if batch in self.backend_by_batch:
            return self.backend_by_batch[batch]
        candidates = {
            "fused_dot_one_launch": lambda: self._rg_fused_step(values, state),
            "block_bmm_plus_update_two_launches": lambda: self._rg_split_step(values, state),
        }
        timings: dict[str, float] = {}
        for name, operation in candidates.items():
            result, _ = measure(operation, warmup_ms=15, rep_ms=40)
            timings[name] = result["median_ms"]
        selected = min(timings, key=timings.get)
        self.backend_by_batch[batch] = selected
        self.backend_calibration_ms[batch] = timings
        return selected

    def step(self, values: torch.Tensor, state):
        if self.kind == "samu_canonical":
            return samu_triton_decode_split(values, state, self.packed)
        backend = self._select_rg_backend(values, state)
        if backend == "block_bmm_plus_update_two_launches":
            return self._rg_split_step(values, state)
        return self._rg_fused_step(values, state)

    def logical_accounting(self) -> dict[str, Any]:
        state_bytes = self.width * 4
        parameter_bytes = self.learned_parameters * 2
        if self.kind == "samu_canonical":
            dynamic_nonlinearities = {
                "token_shared_tanh": 2,
                "per_real_state_exp_equivalent": self.width // 2,
                "per_complex_mode_sincos": self.width // 2,
            }
        else:
            dynamic_nonlinearities = {
                "per_real_state_sigmoid": 2 * self.width,
                "per_real_state_exp": 2 * self.width,
                "per_real_state_sqrt": self.width,
            }
        return {
            "learned_temporal_parameters": self.learned_parameters,
            "bf16_parameter_bytes": parameter_bytes,
            "fp32_state_bytes_per_sequence": state_bytes,
            "projection_weight_scalars": self.projection_parameters,
            "cuda_launches_per_decode_step": (
                self.launches_per_step if self.kind == "samu_canonical"
                else "1 for fused dot; 2 for block BMM + update"
            ),
            "selected_backend_by_batch": self.backend_by_batch,
            "backend_calibration_median_ms": self.backend_calibration_ms,
            "equation_level_dynamic_nonlinearities_per_token": dynamic_nonlinearities,
            "note": "operation counts are equation-level accounting, not hardware counter measurements",
        }


def timed_trajectory(layer: DecodeLayer, values: torch.Tensor, state,
                     checkpoints: tuple[int, ...], repeats: int) -> dict[int, list[float]]:
    checkpoints = tuple(sorted(checkpoints))
    # Compile and warm the exact path before recording events.
    warm_value, warm_state = values, state
    for _ in range(4):
        warm_value, warm_state = layer.step(warm_value, warm_state)
    torch.cuda.synchronize()
    results = {checkpoint: [] for checkpoint in checkpoints}
    for _ in range(repeats):
        current, current_state = values, state
        start = torch.cuda.Event(enable_timing=True)
        ends = {checkpoint: torch.cuda.Event(enable_timing=True) for checkpoint in checkpoints}
        start.record()
        for step in range(1, checkpoints[-1] + 1):
            current, current_state = layer.step(current, current_state)
            if step in ends:
                ends[step].record()
        ends[checkpoints[-1]].synchronize()
        for checkpoint in checkpoints:
            results[checkpoint].append(float(start.elapsed_time(ends[checkpoint])))
    return results


def make_decode_start(layer: DecodeLayer, batch: int, prompt_length: int,
                      seed: int) -> tuple[torch.Tensor, Any, float]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + batch * 101 + prompt_length)
    if prompt_length == 0:
        return (
            torch.randn(batch, layer.width, device="cuda", dtype=torch.bfloat16, generator=generator),
            layer.zero_state(batch),
            0.0,
        )
    # Section 5 excludes prompt processing from decode latency.  Construct the
    # 4K-history state with the already-audited one-token path instead of using
    # the width-2560 chunk prefill kernel: that kernel is not part of this
    # decode claim and currently fails closed at this shape on Triton 3.5.0.
    # The recurrence state is mathematically identical to processing the same
    # prompt in one call, while the construction time remains explicitly
    # outside the reported generation latency.
    prompt = torch.randn(
        prompt_length, batch, layer.width,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    state = layer.zero_state(batch)
    value = prompt[0]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(prompt_length):
        value, state = layer.step(prompt[index], state)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    del prompt
    return value, state, elapsed


def summarize_trajectory(kind: str, workload: str, batch: int,
                         prompt_length: int, samples: dict[int, list[float]],
                         prefill_ms: float) -> list[dict[str, Any]]:
    rows = []
    for decoded, values in samples.items():
        median = statistics.median(values)
        rows.append({
            "model": kind,
            "workload": workload,
            "batch": batch,
            "prompt_length": prompt_length,
            "decode_length": decoded,
            "median_ms": median,
            "p10_ms": percentile(values, .10),
            "p95_ms": percentile(values, .95),
            "raw_samples_ms": values,
            "tokens_per_second": batch * decoded * 1000.0 / median,
            "prompt_prefill_ms_excluded_from_decode": prefill_ms,
        })
    return rows


def run_decode_track(RGLRU, width: int, latency_repeats: int,
                     throughput_repeats: int) -> dict[str, Any]:
    layers = {
        kind: DecodeLayer(kind, width, RGLRU)
        for kind in ("samu_canonical", "rglru_official16")
    }
    rows: list[dict[str, Any]] = []
    searches: dict[str, list[dict[str, Any]]] = {}

    for order_name, order in (
        ("AB", ("samu_canonical", "rglru_official16")),
        ("BA", ("rglru_official16", "samu_canonical")),
    ):
        stabilize_gpu(seconds=1.0)
        for kind in order:
            layer = layers[kind]
            for prompt_length in (0, 4096):
                value, state, prefill_ms = make_decode_start(
                    layer, 16, prompt_length, seed=12061
                )
                samples = timed_trajectory(
                    layer, value, state, DECODE_LENGTHS, latency_repeats
                )
                for row in summarize_trajectory(
                    kind, "continuous_decode_latency", 16,
                    prompt_length, samples, prefill_ms,
                ):
                    rows.append({**row, "measurement_order": order_name})
                del value, state
                torch.cuda.empty_cache()

    # A predeclared finite search followed by full-trajectory confirmation.
    # Search once per method, then test the union of both methods' top two
    # batches so neither method receives a more favorable candidate set.
    finalists: set[int] = set()
    for kind, layer in layers.items():
        searches[kind] = []
        for batch in BATCH_CANDIDATES:
            try:
                value, state, _ = make_decode_start(layer, batch, 0, seed=13061)
                sample = timed_trajectory(layer, value, state, (64,), 1)[64][0]
                result = {
                    "batch": batch,
                    "status": "measured",
                    "probe_steps": 64,
                    "latency_ms": sample,
                    "tokens_per_second": batch * 64 * 1000.0 / sample,
                }
                del value, state
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                result = {"batch": batch, "status": "unsupported", "reason": repr(error)}
                torch.cuda.empty_cache()
            searches[kind].append(result)
        measured = sorted(
            (row for row in searches[kind] if row["status"] == "measured"),
            key=lambda row: row["tokens_per_second"], reverse=True,
        )
        finalists.update(row["batch"] for row in measured[:2])

    for order_name, order in (
        ("AB", ("samu_canonical", "rglru_official16")),
        ("BA", ("rglru_official16", "samu_canonical")),
    ):
        stabilize_gpu(seconds=1.0)
        for kind in order:
            layer = layers[kind]
            for batch in sorted(finalists):
                value, state, _ = make_decode_start(layer, batch, 0, seed=14061)
                samples = timed_trajectory(
                    layer, value, state, THROUGHPUT_LENGTHS, throughput_repeats
                )
                for row in summarize_trajectory(
                    kind, "maximum_throughput_candidate", batch, 0, samples, 0.0
                ):
                    rows.append({**row, "measurement_order": order_name})
                del value, state
                torch.cuda.empty_cache()

    return {
        "paper_mapping": {
            "source": "Griffin Section 5 and Appendix F",
            "matched": "D_RNN=2560 from the paper's 1.3B configuration; B=16; empty/4K prompt; decode 128..4096; throughput lengths 512..4096",
            "scope": "one recurrent layer including its canonical dynamic projections and state update",
            "not_full_model": "no canonical trained SAMU Griffin block or checkpoint exists; embedding, Conv1D, y branch, output projection, MLP and sampler are intentionally not invented here",
        },
        "fairness": {
            "state": f"both methods store {width} FP32 real scalars per sequence",
            "precision": "BF16 inputs/outputs and projections; FP32 recurrent cache",
            "rglru": "unmodified official equation, 16 block-diagonal gate groups, reset semantics included in correctness audit",
            "samu": "canonical coherent G=1 equation; dense complex write; normalized affine selectors; generic exp/sin/cos only",
            "timing": "steady-state CUDA events; prompt prefill excluded from decode latency; AB/BA order; complete trajectories",
        },
        "measurement_protocol": {
            "latency_repeats_per_order": latency_repeats,
            "throughput_repeats_per_order": throughput_repeats,
            "throughput_summary": "median within each order, then equal-weight mean of AB and BA order medians",
            "batch_probe_steps": 64,
        },
        "resource_accounting": {
            kind: layer.logical_accounting() for kind, layer in layers.items()
        },
        "batch_search": searches,
        "finalist_union": sorted(finalists),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--scan-rep-ms", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--throughput-repeats", type=int, default=5)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    RGLRU = load_official_rglru(args.rglru_source)
    result: dict[str, Any] = {
        "schema_version": 1,
        "title": "Canonical SAMU versus official RG-LRU on Griffin Section 4/5 axes",
        "environment": environment(),
        "correctness": correctness(RGLRU),
        "evidence_boundary": {
            "architecture_changes_in_primary_results": False,
            "trained_model_quality_measured": False,
            "full_training_step_measured": False,
            "reason_full_training_missing": "the custom Triton recurrence kernels do not yet implement backward; forward timing is not renamed as training throughput",
            "multi_device_measured": False,
            "reason_multi_device_missing": "the rented system exposes one H800",
        },
    }
    if not args.skip_scan:
        result["training_scan"] = run_scan_track(RGLRU, args.scan_rep_ms)
        atomic_json(args.output, result)
    if not args.skip_decode:
        result["inference_recurrent_layer"] = run_decode_track(
            RGLRU, width=2560,
            latency_repeats=args.latency_repeats,
            throughput_repeats=args.throughput_repeats,
        )
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
