"""Measured H800 roofline and Griffin Appendix F decode accounting.

The script measures large on-device copy bandwidth and BF16 matrix-multiply
throughput, derives the measured ridge point, and combines it with complete
trajectory decode measurements.  It does not claim DRAM/L2 instruction counts:
the rented host disables NVIDIA performance counters.  All reported effective
bandwidth values are therefore explicit byte-model estimates divided by
measured wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from run_paper_scale_h800 import PAPER_SCALES, exact_parameter_count
from run_small_model_study import StudyConfig


MIB = 2**20
GIB = 2**30


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def event_ms(function, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def bandwidth_probe(mebibytes: int) -> dict:
    elements = mebibytes * MIB // torch.tensor([], dtype=torch.bfloat16).element_size()
    source = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    samples = event_ms(lambda: destination.copy_(source), warmup=5, repetitions=25)
    median_ms = statistics.median(samples)
    # One device-to-device copy reads and writes the complete tensor.
    transferred = 2 * source.numel() * source.element_size()
    result = {
        "tensor_bytes": source.numel() * source.element_size(),
        "logical_read_plus_write_bytes": transferred,
        "median_ms": median_ms,
        "median_gb_per_second": transferred / (median_ms / 1000.0) / 1e9,
        "samples_ms": samples,
    }
    del source, destination
    torch.cuda.empty_cache()
    return result


def gemm_probe(dimension: int) -> dict:
    left = torch.randn(dimension, dimension, device="cuda", dtype=torch.bfloat16)
    right = torch.randn_like(left)
    output = None

    def multiply():
        nonlocal output
        output = left @ right

    samples = event_ms(multiply, warmup=5, repetitions=20)
    median_ms = statistics.median(samples)
    flops = 2 * dimension**3
    result = {
        "m": dimension,
        "n": dimension,
        "k": dimension,
        "flops": flops,
        "median_ms": median_ms,
        "median_tflops_per_second": flops / (median_ms / 1000.0) / 1e12,
        "samples_ms": samples,
    }
    del left, right, output
    torch.cuda.empty_cache()
    return result


def recurrent_cache_bytes(config: StudyConfig) -> dict:
    recurrent = config.depth * config.rnn_width * 4
    convolution = (
        config.depth
        * (config.conv_width - 1)
        * config.rnn_width
        * 2
    )
    return {
        "fp32_recurrent_state_bytes_per_sequence": recurrent,
        "bf16_convolution_cache_bytes_per_sequence": convolution,
        "total_bytes_per_sequence": recurrent + convolution,
    }


def best_throughput(rows: list[dict], generated_tokens: int) -> dict:
    candidates = [row for row in rows if row["generated_tokens"] == generated_tokens]
    return max(candidates, key=lambda row: row["median_tokens_per_second"])


def decode_accounting(
    training: dict,
    benchmark: dict,
    config: StudyConfig,
) -> dict:
    cache = recurrent_cache_bytes(config)
    output = {}
    for architecture in ("samu", "rglru"):
        parameters = training["aggregate"][architecture]["parameters"]
        bf16_compute_weight_bytes = 2 * parameters
        fp32_master_parameter_bytes = 4 * parameters
        latency = []
        for row in benchmark[architecture]["decode"]["latency"]:
            step_seconds = row["median_trajectory_ms"] / 1000.0 / row["generated_tokens"]
            batch = row["batch_size"]
            bf16_bytes = bf16_compute_weight_bytes + batch * cache["total_bytes_per_sequence"]
            fp32_bytes = fp32_master_parameter_bytes + batch * cache["total_bytes_per_sequence"]
            latency.append({
                **row,
                "median_ms_per_generated_token": 1000.0 * step_seconds,
                "bf16_weight_plus_cache_bytes_per_step": bf16_bytes,
                "fp32_master_plus_cache_bytes_per_step": fp32_bytes,
                "bf16_byte_model_effective_gb_per_second": bf16_bytes / step_seconds / 1e9,
                "fp32_byte_model_effective_gb_per_second": fp32_bytes / step_seconds / 1e9,
            })
        throughput = []
        for generated_tokens in (512, 1024, 2048, 4096):
            row = best_throughput(
                benchmark[architecture]["decode"]["throughput"], generated_tokens
            )
            batch = row["batch_size"]
            step_seconds = batch / row["median_tokens_per_second"]
            bf16_bytes = bf16_compute_weight_bytes + batch * cache["total_bytes_per_sequence"]
            throughput.append({
                **row,
                "median_ms_per_generated_token_step": 1000.0 * step_seconds,
                "bf16_weight_plus_cache_bytes_per_step": bf16_bytes,
                "bf16_byte_model_effective_gb_per_second": bf16_bytes / step_seconds / 1e9,
            })
        output[architecture] = {
            "parameters": parameters,
            "fp32_master_parameter_bytes": fp32_master_parameter_bytes,
            "bf16_compute_weight_bytes": bf16_compute_weight_bytes,
            "cache": cache,
            "fixed_batch_latency": latency,
            "best_throughput": throughput,
        }
    return output


def scale_cache_table(vocab_size: int) -> list[dict]:
    rows = []
    for scale, dimensions in PAPER_SCALES:
        config = StudyConfig(vocab_size=vocab_size, **dimensions)
        cache = recurrent_cache_bytes(config)
        for architecture in ("samu", "rglru"):
            parameters = exact_parameter_count(config, architecture)
            rows.append({
                "scale": scale,
                "architecture": architecture,
                "config": {
                    "width": config.width,
                    "rnn_width": config.rnn_width,
                    "depth": config.depth,
                },
                "parameters": parameters,
                "bf16_compute_weight_bytes": 2 * parameters,
                "fp32_master_parameter_bytes": 4 * parameters,
                **cache,
                "cache_to_bf16_weight_ratio": (
                    cache["total_bytes_per_sequence"] / (2 * parameters)
                ),
            })
    return rows


def paper_scale_decode_accounting(inference: dict) -> dict:
    """Apply Appendix F's weight-plus-cache model to the 1.3B-shape run."""
    output = {}
    for architecture in ("samu", "rglru"):
        metadata = next(
            item for item in inference["measurement_passes"]
            if item["architecture"] == architecture
        )
        weight_bytes = metadata["bf16_parameter_bytes"]
        cache_bytes = (
            metadata["fp32_state_bytes_per_sequence"]
            + metadata["bf16_convolution_cache_bytes_per_sequence"]
        )
        fixed = []
        for row in inference["rows"]:
            if (
                row["architecture"] != architecture
                or row["workload"] != "continuous_decode_latency"
            ):
                continue
            seconds_per_step = row["median_ms"] / 1000.0 / row["decode_length"]
            logical_bytes = weight_bytes + row["batch_size"] * cache_bytes
            fixed.append({
                **row,
                "median_ms_per_generated_token": 1000.0 * seconds_per_step,
                "bf16_weight_plus_cache_bytes_per_step": logical_bytes,
                "bf16_byte_model_effective_gb_per_second": (
                    logical_bytes / seconds_per_step / 1e9
                ),
            })
        throughput = []
        for row in inference["rows"]:
            if (
                row["architecture"] != architecture
                or row["workload"] != "maximum_throughput"
            ):
                continue
            seconds_per_step = row["batch_size"] / row["tokens_per_second"]
            logical_bytes = weight_bytes + row["batch_size"] * cache_bytes
            throughput.append({
                **row,
                "median_ms_per_generated_token_step": 1000.0 * seconds_per_step,
                "bf16_weight_plus_cache_bytes_per_step": logical_bytes,
                "bf16_byte_model_effective_gb_per_second": (
                    logical_bytes / seconds_per_step / 1e9
                ),
            })
        output[architecture] = {
            "parameters": metadata["parameters"],
            "bf16_compute_weight_bytes": weight_bytes,
            "cache_bytes_per_sequence": cache_bytes,
            "fixed_batch_latency": fixed,
            "best_throughput": throughput,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--full-model-benchmark", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--paper-scale-inference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-mebibytes", type=int, default=512)
    parser.add_argument("--gemm-dimension", type=int, default=8192)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    benchmark = json.loads(args.full_model_benchmark.read_text(encoding="utf-8"))
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    paper_inference = (
        json.loads(args.paper_scale_inference.read_text(encoding="utf-8"))
        if args.paper_scale_inference else None
    )
    config_values = environment["config"]
    config = StudyConfig(**config_values)

    bandwidth = bandwidth_probe(args.copy_mebibytes)
    gemm = gemm_probe(args.gemm_dimension)
    ridge = (
        gemm["median_tflops_per_second"] * 1e12
        / (bandwidth["median_gb_per_second"] * 1e9)
    )
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_mapping": {
            "source": "Griffin Sections 4.2, 5.1 and Appendix D.1, F.1-F.4",
            "measured": [
                "large device-to-device BF16 copy bandwidth",
                "large BF16 GEMM throughput",
                "complete-trajectory decode latency and throughput",
            ],
            "derived_not_counter_measured": [
                "roofline ridge point",
                "weight-plus-cache byte-model effective bandwidth",
                "cache-to-weight ratios",
            ],
            "counter_limitation": (
                "the host disables NVIDIA DRAM/L2/SFU performance counters; "
                "derived byte models are not presented as profiler traffic"
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "measured_copy_bandwidth": bandwidth,
        "measured_bf16_gemm": gemm,
        "measured_roofline": {
            "ridge_flops_per_byte": ridge,
            "approximate_linear_outer_dimension_for_compute_bound": math.ceil(ridge),
            "interpretation": (
                "for D much larger than the outer dimension, Griffin Appendix D.1 "
                "approximates GEMM arithmetic intensity by that outer dimension"
            ),
        },
        "paper_recurrence_arithmetic_intensity": {
            "rglru_flops_per_real_state_update": 6,
            "rglru_bf16_read_plus_write_bytes": 8,
            "rglru_flops_per_byte": 0.75,
            "source": "Griffin Section 4.2",
        },
        "decode_accounting": decode_accounting(training, benchmark, config),
        "paper_scale_decode_accounting": (
            paper_scale_decode_accounting(paper_inference)
            if paper_inference else None
        ),
        "paper_scale_cache_accounting": scale_cache_table(
            paper_inference["config"]["vocab_size"]
            if paper_inference else config.vocab_size
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "copy_gb_s": bandwidth["median_gb_per_second"],
        "gemm_tflops_s": gemm["median_tflops_per_second"],
        "ridge_flops_per_byte": ridge,
    }, indent=2), flush=True)
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
