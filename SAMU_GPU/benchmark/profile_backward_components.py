"""CUDA-event attribution for selected mixer backward passes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from run_small_model_study import RGLRUMixer, SAMUMixer, set_seed


CASES = (
    (4, 2048, 2048, "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_chunk32"),
    (1, 8192, 2560, "fused_grouped_prefix32_hierarchical_chunk16", 256,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
    (1, 32768, 1024, "fused_grouped_prefix32_hierarchical_chunk16", 256,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
    (1, 32768, 2048, "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32"),
)


def category(name: str) -> str:
    if "reverse_chunk_replay" in name:
        return "reverse_replay"
    if "reverse_chunk_summary" in name:
        return "reverse_summary"
    if ("affine_prefix" in name or "affine_group" in name
            or "reverse_chunk_prefix" in name):
        return "reverse_prefix"
    if "reduce_shared_control" in name:
        return "shared_control_reduce"
    if "formal_coordinate" in name:
        return "formal_coordinate"
    if "controller_grad" in name:
        return "controller_projection"
    if "gemm" in name or "cublas" in name:
        return "gemm"
    if "Cat" in name or "cat" in name:
        return "cat"
    if "Memset" in name:
        return "memset"
    if "Memcpy" in name or "copy" in name:
        return "copy"
    if "reduce_kernel" in name:
        return "framework_reduce"
    if "elementwise" in name or "Elementwise" in name:
        return "framework_elementwise"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_seed(76401)
    rows = []
    for batch, length, width, rg_backend, rg_block, samu_backend in CASES:
        source = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16)
        cotangent = torch.randn_like(source)
        for architecture in ("rglru", "samu"):
            if architecture == "rglru":
                module = RGLRUMixer(width, 16).cuda().train()
                module.set_scan_backend(rg_backend)
                module.set_fused_training_block_size(rg_block)
                backend = rg_backend
            else:
                module = SAMUMixer(width).cuda().train()
                module.set_scan_backend(samu_backend)
                module.set_tiled_training_block_size(256)
                module.set_controller_projection_dtype("fp32_fused_coords")
                with torch.no_grad():
                    module.phase_amplitude.fill_(0.7)
                    module.radial_amplitude.fill_(0.5)
                backend = samu_backend

            def make_output():
                module.zero_grad(set_to_none=True)
                value = source.detach().clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(value)[0]
                return value, output

            value, output = make_output()
            (output.float() * cotangent.float()).sum().backward()
            torch.cuda.synchronize()
            value, output = make_output()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                (output.float() * cotangent.float()).sum().backward()
                torch.cuda.synchronize()
            by_name = defaultdict(lambda: [0, 0.0])
            by_category = defaultdict(lambda: [0, 0.0])
            for event in trace.events():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                duration = float(event.time_range.elapsed_us())
                by_name[event.name][0] += 1
                by_name[event.name][1] += duration
                bucket = category(event.name)
                by_category[bucket][0] += 1
                by_category[bucket][1] += duration
            top = sorted(by_name.items(), key=lambda item: item[1][1], reverse=True)[:30]
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "architecture": architecture,
                "backend": backend,
                "cuda_events": sum(value[0] for value in by_name.values()),
                "summed_cuda_duration_us": sum(value[1] for value in by_name.values()),
                "categories": {
                    name: {"events": value[0], "duration_us": value[1]}
                    for name, value in sorted(by_category.items())
                },
                "top_cuda_events": [
                    {"name": name, "events": value[0], "duration_us": value[1]}
                    for name, value in top
                ],
            })
            del module, value, output
            torch.cuda.empty_cache()
        del source, cotangent
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "method": "one warmed backward pass; CUDA device event durations from torch.profiler",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
