"""Count CUDA kernel events for selected RG-LRU and SAMU F+B paths."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from run_selected_dispatch_comparison import SHAPES
from run_small_model_study import RGLRUMixer, SAMUMixer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_seed(76301)
    rows = []
    for (batch, length, width, rg_backend, rg_block,
         samu_backend, controller) in SHAPES:
        x = torch.randn(batch, length, width, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        cotangent = torch.randn_like(x)
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
                module.set_controller_projection_dtype(controller)
                with torch.no_grad():
                    module.phase_amplitude.fill_(0.7)
                    module.radial_amplitude.fill_(0.5)
                backend = samu_backend

            def step():
                module.zero_grad(set_to_none=True)
                x.grad = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(x)[0]
                (output.float() * cotangent.float()).sum().backward()

            step()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                step()
                torch.cuda.synchronize()
            cuda_events = [event for event in trace.events()
                           if event.device_type == torch.autograd.DeviceType.CUDA]
            counts = Counter(event.name for event in cuda_events)
            rows.append({
                "shape": {"batch": batch, "length": length, "width": width},
                "architecture": architecture,
                "backend": backend,
                "cuda_kernel_events": len(cuda_events),
                "unique_cuda_kernel_names": len(counts),
                "top_cuda_kernels": counts.most_common(20),
            })
            del module
            torch.cuda.empty_cache()
        del x, cotangent
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "method": "one warmed complete mixer F+B under torch.profiler; CUDA device events",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
