"""Direct H800 A/B of strengthened RG-LRU and shape-dispatched SAMU mixers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from run_small_model_study import RGLRUMixer, SAMUMixer, set_seed


SHAPES = (
    # batch, length, width, RG backend/block, SAMU backend/controller
    (4, 2048, 2048, "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_chunk32", "fp32_fused_coords"),
    (1, 8192, 2560, "fused_grouped_prefix32_hierarchical_chunk16", 256,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32",
     "fp32_fused_coords"),
    (1, 32768, 1024, "fused_grouped_prefix32_hierarchical_chunk16", 256,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32",
     "fp32_fused_coords"),
    (1, 32768, 2048, "fused_grouped_prefix32_hierarchical_chunk16", 128,
     "fused_output_fused_write_shared_sfu_grouped_prefix64_hierarchical_chunk32",
     "fp32_fused_coords"),
)


def parse_case(value: str):
    pieces = value.split(":")
    dimensions = pieces[0].lower().replace("x", ",").split(",")
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError(
            "case must be BxLxD or BxLxD:rg_backend:samu_backend"
        )
    batch, length, width = (int(piece) for piece in dimensions)
    if len(pieces) == 1:
        rg_backend = "fused_grouped_prefix32_hierarchical_chunk16"
        rg_block = (128 if width >= 1536 and length >= 16384
                    else 256 if width > 2048 or length >= 16384 else 128)
        samu_backend = ("fused_output_fused_write_shared_sfu_chunk32"
                        if batch >= 4 and length <= 2048
                        else "fused_output_fused_write_shared_sfu_"
                             "grouped_prefix64_hierarchical_chunk32")
        controller = "fp32_fused_coords"
    elif len(pieces) == 3:
        rg_backend, samu_backend = pieces[1:]
        rg_block = 64
        controller = "fp32"
    elif len(pieces) == 4:
        rg_backend, samu_backend, controller = pieces[1:]
        rg_block = 64
    elif len(pieces) == 5:
        rg_backend, rg_block_text, samu_backend, controller = pieces[1:]
        rg_block = int(rg_block_text.removeprefix("b"))
    else:
        raise argparse.ArgumentTypeError(
            "case must be BxLxD or BxLxD:rg_backend:samu_backend"
        )
    return (batch, length, width, rg_backend, rg_block,
            samu_backend, controller)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def timed(callable_) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def summarize(samples: list[float]) -> dict:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size()
               for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=75031)
    parser.add_argument("--case", action="append", type=parse_case, default=[])
    args = parser.parse_args()
    shapes = tuple(args.case) if args.case else SHAPES
    set_seed(args.seed)
    result = {
        "schema_version": 1,
        "protocol": {
            "scope": "complete mixer preparation, recurrence, input and parameter gradients",
            "dtype": "BF16 autocast and FP32 recurrent carry",
            "input_gradient": True,
            "order": "AB/BA counterbalanced",
            "compile_excluded": True,
            "baseline_policy": "strongest previously calibrated legal RG-LRU backend per shape",
            "samu_policy": "strongest previously calibrated legal SAMU backend per shape",
            "resets": "independent sequences reset at first token; no internal packed reset",
        },
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "rows": [],
    }
    for shape_index, (batch, length, width, rg_backend, rg_block, samu_backend,
                      samu_controller) in enumerate(shapes):
        print(f"shape B={batch} L={length} D={width}", flush=True)
        x = torch.randn(batch, length, width, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        cotangent = torch.randn_like(x)
        rglru = RGLRUMixer(width, 16).cuda().train()
        rglru.set_scan_backend(rg_backend)
        rglru.set_fused_training_block_size(rg_block)
        samu = SAMUMixer(width).cuda().train()
        samu.set_scan_backend(samu_backend)
        samu.set_tiled_training_block_size(256)
        samu.set_controller_projection_dtype(samu_controller)
        with torch.no_grad():
            samu.phase_amplitude.fill_(0.7)
            samu.radial_amplitude.fill_(0.5)
        modules = {"rglru": rglru, "samu": samu}
        backends = {"rglru": rg_backend, "samu": samu_backend}

        def forward(name):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return modules[name](x)[0]

        def backward_only(name):
            modules[name].zero_grad(set_to_none=True)
            x.grad = None
            output = forward(name)
            return timed(lambda o=output: (o.float() * cotangent.float()).sum().backward())

        def forward_backward(name):
            modules[name].zero_grad(set_to_none=True)
            x.grad = None
            output = forward(name)
            (output.float() * cotangent.float()).sum().backward()

        for name in modules:
            forward_backward(name)
        raw = {name: {"forward": [], "backward": [], "forward_backward": [],
                      "rounds": []} for name in modules}
        names = tuple(modules)
        for round_index in range(args.rounds):
            order = names if (round_index + shape_index) % 2 == 0 else tuple(reversed(names))
            for order_index, name in enumerate(order):
                for _ in range(args.warmup):
                    forward(name)
                forward_samples = [timed(lambda n=name: forward(n))
                                   for _ in range(args.repetitions)]
                backward_samples = []
                for index in range(args.warmup + args.repetitions):
                    sample = backward_only(name)
                    if index >= args.warmup:
                        backward_samples.append(sample)
                for _ in range(args.warmup):
                    forward_backward(name)
                fb_samples = [timed(lambda n=name: forward_backward(n))
                              for _ in range(args.repetitions)]
                raw[name]["forward"].extend(forward_samples)
                raw[name]["backward"].extend(backward_samples)
                raw[name]["forward_backward"].extend(fb_samples)
                raw[name]["rounds"].append({
                    "round": round_index,
                    "order_index": order_index,
                    "forward_ms": forward_samples,
                    "backward_ms": backward_samples,
                    "forward_backward_ms": fb_samples,
                })
                print(name, statistics.median(forward_samples),
                      statistics.median(backward_samples),
                      statistics.median(fb_samples), flush=True)

        for name, module in modules.items():
            module.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            forward_backward(name)
            torch.cuda.synchronize()
            result["rows"].append({
                "shape": {"batch": batch, "length": length, "width": width},
                "architecture": name,
                "backend": backends[name],
                "controller": samu_controller if name == "samu" else None,
                "forward": summarize(raw[name]["forward"]),
                "backward": summarize(raw[name]["backward"]),
                "forward_backward": summarize(raw[name]["forward_backward"]),
                "rounds": raw[name]["rounds"],
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - allocated,
                "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - reserved,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "baseline_allocated_bytes": allocated,
                "baseline_reserved_bytes": reserved,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
                "parameter_bytes": parameter_bytes(module),
                "dynamic_control_values_per_token": 2 * width if name == "rglru" else 2,
                "materialized_state_sized_transition": False,
            })
        atomic_json(args.output, result)
        del x, cotangent, rglru, samu, modules
        torch.cuda.empty_cache()
    print(f"complete {args.output}", flush=True)


if __name__ == "__main__":
    main()
