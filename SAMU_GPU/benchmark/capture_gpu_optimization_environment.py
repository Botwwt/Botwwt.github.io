"""Capture reproducibility metadata for the H800 optimization study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import torch
import triton


def command(*arguments):
    try:
        return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"UNAVAILABLE: {type(error).__name__}: {error}"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--public-python", type=Path, required=True)
    args = parser.parse_args()
    gpu_query = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.current.sm,clocks.current.memory,ecc.mode.current,mig.mode.current,compute_cap",
        "--format=csv,noheader,nounits",
    )
    public_environment = command(
        str(args.public_python), "-c",
        "import json,torch,triton,sys; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__}))",
    )
    try:
        public_environment = json.loads(public_environment)
    except Exception:
        pass
    payload = {
        "schema_version": 1,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "torch_name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "nvidia_smi_query_fields": "name,uuid,driver,memory MiB,power W,SM MHz,memory MHz,ECC,MIG,compute capability",
            "nvidia_smi_query": gpu_query,
        },
        "profilers": {
            "ncu": command("sh", "-c", "command -v ncu || true"),
            "nsys": command("sh", "-c", "command -v nsys || true"),
            "policy": "no hardware-counter claim when tools are unavailable",
        },
        "git": {
            "worktree": str(args.worktree),
            "head": command("git", "-C", str(args.worktree), "rev-parse", "HEAD"),
            "branch": command("git", "-C", str(args.worktree), "branch", "--show-current"),
            "status": command("git", "-C", str(args.worktree), "status", "--short"),
            "diff_stat": command("git", "-C", str(args.worktree), "diff", "--stat"),
            "original": str(args.original),
            "original_head": command("git", "-C", str(args.original), "rev-parse", "HEAD"),
            "original_status": command("git", "-C", str(args.original), "status", "--short"),
        },
        "public_reproduction_environment": public_environment,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
