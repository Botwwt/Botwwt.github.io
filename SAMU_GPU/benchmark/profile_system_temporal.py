"""Profile the two-launch width-2048 temporal decode paths used by the 1B proxy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kernels import load_official_rglru, make_samu_parameters
from profile_equal_kernels import (
    atomic_json,
    counter_permission_evidence,
    device_limits,
    profile_operation,
)
import triton_rglru as rg
import triton_samu as samu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglru-source", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    width, modes = 2048, 1024
    props = torch.cuda.get_device_properties(0)
    limits = device_limits(props)

    samu_params = make_samu_parameters(
        width, modes, "cuda", torch.bfloat16, seed=5701
    )
    packed_samu = samu.pack_samu_parameters(samu_params)
    RGLRU = load_official_rglru(args.rglru_source)
    torch.manual_seed(5701)
    rg_model = RGLRU(
        width=width, num_heads=2, device="cuda", dtype=torch.bfloat16
    ).eval()
    packed_rg = rg.pack_rglru(rg_model)

    profiles = []
    for batch in (1, 16, 128):
        token = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
        samu_state = (
            torch.randn(batch, modes, device="cuda", dtype=torch.float32),
            torch.randn(batch, modes, device="cuda", dtype=torch.float32),
        )
        positions = torch.ones(batch, 1, device="cuda", dtype=torch.long)
        rg_state = torch.randn(batch, width, device="cuda", dtype=torch.float32)
        profiles.append(profile_operation(
            f"samu_split_decode_D2048_B{batch}",
            lambda: samu.samu_triton_decode_split(token, samu_state, packed_samu),
            [samu._decode_projected_kernel], limits,
        ))
        profiles.append(profile_operation(
            f"rglru_split_decode_D2048_B{batch}",
            lambda: rg.rglru_triton_serial(
                token[:, None], positions, packed_rg, rg_state
            ),
            [rg._serial_scan_kernel], limits,
        ))
    atomic_json(args.output, {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": props.name,
        "width": width,
        "state_bytes_per_sequence": width * 4,
        "temporal_projection_parameters": {
            "samu": int(packed_samu.weight.numel()),
            "rglru": int(packed_rg.gate_weight.numel() + packed_rg.gate_bias.numel()
                         + packed_rg.softplus_a.numel()),
        },
        "device_limits": limits,
        "counter_permission": counter_permission_evidence(),
        "profiles": profiles,
    })
    print(json.dumps({
        profile["name"]: {
            "launches": profile["cuda_kernel_launch_count"],
            "self_cuda_total_us": profile["self_cuda_total_us"],
        }
        for profile in profiles
    }, indent=2))


if __name__ == "__main__":
    main()
