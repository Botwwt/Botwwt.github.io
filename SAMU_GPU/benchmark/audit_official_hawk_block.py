"""Compare the H800 Hawk block shell with pinned RecurrentGemma modules."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types

import torch

from run_small_model_study import RecurrentBlock, SAMUMixer, StudyConfig


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_official_block(official, ours) -> None:
    """Copy every shared RG-LRU Hawk tensor without changing its layout."""
    with torch.no_grad():
        ours.temporal_norm.scale.copy_(official.temporal_pre_norm.scale)
        ours.channel_norm.scale.copy_(official.channel_pre_norm.scale)
        for official_linear, ours_linear in (
            (official.recurrent_block.linear_y, ours.linear_y),
            (official.recurrent_block.linear_x, ours.linear_x),
            (official.recurrent_block.linear_out, ours.linear_out),
            (official.mlp_block.ffw_down, ours.mlp.down),
        ):
            ours_linear.weight.copy_(official_linear.weight)
            ours_linear.bias.copy_(official_linear.bias)
        ours.conv.w.copy_(official.recurrent_block.conv_1d.w)
        ours.conv.b.copy_(official.recurrent_block.conv_1d.b)
        for official_gate, ours_gate in (
            (official.recurrent_block.rg_lru.input_gate, ours.mixer.input_gate),
            (official.recurrent_block.rg_lru.a_gate, ours.mixer.a_gate),
        ):
            ours_gate.w.copy_(official_gate.w)
            ours_gate.b.copy_(official_gate.b)
        ours.mixer.a_param.copy_(official.recurrent_block.rg_lru.a_param)

        # Official Einsum stores [branch, input, expanded].  nn.Linear stores
        # [output, input], with the first expanded-width rows being the gate.
        ffw_weight = official.mlp_block.ffw_up.w.permute(0, 2, 1).reshape(
            ours.mlp.up.weight.shape
        )
        ffw_bias = official.mlp_block.ffw_up.b.reshape(ours.mlp.up.bias.shape)
        ours.mlp.up.weight.copy_(ffw_weight)
        ours.mlp.up.bias.copy_(ffw_bias)


def numerical_equivalence(common, modules) -> dict:
    """Compare the whole official RG-LRU residual block, not only counts."""
    torch.manual_seed(20260830)
    width, rnn_width, depth = 32, 64, 12
    official = modules.ResidualBlock(
        width=width,
        mlp_expanded_width=3 * width,
        num_heads=16,
        attention_window_size=1024,
        temporal_block_type=common.TemporalBlockType.RECURRENT,
        lru_width=rnn_width,
        conv1d_temporal_width=4,
        final_w_init_variance_scale=2.0 / depth,
        dtype=torch.float32,
    ).eval()
    ours = RecurrentBlock(
        StudyConfig(
            vocab_size=256,
            width=width,
            rnn_width=rnn_width,
            depth=depth,
            mlp_expansion=3,
            num_gate_blocks=16,
            conv_width=4,
            embedding_scale_by_sqrt_dim=False,
            final_w_init_variance_scale=2.0 / depth,
        ),
        "rglru",
    ).eval()
    ours.mixer.set_scan_backend("framework_eager")
    copy_official_block(official, ours)

    x = torch.randn(3, 19, width)
    segment_pos = torch.arange(19).repeat(3, 1)
    with torch.no_grad():
        official_output, _ = official(
            x.clone(), segment_pos, cache=None, return_cache=False
        )
        ours_output, _ = ours(x.clone(), return_cache=False)

        official_normalized = official.temporal_pre_norm(x)
        official_recurrent_input = official.recurrent_block.linear_x(
            official_normalized
        )
        official_convolved, _ = official.recurrent_block.conv_1d(
            official_recurrent_input.clone(),
            segment_pos,
            cache=None,
            return_cache=False,
        )
        official_recurrent, official_last = official.recurrent_block.rg_lru(
            official_convolved,
            segment_pos,
            cache=None,
            return_cache=True,
        )

        ours_normalized = ours.temporal_norm(x)
        ours_recurrent_input = ours.linear_x(ours_normalized)
        ours_convolved, _ = ours.conv(ours_recurrent_input, return_cache=False)
        ours_recurrent, _ = ours.mixer(ours_convolved, return_cache=False)
        ours_last = ours_recurrent[:, -1].float()

    component_errors = {
        "temporal_norm_max_abs": float(
            (ours_normalized - official_normalized).abs().max()
        ),
        "linear_x_max_abs": float(
            (ours_recurrent_input - official_recurrent_input).abs().max()
        ),
        "causal_conv1d_max_abs": float(
            (ours_convolved - official_convolved).abs().max()
        ),
        "rglru_sequence_max_abs": float(
            (ours_recurrent - official_recurrent).abs().max()
        ),
        "rglru_last_state_max_abs": float(
            (ours_last - official_last).abs().max()
        ),
        "whole_residual_block_max_abs": float(
            (ours_output - official_output).abs().max()
        ),
    }
    return {
        "dtype": "float32",
        "shape": [3, 19, width],
        "segment_layout": "three independent sequences; position zero resets each sequence",
        "component_max_abs_errors": component_errors,
        "tolerance": 2e-5,
        "passed": max(component_errors.values()) <= 2e-5,
    }


def canonical_samu_equivalence() -> dict:
    """Audit the G=1 adapter against canonical_grouped_samu_math equations."""
    torch.manual_seed(20260831)
    width, batch, length = 32, 3, 19
    mixer = SAMUMixer(width).eval()
    mixer.phase_amplitude.data.fill_(0.63)
    mixer.radial_amplitude.data.fill_(-0.41)
    mixer.set_scan_backend("framework_eager")
    x = torch.randn(batch, length, width)
    with torch.no_grad():
        observed, _ = mixer(x, return_cache=False)

        augmented = torch.cat(
            (x.float(), torch.ones(batch, length, 1)), dim=-1
        )
        phase_direction = mixer.phase_direction.float()
        radial_direction = mixer.radial_direction.float()
        phase_direction = phase_direction / phase_direction.norm()
        radial_direction = radial_direction / radial_direction.norm()
        phase_selector = torch.tanh(augmented @ phase_direction)
        radial_selector = torch.tanh(augmented @ radial_direction)
        physical_radial = radial_selector / (1.0 + radial_selector.square())
        scale = 1.0 / (mixer.modes ** 0.5)
        delta = scale * torch.tanh(mixer.phase_amplitude.float()) * phase_selector
        eta = scale * torch.tanh(mixer.radial_amplitude.float()) * physical_radial
        nu = torch.exp(mixer.nu_log.float())
        theta = torch.exp(mixer.theta_log.float())
        transition_radius = torch.exp(-nu * torch.exp(eta).unsqueeze(-1))
        transition_phase = theta + delta.unsqueeze(-1)
        ar = transition_radius * torch.cos(transition_phase)
        ai = transition_radius * torch.sin(transition_phase)
        gamma = torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8
        write_r = x[..., :mixer.modes].float() * gamma
        write_i = x[..., mixer.modes:].float() * gamma
        state_r = torch.zeros(batch, mixer.modes)
        state_i = torch.zeros_like(state_r)
        reference_r, reference_i = [], []
        for time_index in range(length):
            next_r = (
                ar[:, time_index] * state_r
                - ai[:, time_index] * state_i
                + write_r[:, time_index]
            )
            next_i = (
                ai[:, time_index] * state_r
                + ar[:, time_index] * state_i
                + write_i[:, time_index]
            )
            state_r, state_i = next_r, next_i
            reference_r.append(state_r)
            reference_i.append(state_i)
        reference = torch.cat(
            (torch.stack(reference_r, 1), torch.stack(reference_i, 1)), dim=-1
        ).relu()

        mixer.prepare_inference()
        packed_gamma = mixer._inference_pack[-1]
    errors = {
        "full_sequence_max_abs": float((observed.float() - reference).abs().max()),
        "packed_gamma_max_abs": float((packed_gamma - gamma).abs().max()),
    }
    return {
        "reference": "canonical_grouped_samu_math.py with groups=1",
        "equation_contract": {
            "write_normalization": "sqrt(1-exp(-2*nu)) + 1e-8",
            "shared_controls": "one normalized affine-tanh phase selector and one normalized affine-tanh rational radial selector",
            "output": "ReLU(concat(real_state, imaginary_state))",
        },
        "component_max_abs_errors": errors,
        "tolerance": 2e-6,
        "passed": max(errors.values()) <= 2e-6,
    }


def decode_consistency() -> dict:
    """Compare full-sequence BF16 blocks with one-token GPU decode paths."""
    if not torch.cuda.is_available():
        return {"status": "cuda_unavailable", "passed": False}
    torch.manual_seed(20260901)
    config = StudyConfig(
        vocab_size=256,
        width=32,
        rnn_width=64,
        depth=6,
        mlp_expansion=3,
        num_gate_blocks=16,
        conv_width=4,
    )
    x = torch.randn(2, 11, config.width, device="cuda", dtype=torch.bfloat16)
    rows = []
    for architecture, backends in (
        ("samu", ("packed",)),
        ("rglru", ("fused", "bmm")),
    ):
        block = RecurrentBlock(config, architecture).cuda().to(torch.bfloat16).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected, expected_cache = block(x, return_cache=True)
        block.mixer.prepare_inference()
        for backend in backends:
            if architecture == "samu":
                block.mixer.set_decode_backend(backend)
            else:
                block.mixer.set_decode_backend(backend)
                block.mixer.set_decode_num_warps(4)
            conv_cache = torch.zeros(
                x.shape[0], config.conv_width - 1, config.rnn_width,
                device="cuda", dtype=torch.bfloat16,
            )
            if architecture == "rglru":
                recurrent_cache = torch.zeros(
                    x.shape[0], config.rnn_width,
                    device="cuda", dtype=torch.float32,
                )
            else:
                recurrent_cache = (
                    torch.zeros(x.shape[0], config.rnn_width // 2, device="cuda"),
                    torch.zeros(x.shape[0], config.rnn_width // 2, device="cuda"),
                )
            cache = (conv_cache, recurrent_cache)
            observed = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for time_index in range(x.shape[1]):
                    value, cache = block.step(
                        x[:, time_index], cache,
                        torch.full(
                            (x.shape[0],), time_index,
                            device="cuda", dtype=torch.long,
                        ),
                    )
                    observed.append(value)
            observed = torch.stack(observed, dim=1)
            if architecture == "rglru":
                recurrence_error = float(
                    (cache[1] - expected_cache[1]).abs().max()
                )
            else:
                recurrence_error = max(
                    float((cache[1][0] - expected_cache[1][0]).abs().max()),
                    float((cache[1][1] - expected_cache[1][1]).abs().max()),
                )
            row = {
                "architecture": architecture,
                "decode_backend": backend,
                "block_output_max_abs": float(
                    (observed.float() - expected.float()).abs().max()
                ),
                "conv_cache_max_abs": float(
                    (cache[0].float() - expected_cache[0].float()).abs().max()
                ),
                "recurrent_cache_max_abs": recurrence_error,
                "convolution_cache_dtype": str(cache[0].dtype),
                "recurrent_cache_dtype": (
                    str(cache[1].dtype)
                    if architecture == "rglru"
                    else str(cache[1][0].dtype)
                ),
            }
            row["passed"] = max(
                row["block_output_max_abs"], row["conv_cache_max_abs"],
                row["recurrent_cache_max_abs"],
            ) <= 0.05
            rows.append(row)
        del block
    return {
        "precision": "BF16 block activations and convolution cache; FP32 recurrence cache",
        "shape": [2, 11, config.width],
        "tolerance": 0.05,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rglru-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.rglru_source.resolve()
    sys.path.insert(0, str(source_root))
    # The top-level package imports optional JAX/Flax modules.  Install a
    # namespace package so this audit loads the official PyTorch files only,
    # exactly as the kernel equation audit does.
    package = types.ModuleType("recurrentgemma")
    package.__path__ = [str(source_root / "recurrentgemma")]
    sys.modules["recurrentgemma"] = package
    from recurrentgemma import common
    from recurrentgemma.torch import modules

    rows = []
    for scale, width, rnn_width, depth in (
        ("400m", 1536, 2048, 12),
        ("1.3b", 2048, 2560, 24),
    ):
        variance_scale = 2.0 / depth
        official = modules.ResidualBlock(
            width=width,
            mlp_expanded_width=3 * width,
            num_heads=16,
            attention_window_size=1024,
            temporal_block_type=common.TemporalBlockType.RECURRENT,
            lru_width=rnn_width,
            conv1d_temporal_width=4,
            final_w_init_variance_scale=variance_scale,
            dtype=torch.bfloat16,
        )
        config = StudyConfig(
            vocab_size=32000,
            width=width,
            rnn_width=rnn_width,
            depth=depth,
            mlp_expansion=3,
            num_gate_blocks=16,
            conv_width=4,
            embedding_scale_by_sqrt_dim=False,
            final_w_init_variance_scale=variance_scale,
        )
        ours = RecurrentBlock(config, "rglru").to(torch.bfloat16)
        official_components = {
            "temporal_norm": count(official.temporal_pre_norm),
            "linear_y": count(official.recurrent_block.linear_y),
            "linear_x": count(official.recurrent_block.linear_x),
            "conv1d": count(official.recurrent_block.conv_1d),
            "mixer": count(official.recurrent_block.rg_lru),
            "linear_out": count(official.recurrent_block.linear_out),
            "channel_norm": count(official.channel_pre_norm),
            "gated_mlp": count(official.mlp_block),
        }
        ours_components = {
            "temporal_norm": count(ours.temporal_norm),
            "linear_y": count(ours.linear_y),
            "linear_x": count(ours.linear_x),
            "conv1d": count(ours.conv),
            "mixer": count(ours.mixer),
            "linear_out": count(ours.linear_out),
            "channel_norm": count(ours.channel_norm),
            "gated_mlp": count(ours.mlp),
        }
        rows.append({
            "scale": scale,
            "config": {
                "width": width,
                "rnn_width": rnn_width,
                "depth": depth,
                "mlp_expanded_width": 3 * width,
                "gate_blocks": 16,
                "conv1d_temporal_width": 4,
                "embedding_scale_by_sqrt_dim": False,
                "final_projection_initialization_variance_scale": variance_scale,
            },
            "official_component_parameters": official_components,
            "h800_component_parameters": ours_components,
            "official_total_block_parameters": count(official),
            "h800_total_block_parameters": count(ours),
            "component_counts_equal": official_components == ours_components,
            "total_count_equal": count(official) == count(ours),
        })
        del official, ours
    result = {
        "official_source": str(args.rglru_source),
        "official_commit": "2efa84dac0e68e63547a27a18fa943c98f1c312e",
        "comparison": "pinned official Hawk RG-LRU residual block versus H800 RG-LRU block before replacing only the mixer with SAMU",
        "rows": rows,
        "numerical_equivalence": numerical_equivalence(common, modules),
        "canonical_samu_equivalence": canonical_samu_equivalence(),
        "decode_consistency": decode_consistency(),
        "passed": all(
            row["component_counts_equal"] and row["total_count_equal"]
            for row in rows
        ),
    }
    result["passed"] = (
        result["passed"]
        and result["numerical_equivalence"]["passed"]
        and result["canonical_samu_equivalence"]["passed"]
        and result["decode_consistency"]["passed"]
    )
    atomic_json(args.output, result)
    if not result["passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
