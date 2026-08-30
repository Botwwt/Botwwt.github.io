# SAMU strengthened-baseline H800 report

This report records the final correctness-first training dispatch measured on
one NVIDIA H800 PCIe 80 GB GPU.  It compares SAMU with the strongest legal
canonical RG-LRU backend calibrated in this repository.  Both methods receive
shape dispatch, exact chunk scans, fused preparation/replay, and a linear-work
affine prefix hierarchy.  Compilation is excluded; every formal timing contains
two counterbalanced rounds, three warmups per round, and five timed samples per
round.

The current end-to-end scope is one complete mixer, one complete
`RecurrentBlock`, and the AdamW step for that same single block.  The
`400M-width` label below denotes the width configuration of one block; it is
not a current strengthened-baseline measurement of a 12-layer 400M language
model.  Earlier multi-layer training and generation runs remain historical
artifacts and are not used in this report's ranking.

`grouped_prefix` in backend names means that adjacent **chunk summaries** are
composed in execution groups before an outer prefix.  It does not group SAMU
controls, modes, or parameters and does not change the model equation.

## Selected exact backends

| Region | SAMU | strengthened RG-LRU |
|---|---|---|
| `B>=4,L<=2048` | shared-SFU, fused write/output, serial prefix, K=32 | grouped real-affine prefix 32, K=16 |
| `B=1,L=8192,D<2048` | shared-SFU, fused write/output, serial prefix, K=32 | grouped real-affine prefix 32, K=16 |
| `B=1,L=8192,D>=2048` | shared-SFU, fused write/output, grouped complex-affine prefix 64, K=32 | grouped real-affine prefix 32, K=16 |
| `B=1,16384<=L<65536` | shared-SFU, fused write/output, grouped complex-affine prefix 64, K=32 | grouped real-affine prefix 32, K=16 |
| `B=1,L>=65536` | shared-SFU, fused write/output, serial forward prefix plus grouped conjugate reverse prefix 64, K=32 | grouped real-affine prefix 32, K=16 |
| reset/packed or unsupported shape | exact canonical fallback | exact reset-aware canonical path |

The formal SAMU controller remains unchanged: normalized directions, signed
zero-initialized amplitudes, prescribed mode scaling, bounded radial map, and
exact RTU degeneration at zero controller amplitude.

## Complete mixer forward and backward

Times are medians in milliseconds.  Backward includes the dense output
cotangent, input gradient, and all parameter gradients.

| Shape `(B,L,D)` | RG F | SAMU F | RG B | SAMU B | RG F+B | SAMU F+B | SAMU F+B advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4,2048,2048` | 0.904 | 0.743 | 1.588 | 1.098 | 2.553 | 1.924 | 24.6% |
| `1,8192,2560` | 1.085 | 0.885 | 1.977 | 1.294 | 3.130 | 2.265 | 27.6% |
| `1,32768,1024` | 1.700 | 0.885 | 3.090 | 2.060 | 4.853 | 3.041 | 37.3% |
| `1,32768,2048` | 2.951 | 1.372 | 5.732 | 3.838 | 8.763 | 5.238 | 40.2% |

Source: [`selected_dispatch_grouped_k32_h800.json`](../results/gpu_optimization/selected_dispatch_grouped_k32_h800.json).

The mixer result is a Pareto result, not a claim that every SAMU shape wins.
At `B=1,L=8192,D=1024`, the corrected serial-K32 dispatch measures 1.801 ms
versus 1.603 ms for RG-LRU, so SAMU is 12.4% slower there.  The earlier
width-blind grouped dispatch was 2.04 ms and has been removed.  Source:
[`selected_dispatch_l8192_d1024_serial_k32_h800.json`](../results/gpu_optimization/selected_dispatch_l8192_d1024_serial_k32_h800.json).

## Sequence and width scaling

At fixed `B=1,D=1024`, SAMU crosses from a loss at 8K to a clear win at 32K.
The strict very-long output gate then requires serial forward prefix, so the
percentage advantage narrows at 65K and 131K while remaining positive.

| Length | RG F+B (ms) | SAMU F+B (ms) | SAMU advantage |
|---:|---:|---:|---:|
| 8,192 | 1.603 | 1.801 | -12.4% with corrected serial dispatch |
| 32,768 | 4.844 | 3.051 | 37.0% |
| 65,536 | 9.154 | 6.666 | 27.2% |
| 131,072 | 17.712 | 13.163 | 25.7% |

At fixed `B=1,L=32768`, F+B speedups are 37.3%, 40.8%, 40.2%, and
40.1% at widths 1024, 1536, 2048, and 2560 respectively.  This supports a
wide-state region rather than a claim of monotonically increasing percentage
gain at every width.

Sources: [`selected_dispatch_length_scaling_grouped_k32_h800.json`](../results/gpu_optimization/selected_dispatch_length_scaling_grouped_k32_h800.json),
[`selected_dispatch_very_long_hybrid_h800_v2.json`](../results/gpu_optimization/selected_dispatch_very_long_hybrid_h800_v2.json),
[`selected_dispatch_width_scaling_extra_grouped_k32_h800.json`](../results/gpu_optimization/selected_dispatch_width_scaling_extra_grouped_k32_h800.json), and
[`samu_l8192_width_dispatch_ab_h800.json`](../results/gpu_optimization/samu_l8192_width_dispatch_ab_h800.json).

## Recurrent block and optimizer step

The block adds shared convolution, projections, normalization, and output
layers, so the mixer advantage is diluted but remains measurable.  Every row
in both tables instantiates exactly one complete `RecurrentBlock`.

| Shape | RG block F+B | SAMU block F+B | advantage | RG peak allocated | SAMU peak allocated |
|---|---:|---:|---:|---:|---:|
| state-2048 short | 15.764 ms | 14.699 ms | 6.8% | 1.573 GiB | 1.448 GiB |
| state-2560 medium | 20.746 ms | 19.544 ms | 5.8% | 1.943 GiB | 1.825 GiB |
| state-1024 long | 24.120 ms | 22.276 ms | 7.6% | 2.843 GiB | 2.656 GiB |
| 400M-width long | 43.202 ms | 39.949 ms | 7.5% | 4.634 GiB | 4.259 GiB |

| Shape | RG optimizer step | SAMU optimizer step | advantage |
|---|---:|---:|---:|
| state-2048 short | 24.808 ms | 23.728 ms | 4.4% |
| state-2560 medium | 34.394 ms | 33.101 ms | 3.8% |
| 400M-width long | 48.997 ms | 45.796 ms | 6.5% |

The optimizer-step peak reserved deltas are allocator-sensitive.  They are
1.795/1.660 GiB (RG/SAMU) for short, 1.953/1.949 GiB for medium, and
4.598/4.658 GiB for the 400M-width single-block case.  Thus SAMU lowers peak allocated memory in all
three cases, while reserved memory is essentially tied and is 1.3% higher in
the 400M-width single-block sample.

Sources: [`block_dispatch_grouped_k32_h800.json`](../results/gpu_optimization/block_dispatch_grouped_k32_h800.json) and
[`optimizer_step_grouped_k32_h800.json`](../results/gpu_optimization/optimizer_step_grouped_k32_h800.json).

## Memory, scratch, parameters, and launches

The mixer peak allocated delta is 22.0%, 22.0%, 23.4%, and 23.5% lower for
SAMU on the four primary shapes.  Mixer parameter counts are 6,148 versus
530,432 at `D=2048`, 7,684 versus 826,880 at `D=2560`, and 3,076 versus
134,144 at `D=1024` (SAMU versus RG-LRU).

For the selected prefix implementations, analytical transient affine-prefix
scratch is

```text
RG-LRU: 12 * B * D * (ceil(L/16) + groups32) bytes
SAMU:   12 * B * D * (ceil(L/32) + groups64) bytes
```

where the short SAMU serial-prefix path has no outer-group term.  This is 6.0
versus 12.375 MiB on the short shape, 7.617 versus 15.469 MiB on medium,
12.188 versus 24.750 MiB on long D1024, and 24.375 versus 49.500 MiB on long
D2048.  These figures count prefix summary/boundary arrays and exclude model
activations and outputs; PyTorch peak measurements above include the complete
measured scope.

Complete mixer F+B launches are 70/50 (SAMU/RG) on short, 74/50 on medium,
and 73/51 on both long shapes.  SAMU is faster despite more small controller
and framework launches.  This launch-count disadvantage remains an optimization
limit.  Source: [`selected_dispatch_launches_grouped_k32_h800.json`](../results/gpu_optimization/selected_dispatch_launches_grouped_k32_h800.json).

The full grouped-K32 path passes the wide-state 32K gate.  At wide-state
65K/131K it produced BF16 output relative errors `5.84e-6`/`7.49e-6`, slightly
above the predeclared `5e-6` limit, even though input and parameter gradients
passed.  That failure is retained in
[`samu_grouped_k32_long_correctness.json`](../results/gpu_optimization/samu_grouped_k32_long_correctness.json) and the full-group path is not dispatched there.

The very-long replacement keeps serial forward chunk-prefix parenthesization
and retains the grouped conjugate hierarchy in reverse.  At the actual
`D=1024` 65K/131K shapes, outputs are bitwise identical to serial K32,
input-gradient relative errors are `4.72e-6`/`7.09e-6`, and maximum
parameter-gradient errors are `3.29e-5`/`1.67e-4`; all pass the original
thresholds.  Sources:
[`samu_hybrid_prefix64_65536_d1024_correctness_v2.json`](../results/gpu_optimization/samu_hybrid_prefix64_65536_d1024_correctness_v2.json) and
[`samu_hybrid_prefix64_131072_d1024_correctness_v2.json`](../results/gpu_optimization/samu_hybrid_prefix64_131072_d1024_correctness_v2.json).
Runs produced before the SAMU parser correctly decoded the hybrid marker are
kept under `results/gpu_optimization/quarantined_hybrid_parser_bug` and are not
used in any table.

## Reverse complex backward

The selected backward mirrors forward at the chunk-summary level:

1. each chunk constructs a conjugated reverse affine summary;
2. a group-local, outer, and correction hierarchy performs the noncommutative
   reverse prefix in linear work;
3. K=32 replay reconstructs transitions in registers;
4. each mode tile emits two shared controller-gradient partials per token;
5. a deterministic FP32 second stage reduces `[B,L,N_tiles,2]` to `[B,L,2]`.

On the long D2048 profiler shape, successive exact reverse/output/prefix
changes reduce summed backward CUDA event time from 6.265 ms to 3.777 ms.
The final reverse prefix is 0.250 ms, replay 0.602 ms, reverse summary
0.166 ms, and shared-control reduction 0.022 ms.  The largest reported copy
kernels include the common FP32 dense-cotangent/loss boundary and are not all
SAMU recurrence work.  Source:
[`backward_components_grouped_k32_h800.json`](../results/gpu_optimization/backward_components_grouped_k32_h800.json).

## Candidates that remain disabled

- **Compressed G/D plus grouped K32:** exact closure is implemented and useful
  as a memory representation, but the final strict interaction gate exceeded
  its predeclared BF16 output threshold (`6.38e-6` versus `5e-6`) and the
  hierarchical path must reconstruct transient P.  It is not the default.
- **FP32 atomic shared-control reduction:** numerically close but not bitwise
  deterministic and no faster than deterministic two-stage reduction.
- **Two-stage spectral reduction:** neutral at D1024 and about 0.7% slower at
  D2048, with extra scratch.
- **Static spectral cache:** neutral at medium/D2048 and about 1.7% slower at
  long D1024.
- **BF16 controller projection/accumulation:** direction gradients are around
  `2e-3` relative error.  The selected fused-coordinates controller keeps its
  normalization, direction-gradient projection, and reductions in FP32 and
  retains the compact p/r cache required by the formal derivative.
- **K=64 complex replay:** Triton compilation was repeatedly terminated before
  a valid runtime result.  K=32 remains selected.
- **Full grouped forward at 65K/131K:** failed the predeclared wide-state BF16
  output gate and was replaced by serial forward plus grouped reverse.

No rejection threshold was relaxed after seeing a result.

## Supported claim

The measurements support a shape-dependent GPU Pareto advantage from SAMU's
low-rank coherent control: shared token special functions, on-the-fly mode
transitions, a complex affine hierarchy, lower prefix scratch, and low-rank
controller-gradient intermediates translate into lower mixer latency and peak
allocated memory for the primary short, medium-wide, and long regions.  They
also translate into 3.8--6.5% one-block optimizer-step and 5.8--7.6% one-block
wins on the measured selected shapes.  They do not establish that SAMU wins every shape,
that its pure scan primitive is the fastest, or that random operator benchmarks
prove model quality.  The current evidence is one H800, one software stack,
and exact operator/block execution; training quality and multi-GPU scaling are
separate questions.
