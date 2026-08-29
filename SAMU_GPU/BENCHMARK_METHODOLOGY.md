# Equal-kernel SAMU / RG-LRU methodology

## Question and scope

This suite measures forward scan and inference execution on one NVIDIA H800
PCIe 80GB. Its primary question is whether SAMU's low-dimensional transition
control produces a tighter GPU path than an equally optimized, paper-default
RG-LRU-16 kernel. It does not measure model quality, backward performance,
multi-device model parallelism, or ZeRO.

The report has three explicitly named tracks: official-structure RG-LRU-16
versus SAMU GPU candidates at `D=2048`; a Griffin Figure 8(a)-shaped forward
scan experiment; and a secondary, artificial parameter-matched RG-LRU-2
microbenchmark at `D=128`. The pinned RecurrentGemma implementation is the
equation and numerical reference for every RG-LRU track.

## Official RG-LRU semantics

The Triton implementation in `benchmark/triton_rglru.py` follows Griffin
equations 1–4 and both the public JAX and PyTorch RecurrentGemma layers at commit
`2efa84dac0e68e63547a27a18fa943c98f1c312e`:

$$
\begin{aligned}
i_t &= \sigma(W_xx_t+b_x),\\
r_t &= \sigma(W_ax_t+b_a),\\
\log a_t &= -8r_t\odot\operatorname{softplus}(a_{\mathrm{param}}),\\
h_t &= a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t).
\end{aligned}
$$

At `segment_pos == 0`, the official implementation resets the recurrent
coefficient to zero and sets the input multiplier to one. The optimized kernel
preserves this behavior at arbitrary mid-sequence positions, uses the official
block-diagonal projection layout, matches eager-BF16 pointwise rounding, and
keeps the accumulator/cache in FP32.

The equation, block geometry, reset rule, and precision policy are official.
The primary opponent uses the paper-default 16 gate groups. Only the secondary
`D=128` parameter-matched track uses two groups, and it is labelled RG-LRU-2 in
the report. Packing both gate projections into one BMM, the serial/chunk Triton scans, the
fused one-token decode kernel, and the shape dispatcher are this project's
optimizations. Affine chunk scan is generic algebra rather than an official
RG-LRU optimization or a new algorithmic invention claimed here.

The JAX source provides the optimized `linear_scan`/Pallas path and independently
confirms the same real RG-LRU equation. Correctness on the H800 is checked
directly against `recurrentgemma.torch.layers.RGLRU`, including a nonzero initial cache,
mid-sequence resets at `t=5` and `t=77`, and mixed reset/non-reset decode.
Serial, C16, decode output, and final-cache errors are stored in
`benchmark_results_equal_kernel/correctness.json`.

## Secondary parameter-matched microbenchmark

This diagnostic track fixes:

- `d_model = 128`, SAMU `M = 64` complex modes;
- SAMU 16,772 parameters versus RG-LRU 16,768 parameters;
- 128 FP32 recurrent scalars, or 512 bytes per batch element, on both sides;
- BF16 input/output with FP32 recurrent accumulation/cache;
- one packed projection followed by recurrence work;
- four CUDA launches for C32 prefill and one launch for fused decode.

The prefill sweep uses `L = 128, 512, 2048, 8192, 32768, 65536`. Batch
scaling is measured at `L=512`; decode uses `B = 1, 4, 16, 64`; explicit
chunk sizes 8, 16, and 32 are retained at `L=2048` and `L=65536`.

Dispatch choices were calibrated on a separate, non-published timing sweep.
On the H800, SAMU uses C8/4 warps through `L=256`, C16/2 warps through
`L=512`, and C32/2 warps beyond that. At `L<=128`, the packed parameters must pass analytic bounds
before the bounded-exp polynomial is enabled; otherwise it falls back to the
general exponential. RG-LRU uses C8 through `L=256`, C16 through `L=1024`,
and C32 beyond that, with 4 warps through `L=2048` and 2 warps on longer
sequences. Dispatch is resolved before timing on both sides. These
thresholds are device-specific measured choices, not claims about other GPUs.

The official RecurrentGemma PyTorch result is retained as a formula/source
reference. Its Python sequence loop is not used as evidence that RG-LRU's
architecture is intrinsically slow.

Steady-state timing excludes JIT compilation. In the current SAMU system proxy,
four layer-specific control constants are Triton compile-time constants, so a
24-layer cold start produces specialized cubins for each layer. This improves
specialized steady-state code but makes first-run compilation and cache size
worse; it remains a deployment issue and is not counted as a runtime win.

## Timing and profiling

- First-call compile, parameter packing, shape dispatch, and setup are
  synchronized and excluded on both equal-Triton paths.
- Before each complete run, repeated 4096×4096 BF16 GEMMs run for two seconds
  to bring the GPU out of the idle clock state. Each point then receives at
  least 100 ms warm-up and 200 ms measurement time.
- Timing uses `triton.testing.do_bench(return_mode="all")`: driver-level CUDA
  events are allocated before recording, L2 is cleared before every timed
  call, and all samples are retained for median/P10/P90/P95.
- The complete suite is executed in both `SAMU → RG-LRU` and
  `RG-LRU → SAMU` order. Raw samples from both orders are pooled before the
  published median and percentiles are recomputed.
- A cross-check demonstrated that `torch.cuda.Event` on the pinned Torch
  2.1/CUDA 11.8 stack quantized microsecond kernels into misleading
  0.16–0.22 ms values. Those measurements are not used in the final data.
- `torch.profiler` measures actual CUDA launch count and summed kernel time for
  representative operations.
- Triton compiler metadata records registers, spills, shared memory, threads,
  and warps for kernels that can be identified uniquely.
- Occupancy is a resource-limit estimate derived from compiler metadata and
  the H800 device attributes (65,536 32-bit registers and 233,472 bytes shared
  memory per SM). It is an upper bound, not a hardware occupancy counter.

Nsight Compute 2025.3.1 is installed, but the host has
`RmProfilingAdminOnly=1` and the container lacks the required performance
counter capability; collection fails with `ERR_NVGPUCTRPERM`. Hardware DRAM
bandwidth, DRAM/L2 bytes, SFU utilization, and SM utilization therefore remain `null`.
The reported “logical effective GB/s” is only algorithmic traffic lower bound
divided by latency. SFU numbers are source-derived transcendental-operation
counts, not measured utilization.

## Griffin training-scan axis

`run_training_on_device.py` follows Figure 8(a), Section 4.2 and Appendix D.2's
shape sweep: batch 8, 1024 real state scalars, and lengths 2048–16384. The paper
used a two-chip TPU-v3 pod; this suite uses one H800, so only the experimental
question and axes are reproduced.

Timing begins after projection. SAMU receives 1024 real write scalars and two
shared controls per token; RG-LRU receives 1024 token values and 2048 already
projected gate values. Both sides include gate/transition evaluation and scan.
Serial Triton, chunk-32 Triton and transparent materialized associative
references are reported. The latter are correctness-oriented PyTorch
references, not optimized associative GPU baselines.

The current result is forward only. It must not be labelled a complete training
step because backward, optimizer, multi-device all-reduce and ZeRO are absent.

`run_griffin_full_forward_proxy.py` then tests whether the isolated scan result
survives a full forward path. Both candidates reference the exact same
embedding, 24-layer MLP/norm shell and final-normalization tensor objects; only
the recurrent parameters and kernel change. At `B=8` and lengths 2K–16K, each
AB/BA order retains five CUDA-event samples after an untimed shape warm-up.
This is analogous to Figure 8(b)'s systems question, but it is a single ~1B
random-weight H800 forward proxy, not the paper's 400M/1B/7B training-step
sweep.

## Griffin Section 5-inspired system analysis

The current system analysis keeps Griffin Section 5's latency and throughput
axes but does not claim to reproduce its TPU result or trained checkpoint. Both
variants reference the exact same random-weight decoder-shell tensor objects:
width 2048, 24 layers, gated MLP expansion 3, tied 32,000-entry
embedding/output matrix, BF16 parameters, and an equal-size FP32 recurrent
cache. Only the temporal recurrence changes.

RG-LRU is constructed from the pinned official layer with the paper-default 16
gate groups. SAMU uses 16 grouped complex-write projections and direct shared
controls. This SAMU candidate has fewer temporal parameters and requires
retraining; the system shell is therefore not parameter-matched, and the report
states the exact counts instead of hiding the difference.

Latency fixes batch 16, excludes prompt prefill, and records cumulative
trajectories at 128, 256, 512, 1024, 2048 and 4096 decoded tokens after empty
and 4096-token prompts. Throughput first probes
`B ∈ {1,4,16,32,64,96,128,192,256}` for 32 steps, takes the union of each
model's top two batches, then completes 512/1024/2048/4096-token trajectories
for every finalist. Only completed trajectories can be selected as winners.

Every formal trajectory runs twice in AB and twice in BA model order. The site
takes the median inside each order, then combines the two order medians with
equal weight. Because both candidates are recurrent, neither cache grows with
prompt length; no recurrent-versus-attention KV-cache advantage is reused as a
SAMU-versus-RG-LRU claim.

## Reproduction

```bash
cd SAMU_GPU/benchmark
python run_equal_kernel_benchmarks.py \
  --output ../benchmark_results_equal_kernel \
  --rglru-source /path/to/recurrentgemma

python profile_equal_kernels.py \
  --output ../benchmark_results_equal_kernel/profiles.json \
  --rglru-source /path/to/recurrentgemma

python run_samu_gpu_adaptation.py \
  --output ../benchmark_results_griffin_complete/samu_gpu_adaptation.json \
  --rglru-source /path/to/recurrentgemma

python run_training_on_device.py \
  --output ../benchmark_results_griffin_complete/training_on_device.json \
  --rglru-source /path/to/recurrentgemma

python run_griffin_full_forward_proxy.py \
  --output ../benchmark_results_griffin_complete/griffin_full_forward_proxy.json \
  --rglru-source /path/to/recurrentgemma --repeats 5

python run_griffin_candidate_inference.py \
  --output ../benchmark_results_griffin_complete/griffin_candidate_inference.json \
  --rglru-source /path/to/recurrentgemma \
  --latency-repeats 2 --throughput-repeats 2
```

The exact environment, source hashes, raw rows, aggregate JSON/CSV, correctness
audit, and profiler output are committed with the site.
