# Equal-kernel SAMU / RG-LRU methodology

## Question and scope

This experiment measures inference execution on one RTX 3090. Its primary
question is whether SAMU's low-dimensional transition control produces a
tighter GPU path than an equally optimized RG-LRU kernel. It does not measure
model quality, training, or backward performance.

There are two deliberately separate tracks:

- **Track A — equal Triton:** custom Triton SAMU and RG-LRU inference paths,
  both with serial/chunk prefill and fused one-token decode.
- **Track B — official context:** pinned, unmodified public RecurrentGemma and
  Mamba-3 implementations. Mamba-3 is same-width only; its parameters and
  recurrent cache are not matched to Track A.

Only Track A is used for the SAMU-versus-RG-LRU architecture conclusion.

## Official RG-LRU semantics

The Triton implementation in `benchmark/triton_rglru.py` follows Griffin
equations 1–4 and the public RecurrentGemma layer at commit
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
two-head block-diagonal projection geometry, matches eager-BF16 pointwise
rounding, and keeps the accumulator/cache in FP32.

The equation, block geometry, reset rule, and precision policy are official.
Packing both gate projections into one BMM, the serial/chunk Triton scans, the
fused one-token decode kernel, and the shape dispatcher are this project's
optimizations. Affine chunk scan is generic algebra rather than an official
RG-LRU optimization or a new algorithmic invention claimed here.

Correctness is checked directly against
`recurrentgemma.torch.layers.RGLRU`, including a nonzero initial cache,
mid-sequence resets at `t=5` and `t=77`, and mixed reset/non-reset decode.
Serial, C16, decode output, and final-cache errors are stored in
`benchmark_results_equal_kernel/correctness.json`.

## Track A matching

The main point fixes:

- `d_model = 128`, SAMU `M = 64` complex modes;
- SAMU 16,772 parameters versus RG-LRU 16,768 parameters;
- 128 FP32 recurrent scalars, or 512 bytes per batch element, on both sides;
- BF16 input/output with FP32 recurrent accumulation/cache;
- one packed projection followed by recurrence work;
- four CUDA launches for C32 prefill and one launch for fused decode.

The prefill sweep uses `L = 128, 512, 2048, 8192, 32768, 65536`. Batch
scaling is measured at `L=512`; decode uses `B = 1, 4, 16, 64`; explicit
chunk sizes 8, 16, and 32 are retained at `L=2048` and `L=65536`.

The RTX 3090 dispatch is SAMU C8 through `L=256`, C16 through `L=512`, and
C32 beyond that. At `L<=128`, the packed parameters must pass analytic bounds
before the bounded-exp polynomial is enabled; otherwise it falls back to the
general exponential. RG-LRU uses C8 through `L=256`, C16 through `L=1024`,
and C32 beyond that. Dispatch is resolved before timing on both sides. These
thresholds are device-specific measured choices, not claims about other GPUs.

## Track B boundary

Official Mamba-3 is loaded from state-spaces/mamba commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa`, using its BF16 SISO Triton
prefill path at `d_model=128`. At this shape it has 118,920 parameters and a
66,816-byte native cache per batch element, versus SAMU's 16,772 parameters
and 512-byte recurrent state. It is therefore an implementation-context line,
not an equal-model comparison. Its module `step()` directly asserts that the
CuTeDSL `mamba3_step_fn` is available and documents that path as H100-only.
The same repository contains a standalone Triton SISO step file, but the
pinned `Mamba3.step()` does not call it. Wiring that file into the module would
be a custom integration, so unmodified official decode remains explicitly
`unsupported` on this RTX 3090.

The official RecurrentGemma PyTorch result is retained as a formula/source
reference. Its Python sequence loop is not used as evidence that RG-LRU's
architecture is intrinsically slow.

## Timing and profiling

- First-call compile, parameter packing, shape dispatch, and setup are
  synchronized and excluded on both equal-Triton paths.
- Timing uses `triton.testing.do_bench(return_mode="all")`: driver-level CUDA
  events are allocated before recording, L2 is cleared before every timed
  call, and all samples are retained for median/P10/P90/P95.
- A cross-check demonstrated that `torch.cuda.Event` on the pinned Torch
  2.1/CUDA 11.8 stack quantized microsecond kernels into misleading
  0.16–0.22 ms values. Those measurements are not used in the final data.
- `torch.profiler` measures actual CUDA launch count and summed kernel time for
  representative operations.
- Triton compiler metadata records registers, spills, shared memory, threads,
  and warps for kernels that can be identified uniquely.
- Occupancy is a resource-limit estimate derived from compiler metadata and
  RTX 3090 SM limits. It is not a hardware occupancy counter.

Nsight Compute and Nsight Systems were unavailable. Hardware DRAM bandwidth,
DRAM/L2 bytes, SFU utilization, and SM utilization therefore remain `null`.
The reported “logical effective GB/s” is only algorithmic traffic lower bound
divided by latency. SFU numbers are source-derived transcendental-operation
counts, not measured utilization.

The Mamba profiler cache contains 37 autotune candidates. Because
`torch.profiler` cannot uniquely map the executed event back to one candidate,
the Mamba register/occupancy field is reported as `N/A`, not selected after the
fact.

## Reproduction

```bash
cd SAMU_GPU/benchmark
python run_equal_kernel_benchmarks.py \
  --output ../benchmark_results_equal_kernel \
  --rglru-source /path/to/recurrentgemma \
  --mamba-source /path/to/mamba

python profile_equal_kernels.py \
  --output ../benchmark_results_equal_kernel/profiles.json \
  --rglru-source /path/to/recurrentgemma \
  --mamba-source /path/to/mamba
```

The exact environment, source hashes, raw rows, aggregate JSON/CSV, correctness
audit, and profiler output are committed with the site.
