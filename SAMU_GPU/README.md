# SAMU GPU · equal-kernel RG-LRU comparison

A static research report and benchmark lab connecting the audited SAMU
recurrence to GPU dataflow and an official-equation Triton RG-LRU baseline.

The public comparison contains 38 measured rows. It fixes
`d_model=128`, near-identical parameter count (16,772 vs 16,768), and identical
512-byte FP32 recurrent state. It never uses the official RecurrentGemma Python
scan as the equal-kernel opponent.

On the counterbalanced H800 PCIe run, the crossover is the main result rather
than an all-shape win. At batch 1, RG-LRU is faster at `L=128`; the two kernels
are essentially tied at `L=512` and `L=2048`; SAMU is about `1.16x` faster at
`L=8192`/`32768` and `1.33x` faster at `L=65536`. At `L=512`, SAMU's advantage
grows from roughly parity at `B=4` to `1.46x` at `B=16` and `2.18x` at `B=64`.
Conversely, the tiny one-launch `d_model=128` decode microkernel favors RG-LRU
by about `1.31x–1.40x`. These are pooled Triton driver-event samples from both
execution orders; compile, packing, calibration, and shape dispatch are outside
the timed region.

The separate 1.07B-parameter decoder proxy shows why the microkernel and system
numbers must not be conflated.  After pooling both execution orders, SAMU uses
about `0.8%–1.3%` less time for the six empty-prefix, batch-16 decode
trajectories and `2.1%–2.7%` less after a 4096-token prefix.  Within the
predeclared `B≤128` search, its complete-trajectory throughput is
`1.003x–1.013x` that of RG-LRU.  An independently labelled 8-step stress test
reaches 102,521.7 token/s for SAMU and 100,905.3 token/s for RG-LRU; those short
probes are not substituted for full trajectories.

## Preview the site

```powershell
cd D:\Spectral_analysis\spectral\botwwtgithubio_remote_inspect\SAMU_GPU
python -m http.server 8000
```

Open <http://localhost:8000>. A web server is required because the benchmark
lab fetches JSON files.

## Reproduce

```bash
cd SAMU_GPU/benchmark
python run_equal_kernel_benchmarks.py \
  --output ../benchmark_results_equal_kernel \
  --rglru-source /path/to/recurrentgemma

python profile_equal_kernels.py \
  --output ../benchmark_results_equal_kernel/profiles.json \
  --rglru-source /path/to/recurrentgemma

# Griffin Section 5-inspired system proxy: about 1.07B parameters, B=16
# latency after 0/4096-token prompts, and a single-GPU batch search.
python run_griffin_section5.py \
  --output ../benchmark_results_griffin_section5/summary.json \
  --rglru-source /path/to/recurrentgemma
```

## Repository map

- `benchmark/triton_samu.py`: packed projection, serial/chunk prefill, fused
  decode, bounded phase polynomial, and FP32 cache.
- `benchmark/triton_rglru.py`: official-equation serial/chunk/decode Triton
  kernels with reset and eager-BF16 semantics.
- `benchmark/run_equal_kernel_benchmarks.py`: equal-kernel correctness and timing
  runner.
- `benchmark/audit_equal_microbench.py`: event-method and cold-cache timing
  audit for short kernels.
- `benchmark/audit_samu_dispatch.py`: reproducible chunk-size dispatch sweep.
- `benchmark/audit_samu_fastpath.py`: certified bounded-exponential A/B audit.
- `benchmark/audit_samu_compressed_p.py`: experimental compressed-summary A/B
  audit; the slower path remains disabled by default.
- `benchmark/profile_equal_kernels.py`: launch and Triton compiler-resource
  profiler.
- `benchmark/profile_system_temporal.py`: launch and compiler-resource audit
  for the width-2048 temporal paths used by the system proxy.
- `benchmark/run_griffin_section5.py`: equal-shell 1.07B random-weight systems
  proxy following the latency and bounded full-trajectory throughput axes in
  Griffin Section 5.
- `benchmark/run_system_saturation.py`: separate large-batch short-probe stress
  test; its samples are not presented as full-trajectory throughput.
- `benchmark_results_equal_kernel/`: raw samples, aggregate files, environment,
  correctness audit, and profile evidence loaded by the site.
- `benchmark_results_griffin_section5/`: full autoregressive trajectories and
  batch-search results. This is a systems measurement, not a trained-model
  quality comparison or a reproduction of the paper's TPU numbers.
- `BENCHMARK_METHODOLOGY.md`: equations, matching rules, timing, and caveats.
- `SAMU_CANONICAL_AUDIT.md`: canonical SAMU model/source audit.

`Measured`, `Verified`, `Derived`, `Paper Fact`, `Proposed`, and `Hypothesis`
labels separate evidence from interpretation. Hardware DRAM/SFU counters are
`N/A`: Nsight Compute is installed on the H800 node, but host policy blocks
performance counters (`ERR_NVGPUCTRPERM`). Logical traffic and static SFU
counts are not presented as hardware measurements.

Steady-state measurements exclude Triton JIT. SAMU currently specializes four
control constants per layer, so the 24-layer cold start compiles multiple cubin
variants. This is tracked as deployment overhead, not hidden inside a speedup.
