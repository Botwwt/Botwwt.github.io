# SAMU GPU · equal-kernel RG-LRU comparison

A static research course and benchmark lab connecting the audited SAMU
recurrence to GPU dataflow, an official-equation Triton RG-LRU baseline, and a
pinned official Mamba-3 context line.

The current result set contains 58 rows: 54 measured and four explicitly
unsupported Mamba-3 decode configurations. The primary comparison fixes
`d_model=128`, near-identical parameter count (16,772 vs 16,768), and identical
512-byte FP32 recurrent state. It never uses the official RecurrentGemma Python
scan as the equal-kernel opponent.

On the recorded RTX 3090 run, SAMU wins every tested Track-A prefill point:
`1.04x` to `2.03x` at batch 1 and `1.11x` to `1.81x` in the batch-scaling
slice. One-token decode is a tie at the 1.024 microsecond event resolution.
These are Triton driver-event medians with preallocated events and an L2 clear
before every timed call; compile, packing, and Python shape dispatch are outside
the timed region. The Mamba-3 line is a same-width context track, not an
architecture-fair speedup claim.

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
  --rglru-source /path/to/recurrentgemma \
  --mamba-source /path/to/mamba

python profile_equal_kernels.py \
  --output ../benchmark_results_equal_kernel/profiles.json \
  --rglru-source /path/to/recurrentgemma \
  --mamba-source /path/to/mamba
```

## Repository map

- `benchmark/triton_samu.py`: packed projection, serial/chunk prefill, fused
  decode, bounded phase polynomial, and FP32 cache.
- `benchmark/triton_rglru.py`: official-equation serial/chunk/decode Triton
  kernels with reset and eager-BF16 semantics.
- `benchmark/run_equal_kernel_benchmarks.py`: two-track correctness and timing
  runner.
- `benchmark/audit_equal_microbench.py`: event-method and cold-cache timing
  audit for short kernels.
- `benchmark/audit_samu_dispatch.py`: reproducible chunk-size dispatch sweep.
- `benchmark/audit_samu_fastpath.py`: certified bounded-exponential A/B audit.
- `benchmark/audit_samu_compressed_p.py`: experimental compressed-summary A/B
  audit; the slower path remains disabled by default.
- `benchmark/profile_equal_kernels.py`: launch and Triton compiler-resource
  profiler.
- `benchmark_results_equal_kernel/`: raw samples, aggregate files, environment,
  correctness audit, and profile evidence loaded by the site.
- `BENCHMARK_METHODOLOGY.md`: equations, matching rules, timing, and caveats.
- `SAMU_CANONICAL_AUDIT.md`: canonical SAMU model/source audit.

`Measured`, `Verified`, `Derived`, `Paper Fact`, `Proposed`, and `Hypothesis`
labels separate evidence from interpretation. Hardware DRAM/SFU counters are
`N/A` because Nsight Compute was unavailable; logical traffic and static SFU
counts are not presented as hardware measurements.
