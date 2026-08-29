# SAMU GPU · equal-kernel RG-LRU comparison

A static research course and benchmark lab connecting the audited SAMU
recurrence to GPU dataflow, an official-equation Triton RG-LRU baseline, and a
pinned official Mamba-3 context line.

The current result set contains 58 rows: 54 measured and four explicitly
unsupported Mamba-3 decode configurations. The primary comparison fixes
`d_model=128`, near-identical parameter count (16,772 vs 16,768), and identical
512-byte FP32 recurrent state. It never uses the official RecurrentGemma Python
scan as the equal-kernel opponent.

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
