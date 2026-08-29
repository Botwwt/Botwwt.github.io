# SAMU GPU · official RG-LRU comparison

A static research report and reproducible H800 benchmark suite for SAMU and
RG-LRU. The primary opponent follows Griffin equations 1–4, the pinned
RecurrentGemma PyTorch semantics, and the paper-default 16 block-diagonal gate
groups. No additional recurrent baseline is included in this report.

## What is measured

Four tracks are kept separate:

1. **Official-structure decode.** At `D=2048`, both paths time RMSNorm plus one
   recurrent update with equal 8 KiB FP32 state per sequence and two logical
   launches. The SAMU-16 direct-control candidate is `1.032x–1.650x` faster
   than official RG-LRU-16 for every measured batch in `{1,4,16,64,128}`.
2. **Griffin training-scan axis.** With `B=8`, 1024 real state scalars and
   lengths 2K–16K, SAMU chunk-32 is `4.91x–5.69x` faster than the equal-level
   RG-LRU-16 chunk path. Serial paths differ by only about `1.08x`, so the
   larger gain comes from shared control plus chunk parallelization. In the
   exact shared ~1B shell, the complete forward advantage is only
   `1.026x–1.047x`; common embedding/MLP/norm work dominates. Neither
   measurement is a full training step because backward is absent.
3. **Griffin inference-speed axis.** In the random-weight ~1B proxy, both
   candidates reference the exact same embedding, 24-layer MLP/norm shell and
   final-normalization tensors. Official RG-LRU-16 is `1.023x–1.038x` faster on
   the batch-16 continuous decode trajectories and has `1.016x–1.020x` higher
   completed-trajectory throughput. Each AB/BA order contains two complete
   repetitions. The isolated recurrence win has not yet become a full-system
   win.
4. **Artificial parameter-matched microbenchmark.** Dense SAMU versus
   RG-LRU-2 at `D=128` is retained only to isolate equation cost. RG-LRU-2 is
   not the paper-default architecture and is not used for the main claim.

SAMU-16 direct control changes the architecture: two normalized activation
channels provide the shared controls, and the complex write projection is
grouped 16 ways. It requires retraining and quality validation. Current results
are systems evidence, not a perplexity or downstream-quality result.

## Preview

```powershell
cd D:\Spectral_analysis\spectral\botwwtgithubio_remote_inspect\SAMU_GPU
python -m http.server 8000
```

Open <http://localhost:8000>. A web server is required because the figures load
the committed JSON evidence.

## Reproduce the current H800 tracks

```bash
cd SAMU_GPU/benchmark

python run_samu_gpu_adaptation.py \
  --output ../benchmark_results_griffin_complete/samu_gpu_adaptation.json \
  --rglru-source /path/to/recurrentgemma

python run_training_on_device.py \
  --output ../benchmark_results_griffin_complete/training_on_device.json \
  --rglru-source /path/to/recurrentgemma

python run_griffin_candidate_inference.py \
  --output ../benchmark_results_griffin_complete/griffin_candidate_inference.json \
  --rglru-source /path/to/recurrentgemma
```

Calibration and final samples are separate. Final microkernel timing is run in
forward and reverse model order and retains every Triton driver-event sample.
Compilation, packing, dispatch and warm-up are outside the steady-state region.

## Repository map

- `benchmark/triton_rglru.py`: official-equation RG-LRU serial, chunk and fused
  decode Triton kernels, including reset and eager-BF16 behavior.
- `benchmark/triton_samu.py`: dense SAMU serial/chunk/decode kernels and bounded
  spectral fast paths.
- `benchmark/triton_samu_grouped.py`: grouped complex write, fused
  RMSNorm/controller and grouped decode candidate.
- `benchmark/run_samu_gpu_adaptation.py`: D=2048 calibration, correctness,
  compiler resources and counterbalanced timing.
- `benchmark/run_training_on_device.py`: Griffin Figure 8(a)-shaped forward
  scan experiment on one H800.
- `benchmark/run_griffin_candidate_inference.py`: random-weight ~1B shell,
  continuous decode and bounded full-trajectory throughput.
- `benchmark_results_griffin_complete/`: raw timing, calibration, correctness,
  resource and environment JSON loaded by the report.
- `BENCHMARK_METHODOLOGY.md`: equations, matching rules and evidence limits.
- `SAMU_CANONICAL_AUDIT.md`: canonical SAMU source audit.

Nsight Compute counters are unavailable because the host enforces
`RmProfilingAdminOnly=1`; DRAM, L2, SFU utilization and achieved occupancy stay
`null`. Compiler registers, spills and a resource-limit occupancy upper bound
are reported separately and are never presented as hardware counters. Current
custom kernels do not include backward, multi-device all-reduce or ZeRO.
