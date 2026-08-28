# SAMU GPU benchmark methodology

## Questions

The benchmark exists to measure implementation behavior, not to infer model quality. It asks:

1. What is the cost of coherent SAMU control relative to a real gated recurrence under the same PyTorch execution strategy?
2. When does a serial state-stationary recurrence lose to a time-parallel formulation?
3. How much do phase factorization and control precomputation change wall-clock time?
4. How does the best available official Mamba-3 implementation compare on supported native configurations?

## Tracks

### Track A — best available implementation

- SAMU: fastest measured implementation in this repository, named precisely.
- RG-LRU: the best implementation actually available in the pinned official RecurrentGemma source. In this environment that is an official-source PyTorch serial reference, labelled exactly that way; it is not presented as a production kernel.
- Mamba-3: official `state-spaces/mamba` source snapshot. Import/build/runtime failures are recorded as `unsupported` with the failure reason.
- Mamba-2: official `state-spaces/mamba` source snapshot. Its fused forward requires `causal-conv1d`; the missing/failed extension is retained as `unsupported`, never substituted with a simplified competitor.

This track answers what is fastest among the implementations that actually ran. It does not isolate architectural cost.

### Track B — architectural apples-to-apples

Reference SAMU and reference RG-LRU recurrence cores use the same PyTorch version, device, dtype, timing harness, input layout, synchronization, and scan/serial execution class. Every result is labelled `reference`, not production.

Mamba-3 is not replaced by a simplified recurrence in Track B. If the official block cannot be compared under the same core interface, the cell is `N/A`.

## Matching protocols

- `same_width`: equal input/model width; native state configurations are displayed explicitly.
- `state_bytes`: SAMU `M` complex modes are matched against approximately `2M` real state scalars. Padding or residual mismatch is reported.
- `parameter_approx`: write/transition parameter counts are made approximately equal and the residual difference is reported.
- `best_native`: each official implementation uses a supported/recommended native configuration. This is not an architecture-isolating comparison.

## Workloads

- `forward`: full sequence forward only.
- `forward_backward`: forward, scalar loss, backward.
- `decode`: one recurrent update with persistent state tensors outside the timed setup.
- `prefill`: full known sequence. Backend is recorded (`serial`, `tree_scan`, `official_optimized`, and so on).

Backward-only is derived only when the harness can isolate a previously constructed graph without reusing an invalid graph. Otherwise it is `N/A`, never `forward_backward - forward` presented as a direct measurement.

## Timing

- CUDA events bracket only the timed GPU region.
- Device synchronization occurs before and after a measurement series.
- Warmup runs precede recorded runs.
- Very short kernels are repeated inside one sample until the aggregate duration is measurable; `inner_iterations` is saved.
- Raw samples are written after every completed configuration.
- Aggregate fields: median, mean, p10, p90, p95, standard deviation, minimum, maximum, and sample count.
- OOM, unsupported dtype/backend, dependency failure, and runtime failure are first-class rows.

## Precision

- FP32 is the numerical reference/performance baseline.
- BF16 is measured only for operations/backends that support it.
- FP16 is optional and labelled separately.
- Transcendental/control calculations may deliberately execute in FP32 while state/write storage uses a lower precision; this mixed-precision policy is recorded.

## Memory and profiler fields

- `peak_allocated_bytes` comes from PyTorch CUDA allocator counters for the current process. It is not total board memory.
- `logical_bytes` is a formula-derived data-volume model and is explicitly not measured DRAM traffic.
- `dram_bytes`, `l2_bytes`, occupancy, registers/thread, Tensor Core utilization, SFU utilization, and SM utilization are `N/A` unless a profiler actually reports them.
- Torch profiler kernel tables are stored when available and identified as Torch-profiler observations.

## Sweep policy

`quick` covers representative shapes and completes in minutes. `full` enumerates the requested batch, sequence, state, dtype, workload, and design grids with sane memory guards, resume, and per-result persistence. Configurations that exceed a conservative allocation estimate are written as `skipped_memory_guard`; they are not silently deleted.

## Reproduction

```bash
cd SAMU_GPU
python benchmark/run_benchmarks.py --preset quick --output benchmark_results \
  --mamba-source /path/to/mamba --rglru-source /path/to/recurrentgemma --resume
python benchmark/run_benchmarks.py --preset full --output benchmark_results \
  --mamba-source /path/to/mamba --rglru-source /path/to/recurrentgemma --resume
python benchmark/aggregate.py benchmark_results
```

The exact commands actually used are also stored in `benchmark_results/run_manifest.json`.
