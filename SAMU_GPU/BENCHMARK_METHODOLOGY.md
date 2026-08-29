# SAMU vs official-source RG-LRU methodology

## Scope

This benchmark measures inference implementation behavior on one RTX 3090. It asks:

1. Can SAMU's two coherent token controls be mapped to a tighter GPU execution path than the pinned official RG-LRU source?
2. Where does serial state-stationary execution cross over to chunk summary/prefix/replay?
3. Which tested chunk size wins at short, medium, and long sequence lengths?
4. Does a single-launch SAMU decode path reduce total one-token update latency?

It does not measure model quality or training/backward performance.

## Compared implementations

- **SAMU:** custom Triton inference kernels in `benchmark/triton_samu.py`. Prefill uses one packed BF16 projection followed by either serial recurrence or three-stage chunk execution. Decode fuses write projection, controller projection, bounded control, transition reconstruction, and state update into one launch.
- **RG-LRU:** the unmodified `RGLRU` layer from Google DeepMind RecurrentGemma commit `2efa84dac0e68e63547a27a18fa943c98f1c312e`. Its public PyTorch `rnn_scan` uses a Python sequence loop for `L>1` and an optimized source branch for `L=1`.

The comparison therefore answers which available execution path ran faster. It is not a claim against a hypothetical custom optimized RG-LRU kernel.

## Matching protocol

Every paired row uses:

- equal model/input width `d_model`;
- `d_model = 2M`;
- SAMU `M` complex FP32 modes = `2M` real FP32 state scalars;
- RG-LRU width `2M` = `2M` real FP32 state scalars;
- equal recurrent-state bytes per batch;
- BF16 input/output and FP32 recurrent accumulation/cache.

The main point is `d=128`, `M=64`: both recurrent states use 128 FP32 scalars, or 512 bytes per batch. Parameter and arithmetic counts are recorded but are not asserted to be exactly equal.

## Sweep

- Prefill length: `128, 512, 2048, 8192, 32768, 65536`, batch 1, `d=128`, `M=64`.
- SAMU kernel crossover: serial and chunk sizes `8, 16, 32`.
- Prefill batch: `1, 2, 4, 8, 16` at `L=512`.
- Equal-state size: `M=64, 128, 256` with `d=2M` at `L=512`.
- Decode batch: `1, 4, 16, 64`.

The measured dispatcher uses serial for `L<=256`, C16 for `256<L<=512`, and C32 for longer sequences on this GPU. These thresholds are not extrapolated to other devices.

## Correctness and precision

The exact inference policy uses one packed BF16 projection, BF16 outputs, and FP32 controller/recurrent calculations. Before timing, the runner compares:

- Triton serial prefill against the packed PyTorch reference;
- Triton C16 chunk prefill against the same reference;
- fused Triton decode against a one-step reference with nonzero initial state.

Maximum and mean absolute errors are stored in `benchmark_results_samu_rg/correctness.json` and copied into `summary.json`.

## Timing

- The first call is synchronized and recorded as `compile_setup_seconds`, outside samples.
- Five warmup calls precede measurements.
- Synchronized CUDA events measure device latency.
- SAMU uses 30 samples; RG-LRU uses 15 samples, or 7 for the two longest rows.
- Decode uses 100 inner iterations per sample so sub-millisecond launch latency is measurable.
- Every row stores all raw samples plus median, mean, p10, p90, p95, standard deviation, minimum, maximum, peak PyTorch allocation, and throughput.
- Each row is atomically written immediately, so `--resume` is safe.

`peak_allocated_bytes` is a PyTorch allocator observation, not total board memory. No DRAM/L2/occupancy/register/Tensor Core/SFU/SM counter is claimed because Nsight Compute was unavailable.

## Reproduction

```bash
cd SAMU_GPU/benchmark
python run_samu_vs_rglru.py \
  --output ../benchmark_results_samu_rg \
  --rglru-source /path/to/recurrentgemma \
  --resume
```

The exact executed command, environment, source hashes, raw rows, CSV, and JSON are committed with the site.
