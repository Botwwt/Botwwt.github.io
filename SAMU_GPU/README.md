# SAMU GPU · focused RG-LRU comparison

A static research course and benchmark lab connecting the audited SAMU recurrence to GPU dataflow, fused Triton inference kernels, and the pinned official RecurrentGemma RG-LRU source.

The main result set contains 53/53 measured RTX 3090 configurations. It keeps model width and FP32 recurrent-state bytes equal, and never presents the official-source Python scan as a production RG-LRU kernel.

## Preview the site

```powershell
cd D:\Spectral_analysis\spectral\botwwtgithubio_remote_inspect\SAMU_GPU
python -m http.server 8000
```

Open <http://localhost:8000>. A web server is required because the benchmark lab fetches JSON files.

## Reproduce the focused benchmark

```bash
cd SAMU_GPU/benchmark
python run_samu_vs_rglru.py \
  --output ../benchmark_results_samu_rg \
  --rglru-source /path/to/recurrentgemma --resume
```

The runner checks serial, C16 chunk, and fused-decode correctness before timing. Each configuration is saved immediately under `benchmark_results_samu_rg/raw/`; first-call compile/setup is excluded from CUDA-event samples.

## Repository map

- `index.html`: semantic page shell.
- `src/lessons.js`: the continuous 24-lesson narrative.
- `src/visualizations/`: canvas/DOM interactive systems figures.
- `src/benchmark-lab.js`: focused length/batch/state/decode/kernel-crossover charts.
- `benchmark/triton_samu.py`: packed projection, serial prefill, chunked summary/prefix/replay, and fused decode kernels.
- `benchmark/run_samu_vs_rglru.py`: correctness, equal-state RG-LRU comparison, resume, raw samples, and aggregation.
- `benchmark_results_samu_rg/`: primary raw samples, aggregate files, environment, and correctness evidence.
- `benchmark_results/`: historical multi-model experiment archive; it is not loaded by the current page.
- `SAMU_CANONICAL_AUDIT.md`: canonical model/source audit.
- `BENCHMARK_METHODOLOGY.md`: timing, fairness, matching, and caveats.

## Evidence language

`Verified Code`, `Measured`, `Paper Fact`, `Derived`, `Proposed`, and `Hypothesis` labels deliberately separate current facts from future kernel ideas.
