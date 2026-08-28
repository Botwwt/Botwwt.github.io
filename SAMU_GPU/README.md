# SAMU GPU Interactive Course & Benchmark Lab

A static, modular research course that connects the audited SAMU recurrence to GPU dataflow, warp/CTA mappings, chunked affine scan, competing architectures, and real RTX 3090 measurements.

The page never substitutes theoretical traffic for measured DRAM traffic and never presents the official-source serial RG-LRU reference as a production kernel. Unsupported paths remain visible in the result set.

## Preview the site

```powershell
cd D:\Spectral_analysis\spectral\botwwtgithubio_remote_inspect\SAMU_GPU
python -m http.server 8000
```

Open <http://localhost:8000>. A web server is required because the benchmark lab fetches JSON files.

## Verify benchmark code

```bash
cd SAMU_GPU/benchmark
python test_correctness.py
python run_benchmarks.py --preset quick \
  --output ../benchmark_results \
  --mamba-source /path/to/mamba \
  --rglru-source /path/to/recurrentgemma --resume
```

Use `--preset full` for the resumable B/L/M/dtype/workload sweep. Each configuration is saved immediately to `benchmark_results/raw/`; `aggregate.py` rebuilds `summary.json` and `summary.csv` without rerunning GPU work.

## Repository map

- `index.html`: semantic page shell.
- `src/lessons.js`: the continuous 24-lesson narrative.
- `src/visualizations/`: canvas/DOM interactive systems figures.
- `src/benchmark-lab.js`: JSON-backed charts, filters, environment, and profiler views.
- `benchmark/`: audited PyTorch prototypes, official-source loaders, correctness, runner, profiler, and aggregator.
- `benchmark_results/`: raw samples, aggregate files, environment metadata, and profiler evidence.
- `SAMU_CANONICAL_AUDIT.md`: canonical model/source audit.
- `BENCHMARK_METHODOLOGY.md`: timing, fairness, matching, and caveats.

## Evidence language

`Verified Code`, `Measured`, `Paper Fact`, `Derived`, `Proposed`, and `Hypothesis` labels deliberately separate current facts from future kernel ideas.
