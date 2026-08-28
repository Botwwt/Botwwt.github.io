# SAMU GPU interactive report

Pure static HTML/CSS/JavaScript report for the audited SAMU GPU architecture study.

## Local preview

```powershell
cd D:\Spectral_analysis\spectral\botwwtgithubio_remote_inspect\SAMU_GPU
python -m http.server 8000
```

Open <http://localhost:8000>.

KaTeX is loaded from jsDelivr. If the CDN is unavailable, all explanatory text and diagrams remain usable, while formula source stays visible as a fallback.

## Evidence labels

- `VERIFIED` / `FACT`: confirmed in the inspected repository or primary source.
- `PROPOSED`: GPU design proposed by this report; not an existing custom kernel.
- `HYPOTHESIS` / `INFERENCE`: systems interpretation that requires measurement.
- `BENCHMARK NEEDED`: no fair profiler-backed result was found.

## Main repository sources inspected

- `spectral/rtus/research_extensions/equilibrium_polar/canonical_grouped_samu_math.py`
- `spectral/rtus/research_extensions/equilibrium_polar/canonical_grouped_samu_flax.py`
- `spectral/rtus/research_extensions/equilibrium_polar/equilibrium_polar_math.py`
- `spectral/rtus/research_extensions/equilibrium_polar/native_timescale_radial_math.py`
- `spectral/rtus/research_extensions/equilibrium_polar/official_adapter.py`
- `spectral/rtus/research_extensions/equilibrium_polar/run_canonical_grouped_samu_sweep.py`
- `spectral/hippocampal_adaptive_memory_validation/experiments/official_baseline_audit_20260823/formal/models.py`
- `spectral/hippocampal_adaptive_memory_validation/experiments/official_baseline_audit_20260823/engineering/scan.py`
- `spectral/hippocampal_adaptive_memory_validation/experiments/official_baseline_audit_20260823/engineering/benchmark_torch.py`

No SAMU training or experiment code is modified by this static report.
