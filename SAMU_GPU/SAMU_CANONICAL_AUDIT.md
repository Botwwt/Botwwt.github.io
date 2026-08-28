# SAMU canonical implementation audit

Audit date: 2026-08-29 (Asia/Shanghai)

This file is the factual boundary for the interactive course and benchmark harness. It distinguishes the current repository implementation from proposed GPU kernels.

## Source identity

The SAMU research extension is an untracked extension on top of RTU repository commit `be54e13b91edcd7988dd1764f8f2d412ca2db856`. Because it is not represented by that Git commit, the exact audited files are identified by SHA-256:

| File | SHA-256 |
|---|---|
| `canonical_grouped_samu_math.py` | `164500431103d982813fcff505e2828d4b8714e5891b9051c00150d4febc2477` |
| `canonical_grouped_samu_flax.py` | `568e8f80e96313e9ce11556b8c2d4d30e3d14a1558cfae7bcad642389ed11ea8` |
| `official_adapter.py` | `ebd1552aec70b317f25aa49472ec4336826e997a4701aa2a5baa0dbd3f9b0c8e` |
| formal-task `models.py` | `2a92b2fca69bf8e3349a588da8956ee83aaa56616c9b4a72af207be21dd05d4a` |
| formal-task `engineering/scan.py` | `410fc6d4610f4cd4a3f31714f1d242440f29570cdc7e8e9cdff79b2f99bab061` |

External source snapshots used for comparison:

- official `state-spaces/mamba`: `e9594ce1c732d97440f0332fdc43170a2294dbfa`
- official `google-deepmind/recurrentgemma`: `2efa84dac0e68e63547a27a18fa943c98f1c312e`

## 1. Current recurrence

For complex mode `j`, the grouped canonical implementation computes

```text
z[j,t+1] = transition[j,t] * z[j,t] + gamma[j] * write[j,t]
transition[j,t] = exp(-nu[j] * exp(eta[group(j),t]))
                  * exp(i * (theta[j] + delta[group(j),t]))
```

where `nu = exp(nu_log)` and `theta = exp(theta_log)` are trainable, mode-local base spectral parameters. `groups=1` is canonical coherent SAMU; larger group counts are a nested extension up to one controller group per mode.

The course uses `c_t := eta_t` and `d_t := delta_t` for the single-group (`G=1`) case. These are equivalent GPU coordinates, not unconstrained controller outputs.

## 2. Controller construction

For every group, selectors are normalized affine directions applied to an augmented selector input (input plus a constant bias coordinate), followed by `tanh`:

```text
s_phase = tanh([selector_x, 1] @ normalized(phase_direction))
s_radial = tanh([selector_x, 1] @ normalized(radial_direction))
r = s_radial / (1 + s_radial^2)
delta = (1/sqrt(M)) * tanh(raw_phase_amplitude) * s_phase
eta   = (1/sqrt(M)) * tanh(raw_radial_amplitude) * r
```

Therefore the dynamic transition is bounded and width-normalized. Describing `c_t,d_t` as arbitrary free scalars would be inaccurate.

## 3. Write path and gamma

The JAX canonical write is

```text
write = write_x @ wx1 + i * (write_x @ wx2)
gamma[j] = sqrt(1 - exp(-2 * nu[j])) + 1e-8
```

`gamma` uses the base retention, not the token-varying effective retention. Within one fixed parameter version / forward, the write matrix is time-independent and input-independent; it is not an eternal constant because `wx1`, `wx2`, and `nu` are trainable.

The audited formal-task PyTorch wrapper differs: `gamma_log` is initialized from the base radius but is then an independent trainable parameter. It is not guaranteed to remain equal to canonical `gamma(nu)` after optimization. The course presents JAX canonical math and labels formal-wrapper measurements separately.

## 4. PPO versus supervised inputs

- PPO `ObservationControlledRealTimeActorCritic`: `write_x_t = shared_repr` from the observation MLP, while `selector_x_t = raw observation`. They are deliberately separate.
- Supervised / H64 grouped sweeps: the raw task input is used directly by the recurrent model and therefore serves the write and selector roles in that experimental wrapper.

The course must not claim that PPO uses the same tensor for both paths.

## 5. State dimensions and convention

`n_hidden` in the canonical Flax cell is the number of complex modes. This audit writes it as `M`.

```text
M complex modes = 2M real recurrent state scalars
```

All state-memory comparisons must match real scalar bytes, not compare `M` complex SAMU modes to `M` real RG-LRU states as though they were equal.

## 6. Initialization and trainability

- recurrent real and imaginary state are initialized to zero;
- `nu_log`, `theta_log`, `wx1`, `wx2`, controller directions, and controller amplitudes are trainable;
- controller amplitudes are initialized to zero, so canonical SAMU begins at the RTU working null while retaining first-order amplitude credit in the exact-RTRL implementation.

## 7. Existing scan implementation

The formal-task PyTorch path already contains a work-efficient inclusive affine tree scan with composition

```text
(a2,b2) o (a1,b1) = (a2*a1, a2*b1+b2)
```

It materializes a complex multiplier for every `time × mode` element. It is a correctness and framework baseline, not the proposed compressed `(C,G,D,q)` custom GPU scan.

## 8. Derived compressed transition

For single-group canonical SAMU and a segment `S`:

```text
C = |S|
G = sum(t in S) exp(c_t)
D = sum(t in S) d_t
P[j,S] = exp(-nu[j] * G) * exp(i * (C * theta[j] + D))
```

This derivation is exact for the audited recurrence after the identification `c=eta`, `d=delta`. The full affine summary is `(C,G,D,q)`, where `q` is a complex vector of length `M`. Only transition metadata is `O(1)`; the whole summary remains `O(M)`.

## 9. Previous-page corrections retained in the rebuild

The previous page already corrected several common errors and the rebuild must preserve them:

- `c,d` are bounded canonical coordinates, not arbitrary free control;
- `gamma` semantics differ between canonical JAX and the formal PyTorch wrapper;
- PPO write and selector inputs differ;
- state width is `2M` real scalars;
- logical global-memory traffic is not profiler-measured DRAM traffic;
- register residency does not persist across ordinary decode kernel launches.

## 10. GPU claims that remain proposals

The following were not found as profiler-validated production kernels in the SAMU repository:

- a mode-parallel custom SAMU CUDA/Triton decode kernel;
- per-warp or CTA-shared coherent-control reconstruction;
- a compressed `(C,G,D,q)` training scan kernel;
- time × mode custom tiling;
- persistent multi-step decode;
- dense-write / recurrence fusion;
- any measured claim that SAMU is faster than RG-LRU, Mamba-2, or Mamba-3.

The benchmark harness added by the website rebuild is an engineering experiment. Its results must keep backend, framework, matching protocol, and evidence class visible.
