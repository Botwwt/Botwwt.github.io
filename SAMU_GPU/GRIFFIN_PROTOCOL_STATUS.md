# Griffin training and inference protocol status

This file separates completed H800 evidence from experiments that the current
hardware or kernels cannot yet support. “Mapped” means the independent variable,
shape and timing boundary follow the paper; it does not mean the paper's TPU-v3
number is reproduced.

| Griffin item | Current SAMU/RG-LRU status | Evidence boundary |
|---|---|---|
| Section 4.1 model parallelism | Not measured | One H800 cannot measure tensor-parallel all-reduce or the paper's sharding communication. |
| Section 4.1 ZeRO optimizer sharding | Not measured | No multi-device optimizer run is present. |
| Section 4.2 linear recurrence scan | Measured | Post-projection forward scan/gate stage on H800. |
| Appendix D / Figure 8(a), `B=8`, state 1024, length sweep | Mapped and measured at 2K–16K | H800 instead of the paper's two-chip TPU-v3 pod. |
| Linear versus associative scan precision | Measured as a transparent reference | The associative PyTorch implementation is not claimed to be an optimized GPU baseline. |
| Figure 8(b)-style full-model forward effect | Mapped at one ~1B random-weight proxy | Exact shared shell, `B=8`, lengths 2K–16K, five samples per AB/BA order; this isolates the forward effect only. |
| Figure 8(b), complete 400M/1B/7B training runtime | Not measured | Custom backward and full training step do not exist yet; the forward proxy is not substituted for this result. |
| Section 5 latency, `B=16`, prompt 0/4096 | Measured in the 1B random-weight proxy | Prompt prefill is excluded; decode checkpoints are 128–4096 tokens. |
| Section 5 throughput, output 512–4096 | Measured in the 1B random-weight proxy | Candidate batches come from a declared 1–256 probe, then complete full trajectories. |
| Equation (5) cache/bandwidth model | Applied analytically | Both models use equal fixed-size recurrent state; no hardware bandwidth value is inferred. |
| Trained model quality | Not measured | SAMU-16 direct changes the architecture and must be retrained before perplexity or downstream claims. |
| DRAM, L2, SFU and achieved occupancy | Not measured | The host blocks Nsight Compute counters with `RmProfilingAdminOnly=1`. |

The current strongest recurrence-level candidate is grouped SAMU-16 with direct
shared control. It halves the grouped write/gate projection MAC relative to the
official RG-LRU-16 temporal layer while preserving equal recurrent-state bytes.
The exact shared-shell proxy retains a `1.026x–1.047x` full-forward advantage,
but official RG-LRU remains `1.023x–1.038x` faster in continuous decode and
`1.016x–1.020x` higher in completed-trajectory throughput. This is therefore a
hypothesis about a better quality–efficiency point until retraining is complete,
not an established end-to-end win.
