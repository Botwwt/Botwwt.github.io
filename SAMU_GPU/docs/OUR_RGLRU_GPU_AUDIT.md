# Our RG-LRU GPU audit

## Legacy differentiable training path

The pre-existing inference/prefill file `benchmark/triton_rglru.py` already
fused gate activation, transition reconstruction, write and recurrence after
the two gate projections.  The missing optimization was the complete
differentiable training path in `RGLRUMixer._terms`.

That old training path constructed logical full-shape tensors for two gate
activations, `log_a`, transition `a`, square-root multiplier and write, and
used `torch.cat` to install the first-token reset.  It then called a custom
serial or chunk scan.  The chunk scan additionally allocated FP32 transition
and write summaries plus chunk input states.  `torch.compile` could potentially
fuse some pointwise producers, but the custom scan boundary consumed explicit
`a` and `write` tensors, so their logical interface remained.

## Strengthened fused path

`benchmark/triton_fused_rglru_training.py` accepts recurrent input, two gate
logits, `a_param`, optional `segment_pos` and optional initial state.  It
contains:

- a fully fused state-stationary serial forward/backward;
- exact two-level chunk forward/backward for K=8/16/32/64;
- arbitrary resets, including multiple resets within a chunk;
- nonzero initial state and its gradient;
- partial final chunks;
- clipped square-root backward;
- BF16 transition/write boundaries and FP32 recurrent accumulation;
- backward recomputation of pointwise gate/transition/write terms;
- partial `a_param` gradients followed by a reduction.

The fused paths retain the projection outputs because the two official
block-diagonal projections run before the custom operator.  They do not create
logical global `[B,L,D]` activation, transition, multiplier or write
allocations.  Outputs and saved logits/states remain state-sized, and chunk
paths allocate `[B,ceil(L/K),D]` FP32 summaries.  “No logical global
transition” is not a claim of zero HBM traffic or guaranteed register
residency.

The final strengthened baseline also has a linear-work real-affine prefix:
group-local serial composition, an outer group prefix, and local correction.
It preserves the noncommutative affine composition order and is used in both
forward and reverse mode.  The selected H800 backend is grouped-prefix32 with
K=16 and a shape-calibrated mode block.  SAMU therefore is not compared with
the old serial or Hillis--Steele RG-LRU path where this grouped path is faster.

## Exact real chunk algebra

Each token is an affine map `h -> a_t h + b_t`.  For a chunk, the exact summary
is `h_out = P h_in + Q`, updated left to right by

```text
P <- a_t P
Q <- a_t Q + b_t.
```

A reset is encoded by `a_t=0` and the canonical reset write, so it naturally
erases the prefix before it without inexact post-processing.  The last partial
chunk applies identity/no-write for invalid lanes.  Reverse mode uses the same
affine closure on the adjoint recurrence and replays each chunk to recompute
pointwise derivatives.

## Attribution

Fusing RG-LRU pointwise preparation into a serial Triton scan, keeping a
running carry and recomputing activations in backward are Fattori prior art.
The reset-aware exact chunk forward/backward, partial chunks, initial-state
gradient, grouped affine hierarchy, and H800 dispatch in this repository are
our implementation work.  Fattori's repository is described as an open-source
Hawk/RG-LRU implementation, not as an official implementation.

## Validation

`benchmark/test_fused_rglru_training.py` compares all-token output, final state
and gradients for input, both gate logits, `a_param` and initial state against
the canonical eager oracle.  Grouped-prefix correctness, including reset,
nonzero initial state and partial chunks, is recorded in
`results/gpu_optimization/rglru_grouped_prefix_correctness.json`.  Candidate
timings and rejected compile combinations remain in the optimization result
tree.
