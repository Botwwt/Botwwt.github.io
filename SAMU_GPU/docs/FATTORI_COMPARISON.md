# Fattori comparison

Pinned source: `fattorib/hawk-pytorch` commit
`e43239336cf93b6dd79486447150429e472a4b90`, primarily
`hawk/scan_fused.py` and its caller in `hawk/hawk.py`.

The public training call path computes two block-diagonal gate projections in
PyTorch and passes the recurrent input, both raw gate logits and `a_param` to a
custom-autograd Triton operator.  The operator fuses:

- both sigmoid activations;
- `softplus(a_param)`, dynamic decay and stationary square-root factor;
- the input-gated write;
- a sequential, state-stationary recurrence;
- the reverse recurrence and pointwise derivatives in its custom backward.

It does not fuse the two gate projections.  It saves the recurrent input, two
gate-logit tensors, `a_param` and all output states.  Backward recomputes the
sigmoids, transition, multiplier and their derivatives, accumulates a
per-batch `a_param` gradient and reduces it in PyTorch.

The kernel uses grid `[batch, width/64]`, one warp and a 64-channel tile.  The
running carry is intended to stay on chip; this wording does not assert that
the compiler never spills.  Output and large input/logit tensors are still
global tensors.

Semantic limits of the unmodified public source:

- no `segment_pos`, arbitrary reset or nonzero initial-state argument;
- first-token write retains `sqrt(1-a²)`, unlike canonical reset-token normalization;
- width must be a multiple of 64;
- the backward indexes prior states in a way that assumes sequence length at least two;
- output and saved states are BF16 while recurrent arithmetic is FP32.

Consequently Fattori is **NOT COMPARABLE** in the primary canonical reset
ranking.  It is measured in a secondary zero-initial-state, no-reset protocol,
where its recurrence is semantically aligned.  The reproduction script imports
`fused_linear_scan` and `BlockDiagonalLinear` directly from the pinned checkout;
it contains no replacement baseline implementation.

Our serial fusion/recomputation strategy explicitly credits this prior art.
Our additions are canonical reset/initial-state support and exact parallel
chunk forward/backward with partial chunks.
