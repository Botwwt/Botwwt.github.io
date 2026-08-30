# Lingua Hawk comparison

Pinned source: `facebookresearch/lingua` commit
`437d680e521873bb5971067148a69587790da853`.

The Hawk recurrence in `apps/fastRNN/component/rnn_common.py` calls
`apps/fastRNN/component/compilable_scan.py`.  That file registers forward and
backward custom operators with `torch.library.custom_op`, then calls
`accelerated_scan.warp.warpscan_forward` and `warpscan_backward`.  The active
path is therefore the proger/accelerated-scan CUDA extension; a Triton import is
present only as a commented alternative.

Lingua materializes affine gates and writes before the scan.  Its wrapper has a
custom backward supplied by accelerated-scan.  The underlying algorithm is a
parallel associative scan, not Fattori's single-program serial recurrence.
Lingua also has a reference fallback for a cache/long-sequence case whose
comment reports illegal memory access in the accelerated path.

The full FastRNN stack contains `torch.compile`, distributed training,
checkpointing and other training-system machinery.  These capabilities are
reported separately because they do not make the recurrence kernel itself
fused.  Its Hawk gate geometry and parameter sharing also differ from the
canonical 16-block RecurrentGemma component used here, so full preparation and
full-model timings are **NOT COMPARABLE**.  The generic affine scan itself is
comparable after coefficients are pre-materialized.

The pinned wrapper needs `torch.library.custom_op`, absent in the project's
original PyTorch 2.1 environment.  Reproduction therefore uses the documented
separate PyTorch 2.4.1 environment and runs our comparison path there as well;
the Lingua source remains unchanged.
