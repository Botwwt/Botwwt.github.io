# Hippogriff comparison

Pinned source: `proger/hippogriff` commit
`72f6c448b6ce6e4bc742de07390a5914c42163b3`.

`hippogriff.py` imports `scan` directly from `accelerated_scan.warp`.  Before
the call it constructs full token-dependent `alpha` and write tensors,
transposes them to contiguous `[B,D,L]`, runs the extension, and transposes the
states back.  Thus the recurrent speed comes mainly from accelerated-scan; the
application also uses FlashAttention for its attention blocks.

Pinned dependency source: `proger/accelerated-scan` commit
`866fa4cd4eaf01338ee6d77a4f99fcd4b2f656d1`.  Its warp CUDA implementation is
a hierarchical associative prefix scan using warp shuffles and shared memory,
with original custom forward and backward bindings.  This differs from
Fattori's fused state-stationary Triton loop:

| Aspect | Fattori | Hippogriff/accelerated-scan |
|---|---|---|
| preparation | activation/write fused in scan | full affine coefficients materialized |
| time algorithm | sequential carry per channel tile | parallel associative prefix |
| implementation | Triton | CUDA C++ extension |
| backward | recompute RG-LRU pointwise terms | generic affine-scan backward |

Hippogriff's dense gate layer, added epsilon in its normalization and lack of
canonical reset/initial-state interface make its full RG-LRU **NOT
COMPARABLE**.  Its generic scan is reproduced and ranked only in the scan-only
protocol.  The Hippogriff and Lingua rows share the same accelerated-scan
kernel and must not be interpreted as two independent algorithms.
