# Hugging Face RecurrentGemma comparison

Pinned source: `huggingface/transformers` commit
`42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`, under
`src/transformers/models/recurrent_gemma/`.

The implementation supplies model integration, generation cache handling and
a readable PyTorch recurrence.  At the audited commit, RG-LRU prefill still
uses a Python time loop and decode updates the cache with ordinary PyTorch
operations.  No custom CUDA RG-LRU scan, Triton training kernel or custom
backward is present in that path.

HF is therefore an integration/reference baseline, not a candidate for the
strongest optimized GPU ranking.  Small correctness/integration observations
can be reported, but running its Python loop on H800 would not answer the
public optimized-kernel question and is excluded from performance claims.
