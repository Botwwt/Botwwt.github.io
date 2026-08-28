"""Numerical equivalence tests for the SAMU reference and compressed scan."""

from __future__ import annotations

import torch

from kernels import make_samu_parameters, samu_chunked_compressed, samu_serial, samu_tree_materialized


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(7)
    for dtype, atol, rtol in ((torch.float32, 2e-4, 2e-4), (torch.bfloat16, 8e-2, 8e-2)):
        u = torch.randn(2, 32, 16, device="cuda", dtype=dtype)
        p = make_samu_parameters(16, 8, "cuda", dtype, seed=11)
        serial_direct = samu_serial(u, p, factorized=False)
        serial_fact = samu_serial(u, p, factorized=True)
        tree = samu_tree_materialized(u, p, factorized=True)
        chunk = samu_chunked_compressed(u, p, chunk_size=8)
        torch.testing.assert_close(serial_fact, serial_direct, atol=atol, rtol=rtol)
        torch.testing.assert_close(tree, serial_direct, atol=atol, rtol=rtol)
        torch.testing.assert_close(chunk, serial_direct, atol=atol, rtol=rtol)
        print(dtype, "max errors", {
            "factorized": (serial_fact - serial_direct).abs().max().item(),
            "tree": (tree - serial_direct).abs().max().item(),
            "chunk": (chunk - serial_direct).abs().max().item(),
        })
    print("correctness: PASS")


if __name__ == "__main__":
    main()
