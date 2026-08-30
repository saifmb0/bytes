"""One-shot kernels for Nsight Compute attribution.

Run under ncu after selecting one component.  The workload deliberately uses the
paper tensor shape so the same Triton specializations are profiled; it launches each
component once after a warm-up to keep counter reports unambiguous.
"""
import argparse
import math

import torch

from src.attention_kernels import (quantize_and_pack_score_keys, sparse_gather_attn,
                                   sparse_score_packed, sparse_score_scalar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=("fp8-score", "int4-score", "int2-score", "gather", "topk"))
    parser.add_argument("--S", type=int, default=16384)
    parser.add_argument("--frac", type=float, default=0.05)
    args = parser.parse_args()
    torch.manual_seed(0)
    B, Hq, Hk, D = 32, 64, 8, 128
    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hk, args.S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scaling = 1.0 / math.sqrt(D)
    fp8 = k.to(torch.float8_e4m3fn)
    packed4, scale4, zero4 = quantize_and_pack_score_keys(k, 4, 32)
    packed2, scale2, zero2 = quantize_and_pack_score_keys(k, 2, 32)
    kb = int(round(args.S * args.frac))
    scores = sparse_score_scalar(q, fp8, scaling, score_bits=8)
    idx = scores.topk(kb, dim=-1).indices
    table = {
        "fp8-score": lambda: sparse_score_scalar(q, fp8, scaling, score_bits=8),
        "int4-score": lambda: sparse_score_packed(q, packed4, scale4, zero4, scaling, score_bits=4),
        "int2-score": lambda: sparse_score_packed(q, packed2, scale2, zero2, scaling, score_bits=2),
        "gather": lambda: sparse_gather_attn(q, k, v, idx, scaling),
        "topk": lambda: scores.topk(kb, dim=-1).indices,
    }
    table[args.component]()
    torch.cuda.synchronize()
    # The range excludes allocation, quantization, and the warmup, so Nsight
    # Systems can report the selected operation without setup contamination.
    torch.cuda.nvtx.range_push(f"paper_target_{args.component}_S{args.S}_k{kb}")
    table[args.component]()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
