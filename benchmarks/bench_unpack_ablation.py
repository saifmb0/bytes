"""Counter-free software-path ablation for packed score reconstruction.

Each variant reads the same packed key layout.  It isolates the source-level work
introduced by packing without claiming a particular GPU issue/stall mechanism.
"""
import argparse
import json
import math
import os

import torch
import triton
import triton.language as tl

from benchmarks.timing import benchmark_captures
from src.attention_kernels import quantize_and_pack_score_keys, sparse_score_packed, sparse_score_scalar


@triton.jit
def _packed_stage_kernel(Q, KP, Scale, Zero, Out,
                         sqb, sqh, sqd, skb, skh, sks, skd,
                         ssb, ssh, sss, ssg, szb, szh, szs, szg,
                         sob, soh, sos, Hq: tl.constexpr, Hk: tl.constexpr,
                         G: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                         bits: tl.constexpr, stage: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    b, hq = pid // Hq, pid % Hq
    hk = hq // G
    cols = tl.arange(0, D)
    epb = 8 // bits
    byte_cols = cols // epb
    shifts = (cols % epb) * bits
    groups = cols // 32
    q = tl.load(Q + b * sqb + hq * sqh + cols * sqd).to(tl.float32)
    for start in range(0, S, BLOCK_N):
        rows = start + tl.arange(0, BLOCK_N)
        mask = rows < S
        p = tl.load(KP + b*skb + hk*skh + rows[:, None]*sks + byte_cols[None, :]*skd,
                    mask=mask[:, None], other=0)
        if stage == 0:
            vals = p.to(tl.float32)
        else:
            vals = ((p >> shifts[None, :]) & ((1 << bits) - 1)).to(tl.float32)
            if stage >= 2:
                scale = tl.load(Scale + b*ssb + hk*ssh + rows[:, None]*sss + groups[None, :]*ssg,
                                mask=mask[:, None], other=1.0).to(tl.float32)
                zero = tl.load(Zero + b*szb + hk*szh + rows[:, None]*szs + groups[None, :]*szg,
                               mask=mask[:, None], other=0.0).to(tl.float32)
                vals = vals * scale + zero
            if stage == 3:
                vals *= q[None, :]
        out = tl.sum(vals, axis=1)
        tl.store(Out + b*sob + hq*soh + rows*sos, out, mask=mask)


def stage_score(q, packed, scale, zero, bits, stage):
    B, Hq, _, D = q.shape; _, Hk, S, _ = packed.shape
    out = torch.empty((B, Hq, S), device=q.device, dtype=torch.float32)
    _packed_stage_kernel[(B * Hq,)](q.squeeze(2), packed, scale, zero, out,
        q.squeeze(2).stride(0), q.squeeze(2).stride(1), q.squeeze(2).stride(2),
        packed.stride(0), packed.stride(1), packed.stride(2), packed.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        zero.stride(0), zero.stride(1), zero.stride(2), zero.stride(3),
        out.stride(0), out.stride(1), out.stride(2), Hq, Hk, Hq // Hk, S, D,
        bits=bits, stage=stage, BLOCK_N=128, num_warps=4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="results/paper/unpack_ablation.json")
    p.add_argument("--captures", type=int, default=15); p.add_argument("--runs", type=int, default=100)
    a = p.parse_args(); torch.manual_seed(0)
    rows = []
    if os.path.exists(a.output_json):
        prior = json.load(open(a.output_json))
        rows = prior.get("rows", [])

    def checkpoint():
        os.makedirs(os.path.dirname(a.output_json) or ".", exist_ok=True)
        with open(a.output_json, "w") as f:
            json.dump({"schema":"unpack_ablation_v1", "gpu":torch.cuda.get_device_name(),
                       "seed":0, "rows":rows}, f, indent=2)
    for S in (8192, 16384):
        for D in (64, 128, 256):
            if any(r.get("S") == S and r.get("D") == D for r in rows):
                continue
            B, Hq, Hk = 32, 64, 8
            q = torch.randn(B,Hq,1,D,device="cuda",dtype=torch.bfloat16)
            k = torch.randn(B,Hk,S,D,device="cuda",dtype=torch.bfloat16)
            fp8 = k.to(torch.float8_e4m3fn); scale0 = 1/math.sqrt(D)
            row = {"S":S,"D":D,"B":B,"H_q":Hq,"H_k":Hk,"captures":a.captures,
                   "runs_per_capture":a.runs,"latency_ms":{}}
            row["latency_ms"]["fp8_dot"] = benchmark_captures(sparse_score_scalar,q,fp8,scale0,score_bits=8,captures=a.captures,num_runs=a.runs)
            # Keep only one representation live.  At the largest publication
            # shape, retaining fp8 plus both packed variants makes the *setup*
            # OOM even though each measured kernel fits.
            del fp8
            for bits, name in ((4,"int4"),(2,"int2")):
                packed, scale, zero = quantize_and_pack_score_keys(k,bits,32)
                for stage, label in enumerate(("packed_load_reduce","unpack_reduce","dequant_reduce")):
                    row["latency_ms"][f"{name}_{label}"] = benchmark_captures(stage_score,q,packed,scale,zero,bits,stage,captures=a.captures,num_runs=a.runs)
                row["latency_ms"][f"{name}_full_dot"] = benchmark_captures(sparse_score_packed,q,packed,scale,zero,scale0,score_bits=bits,group_size=32,captures=a.captures,num_runs=a.runs)
                del packed, scale, zero
                torch.cuda.empty_cache()
            rows.append(row); checkpoint(); print(S,D)
            del q,k; torch.cuda.empty_cache()
    checkpoint()

if __name__ == "__main__": main()
