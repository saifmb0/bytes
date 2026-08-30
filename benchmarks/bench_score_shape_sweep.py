"""Measure score-kernel shape sensitivity on the paper's single RTX 4000 Ada target."""
import argparse
import json
import math
import os
import torch

from benchmarks.timing import benchmark_captures
from src.attention_kernels import (quantize_and_pack_score_keys, sparse_score_packed,
                                   sparse_score_scalar)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="results/paper/score_shape_sweep.json")
    p.add_argument("--captures", type=int, default=15)
    p.add_argument("--runs", type=int, default=100)
    a = p.parse_args()
    torch.manual_seed(0)
    B, H_q, H_k, frac = 32, 64, 8, 0.05
    # The full B=32, D=256 score path does not fit at 32K on the paper GPU.
    # Keep a rectangular, fully reproducible 8K/16K matrix rather than mixing a
    # partial 32K row into a heatmap.
    contexts, dimensions = (8192, 16384), (64, 128, 256)
    rows = []
    if os.path.exists(a.output_json):
        prior = json.load(open(a.output_json))
        rows = [r for r in prior.get("rows", [])
                if r.get("S") in contexts and r.get("D") in dimensions]

    def checkpoint():
        """Persist raw captures after every shape cell for failure diagnosis."""
        out = {"schema": "score_shape_sweep_v1", "gpu": torch.cuda.get_device_name(),
               "seed": 0, "rows": rows}
        os.makedirs(os.path.dirname(a.output_json) or ".", exist_ok=True)
        with open(a.output_json, "w") as f:
            json.dump(out, f, indent=2)
    for S in contexts:
        for D in dimensions:
            if any(r["S"] == S and r["D"] == D for r in rows):
                continue
            q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.bfloat16)
            scale = 1.0 / math.sqrt(D)
            fp8 = k.to(torch.float8_e4m3fn)
            row = {"S": S, "D": D, "B": B, "H_q": H_q, "H_k": H_k,
                   "frac": frac, "captures": a.captures, "runs_per_capture": a.runs,
                   "latency_ms": {}}
            row["latency_ms"]["fp8"] = benchmark_captures(
                sparse_score_scalar, q, fp8, scale, score_bits=8, captures=a.captures,
                num_runs=a.runs)
            for bits, name in ((4, "int4"), (2, "int2")):
                packed, qscale, zero = quantize_and_pack_score_keys(k, bits, 32)
                row["latency_ms"][name] = benchmark_captures(
                    sparse_score_packed, q, packed, qscale, zero, scale, score_bits=bits,
                    group_size=32, captures=a.captures, num_runs=a.runs)
                del packed, qscale, zero
            rows.append(row)
            checkpoint()
            print(f"S={S} D={D}: fp8={sum(row['latency_ms']['fp8'])/a.captures:.3f} ms")
            del q, k, fp8
            torch.cuda.empty_cache()
    checkpoint()


if __name__ == "__main__":
    main()
