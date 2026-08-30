"""Independent-capture uncertainty for the sparse score+top-k+gather decode paths."""
import argparse
import json
import math
import os
import torch

from benchmarks.timing import benchmark_captures
from src.attention_kernels import quantize_and_pack_score_keys, sparse_decode_scalar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="results/paper/latency_robustness.json")
    p.add_argument("--captures", type=int, default=15)
    p.add_argument("--runs", type=int, default=100)
    a = p.parse_args()
    torch.manual_seed(0)
    B, H_q, H_k, D = 32, 64, 8, 128
    rows = []

    def checkpoint():
        """Persist raw captures after every cell so an OOM cannot erase evidence."""
        out = {"schema": "latency_robustness_v1", "gpu": torch.cuda.get_device_name(),
               "seed": 0, "rows": rows}
        os.makedirs(os.path.dirname(a.output_json) or ".", exist_ok=True)
        with open(a.output_json, "w") as f:
            json.dump(out, f, indent=2)
    # The full B=32 decode path fits reproducibly through 16K on this 20GB card.
    # The 32K resource boundary is recorded separately in sparse_crossover.json.
    for S in (8192, 16384):
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.bfloat16)
        score = {"fp8": (k.to(torch.float8_e4m3fn), None, None, 8)}
        for bits, name in ((4, "int4"), (2, "int2")):
            score[name] = (*quantize_and_pack_score_keys(k, bits, 32), bits)
        for frac in (0.01, 0.05, 0.10):
            row = {"S": S, "D": D, "B": B, "H_q": H_q, "H_k": H_k, "frac": frac,
                   "captures": a.captures, "runs_per_capture": a.runs, "latency_ms": {}}
            for name, (keys, scale, zero, bits) in score.items():
                row["latency_ms"][name] = benchmark_captures(
                    sparse_decode_scalar, q, keys, scale, zero, k, v, frac, 1.0 / math.sqrt(D),
                    score_bits=bits, group_size=32, captures=a.captures, num_runs=a.runs)
            rows.append(row)
            checkpoint()
            print(f"S={S} k={frac}: fp8={sum(row['latency_ms']['fp8'])/a.captures:.3f} ms")
        del q, k, v, score
        torch.cuda.empty_cache()
    checkpoint()


if __name__ == "__main__":
    main()
