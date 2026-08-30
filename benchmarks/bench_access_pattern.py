"""Access-pattern bandwidth study — "not all bytes are equal".

Decode latency depends on both bytes moved and achieved bandwidth. This benchmark
measures the latter for the access pattern of a sparse top-k value gather.

This kernel mirrors the P@V value-aggregation step of sparse decode: for each
(batch*kv_head) program, gather K rows of dim D from an [S, D] cache at given token
indices and accumulate. The ONLY thing that varies across runs is the index pattern:
  - contiguous : idx = arange(K)            (coalesced streaming = best case)
  - random     : idx = random K of S        (scattered = sparse top-k value reads)
  - paged      : random page starts, P=16 contiguous tokens each (vLLM-style)
Reads = B*H_k*K*D*2 bytes for every pattern, so GB/s differences are PURE access-pattern.

Run: python -m benchmarks.bench_access_pattern
"""
import argparse, json
import torch
import triton
import triton.language as tl
from benchmarks.timing import benchmark_with_cuda_graphs

PEAK_GBS = 360.0  # RTX 4000 Ada HBM peak (same constant used in kivi_bandwidth_verify.json)


@triton.jit
def _gather_reduce_kernel(v_ptr, idx_ptr, out_ptr,
                          S, K,
                          stride_vh, stride_vs, stride_vd,
                          stride_ih, stride_ik,
                          D: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)                       # flattened batch*kv_head
    d = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        ks = start + tl.arange(0, BLOCK_K)
        mask = ks < K
        tok = tl.load(idx_ptr + pid * stride_ih + ks * stride_ik, mask=mask, other=0)
        v_off = pid * stride_vh + tok[:, None] * stride_vs + d[None, :] * stride_vd
        v = tl.load(v_ptr + v_off, mask=mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr + pid * D + d, acc)


def gather_reduce(v, idx, D, BLOCK_K=16):
    """v:[H,S,D] fp16, idx:[H,K] int32 -> out:[H,D]. H = B*H_k flattened."""
    H, S, _ = v.shape
    K = idx.shape[1]
    out = torch.empty((H, D), device=v.device, dtype=torch.float32)
    _gather_reduce_kernel[(H,)](
        v, idx, out, S, K,
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1),
        D=D, BLOCK_K=BLOCK_K)
    return out


def make_idx(pattern, H, S, K, page, device, gen):
    if pattern == "contiguous":
        base = torch.arange(K, device=device, dtype=torch.int32)
        return base.unsqueeze(0).expand(H, K).contiguous()
    if pattern == "random":
        # independent random K-of-S per head, unsorted (worst-case scatter)
        idx = torch.empty((H, K), device=device, dtype=torch.int32)
        for h in range(H):
            idx[h] = torch.randperm(S, device=device, generator=gen)[:K].to(torch.int32)
        return idx
    if pattern == "sorted_random":
        idx = torch.empty((H, K), device=device, dtype=torch.int32)
        for h in range(H):
            idx[h] = torch.sort(torch.randperm(S, device=device, generator=gen)[:K]).values.to(torch.int32)
        return idx
    if pattern == "paged":
        n_pages = (K + page - 1) // page
        idx = torch.empty((H, n_pages * page), device=device, dtype=torch.int32)
        max_start = S - page
        for h in range(H):
            starts = (torch.randint(0, max_start + 1, (n_pages,), device=device, generator=gen)
                      // page) * page
            offs = (starts[:, None] + torch.arange(page, device=device)[None, :]).reshape(-1)
            idx[h] = offs.to(torch.int32)
        return idx[:, :K].contiguous()
    raise ValueError(pattern)


def measure(v, idx, D, dtype_bytes=2):
    H, K = idx.shape
    lat_ms = benchmark_with_cuda_graphs(gather_reduce, v, idx, D)
    gb = H * K * D * dtype_bytes / 1e9
    gbs = gb / (lat_ms / 1000.0)
    return lat_ms, gbs, 100.0 * gbs / PEAK_GBS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--H-k", type=int, default=8)
    p.add_argument("--S", type=int, default=16384)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default="results/paper/access_pattern.json")
    a = p.parse_args()

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    H = a.B * a.H_k
    v = torch.randn((H, a.S, a.D), device=dev, dtype=torch.float16)

    budgets = [("1%", max(a.page, int(0.01 * a.S))),
               ("2%", int(0.02 * a.S)),
               ("5%", int(0.05 * a.S)),
               ("10%", int(0.10 * a.S)),
               ("25%", int(0.25 * a.S)),
               ("full", a.S)]

    res = {"_meta": {"B": a.B, "H_k": a.H_k, "S": a.S, "D": a.D, "page": a.page,
                     "peak_gbs": PEAK_GBS,
                     "ref_kivi_achieved_gbs": 120.2, "ref_kivi_pct_peak": 33.4,
                     "ref_bf16_contiguous_pct_peak": 49.0},
           "rows": []}

    print(f"shape B={a.B} H_k={a.H_k} S={a.S} D={a.D} page={a.page}  (H={H} gather programs)")
    print(f"peak={PEAK_GBS} GB/s | KIVI ref ~120 GB/s (33% peak) | BF16 contiguous ~49% peak\n")
    print(f"{'budget':>6} {'K':>6} {'pattern':>11} {'lat(ms)':>9} {'GB/s':>8} {'%peak':>7}")
    print("-" * 56)
    for tag, K in budgets:
        patterns = ["contiguous", "random", "sorted_random", "paged"] if K < a.S else ["contiguous", "random", "sorted_random"]
        for pat in patterns:
            idx = make_idx(pat, H, a.S, K, a.page, dev, gen)
            lat, gbs, pct = measure(v, idx, a.D)
            res["rows"].append({"budget": tag, "K": K, "pattern": pat,
                                "lat_ms": lat, "gbs": gbs, "pct_peak": pct})
            print(f"{tag:>6} {K:>6} {pat:>11} {lat:>9.4f} {gbs:>8.1f} {pct:>6.1f}%")
        print()

    # ---- granularity sweep: WHERE does the scatter penalty appear? ----
    # Full gather (random permutation of ALL S tokens) so the entire [H,S,D] cache is
    # read exactly once -> no L2 reuse confound at any D. Vary the per-token row width D
    # (the contiguous burst per scattered access). Penalty should appear only when the
    # burst drops below the DRAM transaction granularity (~32 B = 16 fp16).
    print("\n=== granularity sweep (full gather, no L2 reuse) — random vs contiguous ===")
    print(f"{'D':>5} {'rowB':>6} {'contig%pk':>10} {'rand%pk':>9} {'rand/contig':>12}")
    print("-" * 46)
    gsweep = []
    for D in [8, 16, 32, 64, 128, 256]:
        vD = torch.randn((H, a.S, D), device=dev, dtype=torch.float16)
        idx_c = make_idx("contiguous", H, a.S, a.S, a.page, dev, gen)
        idx_r = make_idx("random", H, a.S, a.S, a.page, dev, gen)
        _, _, pc = measure(vD, idx_c, D)
        _, _, pr = measure(vD, idx_r, D)
        ratio = pr / pc
        gsweep.append({"D": D, "row_bytes": D * 2, "contig_pct": pc, "rand_pct": pr, "ratio": ratio})
        print(f"{D:>5} {D*2:>6} {pc:>9.1f}% {pr:>8.1f}% {ratio:>11.3f}")
        del vD
        torch.cuda.empty_cache()
    res["granularity_sweep"] = gsweep

    json.dump(res, open(a.output_json, "w"), indent=2)
    print(f"\nsaved -> {a.output_json}")


if __name__ == "__main__":
    main()
