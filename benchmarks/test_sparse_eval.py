"""Tests for the allocation-bounded sparse-attention primitives (Phase A1).

(1) Equivalence: the chunked path == the dense [S,S] reference (the validated B1/B2
    selection semantics) to <1e-3, for exact/approx/quest, across chunk sizes and
    budgets, INCLUDING the early-row regime (frac large => many filler slots) that
    exposed the CUDA topk-tie placeholder bug.
(2) Memory: peak allocation grows sub-quadratically in S (the whole point of the
    chunked rewrite — the dense path was O(S^2) and OOMed past ~8K).

Run:  python -m benchmarks.test_sparse_eval
"""
import torch
import torch.nn.functional as F
from benchmarks.sparse_attention import sparse_attention_chunked, repeat_kv
from benchmarks.eval_ppl import _topk_mask, _quest_mask


def _dense_ref(q, k, v, kq, selector, frac, G, scaling, page):
    """Dense [S,S] reference mirroring eval_ppl's original MODE=='sparse' semantics:
    select top-kb by the selector's score, attend with EXACT scores+values."""
    B, Hq, S, D = q.shape
    kf, vf, kqf = repeat_kv(k, G), repeat_kv(v, G), repeat_kv(kq, G)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=q.device), 1)[None, None]
    exact = (q @ kf.transpose(-1, -2)) * scaling + causal
    kb = max(1, round(frac * S))
    if selector == "quest":
        m = _quest_mask(q, kf, causal, kb, page)
    elif selector == "approx":
        m = _topk_mask((q @ kqf.transpose(-1, -2)) * scaling + causal, kb)
    else:
        m = _topk_mask(exact, kb)
    return (F.softmax(exact + m, dim=-1, dtype=torch.float32).to(v.dtype)) @ vf


def test_chunked_equivalence(device="cuda"):
    torch.manual_seed(0)
    B, Hq, Hk, S, D = 1, 4, 2, 512, 16
    G = Hq // Hk
    scaling = 1.0 / D ** 0.5
    page = 16
    q = torch.randn(B, Hq, S, D, device=device)
    k = torch.randn(B, Hk, S, D, device=device)
    v = torch.randn(B, Hk, S, D, device=device)
    kq = k + 0.05 * torch.randn_like(k)
    n_fail = 0
    # frac=0.5 => many query rows have <kb causal keys (filler-slot / early-row stress)
    for frac in [0.05, 0.20, 0.50]:
        for selector in ["exact", "approx", "quest"]:
            ref = _dense_ref(q, k, v, kq, selector, frac, G, scaling, page)
            for qc, kc in [(128, 128), (S, S)]:
                out = sparse_attention_chunked(q, k, v, kq, selector, frac, G,
                                               scaling, page=page, q_chunk=qc, k_chunk=kc)
                md = (ref - out).abs().max().item()
                ok = md < 1e-3
                n_fail += (not ok)
                print(f"  frac={frac:.2f} {selector:7s} qc={qc:>3} kc={kc:>3}: "
                      f"maxdiff={md:.2e} {'OK' if ok else 'FAIL'}")
    assert n_fail == 0, f"{n_fail} equivalence checks failed"
    print("test_chunked_equivalence: PASS")


def test_memory_subquadratic(device="cuda"):
    """Peak memory must grow ~linearly (not quadratically) in S."""
    Hq, Hk, D = 8, 1, 64
    G = Hq // Hk
    scaling = 1.0 / D ** 0.5
    peaks = {}
    for S in [4096, 8192, 16384]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        q = torch.randn(1, Hq, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, Hk, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, Hk, S, D, device=device, dtype=torch.bfloat16)
        _ = sparse_attention_chunked(q, k, v, None, "exact", 0.1, G, scaling,
                                     q_chunk=256, k_chunk=2048)
        peaks[S] = torch.cuda.max_memory_allocated() / 1e6
        del q, k, v
    r1 = peaks[8192] / peaks[4096]
    r2 = peaks[16384] / peaks[8192]
    print(f"  peak MB: {({s: round(p,1) for s,p in peaks.items()})}")
    print(f"  ratios (2x S): {r1:.2f}, {r2:.2f}  (quadratic would be ~4x)")
    # generous bound: well under quadratic (4x); chunked should be ~2x or less
    assert r1 < 3.0 and r2 < 3.0, "memory grows too fast (not sub-quadratic)"
    print("test_memory_subquadratic: PASS")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device={dev}]")
    test_chunked_equivalence(dev)
    if dev == "cuda":
        test_memory_subquadratic(dev)
    print("\nALL SPARSE-EVAL TESTS PASSED")
