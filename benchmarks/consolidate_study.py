"""Consolidate the matched-conditions selection grids into ONE quality-per-scoring-byte
table + Pareto JSON for the empirical study (axis A2).

Reads results/study_a2_<model>_<ctx>.json (each = {"dense_ppl":.., "grid": {"<sel>_k<frac>": ppl}})
produced by eval_longctx_sparse.py under a FIXED protocol (same local_w, n_windows,
attend cache = exact bf16, eval window). Attaches the per-key SCORING byte cost of each
selector (the bytes a decode step must read per key to rank it) so every method lands on
a common x-axis. Eviction methods (h2o/snapkv) have no per-step scoring read and are
reported on the quality-vs-budget axis only, flagged separately.

Run: python -m benchmarks.consolidate_study
"""
import glob
import json
import math
import os
import re

from benchmarks.sparse_attention import signvq_bytes_per_key, SIGNVQ_SUB, SIGNVQ_MAG_LEVELS

# head_dim per model (D); selection scoring bytes are per-key-head.
MODEL_D = {
    "qwen0.5b": 64, "qwen1.5b": 128, "qwen3b": 128, "llama3b": 128,
    "qwen14b": 128, "qwen32b": 128, "gemma27b": 128,
    "qwen7b": 128,
}

EVICTION = {"h2o", "snapkv"}        # no per-step scoring read (prefill-time importance)
EXACT_REF = "exact"


def scoring_bytes(selector, D):
    """Bytes read PER KEY to score it for top-k selection. None for eviction methods."""
    if selector in EVICTION:
        return None
    if selector == "exact":
        return 2.0 * D                       # fp16 key
    if selector == "fp8":
        return 1.0 * D                       # fp8 key
    if selector == "int4":
        return D / 2.0                       # 4 bits/dim
    if selector == "int2":
        return D / 4.0                       # 2 bits/dim
    if selector == "approx":                 # PQ-LUT (PQCache mech), cb256/d_sub=2
        return D / 2.0                       # M = D/2 uint8 indices/key
    if selector == "signvq":                 # Self-Indexing sign-VQ
        return signvq_bytes_per_key(D)
    m = re.fullmatch(r"sparq(\d+)", selector)
    if m:                                    # reads r fp16 channels/key
        r = min(int(m.group(1)), D)
        return 2.0 * r
    if selector == "quest":                  # page min/max bounds (fp16), amortized/key
        page = 16
        return 4.0 * D / page                # 2 vecs (min,max) * D * 2 B / page tokens
    if selector in ("recent", "random"):
        return 0.0
    return None


def parse_path(p):
    """-> (model, ctx, suffix). suffix is None for the base wikitext panel, or one of
    'sparq<r>' / 'pg19' / 'quest<page>'. The suffix files would otherwise clobber a
    shared grid key (sparq r-sweep, quest page-sweep) or merge corpora (pg19)."""
    b = os.path.basename(p)
    m = re.fullmatch(r"study_a2_([a-z0-9.]+)_(\d+)_(sparq\d+|pg19|quest\d+)\.json", b)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    m = re.fullmatch(r"study_a2_([a-z0-9.]+)_(\d+)\.json", b)
    return (m.group(1), int(m.group(2)), None) if m else (None, None, None)


def load_all():
    """Main-panel rows. The quest page-sweep files are handled by a dedicated emitter
    (they are quest-only and would otherwise duplicate the 'quest' row), so we skip them."""
    rows = []
    paths = glob.glob("results/paper/study_a2_*.json")
    for p in sorted(paths):
        model, ctx, suffix = parse_path(p)
        if model is None:
            continue
        if suffix and suffix.startswith("quest"):
            continue                          # dedicated page-sweep table, not the main grid
        corpus = "pg19" if suffix == "pg19" else "wikitext2"
        sparq_r = int(suffix[5:]) if (suffix and suffix.startswith("sparq")) else None
        D = MODEL_D.get(model)
        with open(p) as f:
            data = json.load(f)
        grid = data.get("grid", {})
        dense = data.get("dense_ppl")
        # collect selectors + fracs present
        sels, fracs = {}, set()
        for k, ppl in grid.items():
            mm = re.fullmatch(r"(.+)_k([0-9.]+)", k)
            if not mm:
                continue
            sel, frac = mm.group(1), float(mm.group(2))
            if sel == "sparq" and sparq_r is not None:
                sel = f"sparq{sparq_r}"      # disambiguate the r-sweep
            sels.setdefault(sel, {})[frac] = ppl
            fracs.add(frac)
        for sel, byfrac in sels.items():
            rows.append({
                "model": model, "ctx": ctx, "corpus": corpus, "D": D, "selector": sel,
                "scoring_bytes": scoring_bytes(sel, D), "is_eviction": sel in EVICTION,
                "dense_ppl": dense, "ppl": byfrac,
            })
    return rows


def fmt(x):
    return "  -  " if x is None else f"{x:6.2f}"


def main():
    rows = load_all()
    if not rows:
        print("No results/study_a2_*.json found yet.")
        return
    # group by (model, ctx, corpus)
    keys = sorted({(r["model"], r["ctx"], r["corpus"]) for r in rows})
    out = {"protocol": "matched: exact bf16 attend cache, local_w=32, loss on last eval-window",
           "signvq_config": {"sub": SIGNVQ_SUB, "mag_levels": SIGNVQ_MAG_LEVELS},
           "byte_axis": "scoring bytes/key-head (bytes read per key to rank it)",
           "panels": []}
    for model, ctx, corpus in keys:
        sub = [r for r in rows if r["model"] == model and r["ctx"] == ctx and r["corpus"] == corpus]
        D = sub[0]["D"]
        dense = sub[0]["dense_ppl"]
        fracs = sorted({f for r in sub for f in r["ppl"]})
        tag = f"  [{corpus}]" if corpus != "wikitext2" else ""
        print(f"\n=== {model}  ctx={ctx}  D={D}  dense={dense:.2f}{tag} ===")
        print(f"{'selector':>10} {'B/key':>7} | " + " ".join(f"k={int(f*100)}%".rjust(7) for f in fracs))
        print("-" * (22 + 9 * len(fracs)))
        # order: exact, approx, signvq, sparq*, int4, int2, fp8, quest, eviction
        order = ["exact", "approx", "signvq", "sparq8", "sparq16", "sparq32",
                 "int4", "int2", "fp8", "quest", "h2o", "snapkv"]
        sub.sort(key=lambda r: (order.index(r["selector"]) if r["selector"] in order else 99))
        for r in sub:
            b = r["scoring_bytes"]
            bs = "evict" if r["is_eviction"] else (f"{b:.0f}" if b is not None else "-")
            print(f"{r['selector']:>10} {bs:>7} | " +
                  " ".join(fmt(r["ppl"].get(f)) for f in fracs))
        out["panels"].append({"model": model, "ctx": ctx, "corpus": corpus, "D": D,
                              "dense_ppl": dense, "fracs": fracs, "rows": sub})
    os.makedirs("results/paper", exist_ok=True)
    with open("results/paper/study_a2_pareto.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/paper/study_a2_pareto.json")


if __name__ == "__main__":
    main()
