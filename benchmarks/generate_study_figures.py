"""Figures for the empirical study. Fail-loud: every figure reads tracked evidence
produced by the consolidators; missing data raises rather than fabricating.

  fig_study_pareto.pdf       A2: PPL vs SCORING bytes/key (quality-per-scoring-byte),
                             one panel per budget; all selectors as points + dense line.
  fig_study_budget.pdf       A2: PPL vs selection budget, one line per method.
  fig_study_profiling.pdf    P1-P3: (a) achieved %peak by access pattern, (b) two-pass
                             latency decomposition + packed-INT latency overlay.

Run: python -m benchmarks.consolidate_study && python -m benchmarks.consolidate_profiling \
     && python -m benchmarks.generate_study_figures
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# consistent styling per selector family
STYLE = {
    "exact":  ("#000000", "o", "exact (fp32 score)"),
    "fp8":    ("#7f7f7f", "s", "fp8 scalar"),
    "approx": ("#1f77b4", "D", "PQ approximation"),
    "int4":   ("#2ca02c", "^", "int4 scalar"),
    "signvq": ("#d62728", "v", "sign-VQ (Self-Index)"),
    "int2":   ("#9467bd", "<", "int2 scalar"),
    "sparq8": ("#ff7f0e", "P", "SparQ r=8"),
    "sparq16":("#ff7f0e", "X", "SparQ r=16"),
    "sparq32":("#ff7f0e", "*", "SparQ r=32"),
    "quest":  ("#8c564b", "h", "Quest (page)"),
    "h2o":    ("#e377c2", "x", "H2O (evict)"),
    "snapkv": ("#bcbd22", "+", "SnapKV (evict)"),
}


def load(name):
    p = f"results/paper/{name}.json"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p} -- run the consolidators first")
    with open(p) as f:
        return json.load(f)


def _panel_lookup(par, model, ctx, corpus="wikitext2"):
    for p in par["panels"]:
        if p["model"] == model and p["ctx"] == ctx and p.get("corpus", "wikitext2") == corpus:
            return p
    return None


def fig_pareto(par, model="qwen0.5b", ctx=16384):
    panel = _panel_lookup(par, model, ctx)
    if panel is None:
        print(f"[pareto] no panel for {model}@{ctx}; skipping")
        return
    fracs = panel["fracs"]
    budgets = [f for f in (0.02, 0.05) if f in fracs] or fracs[:2]
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 4.2), squeeze=False)
    for ax, frac in zip(axes[0], budgets):
        for r in panel["rows"]:
            sel = r["selector"]
            b = r["scoring_bytes"]
            ppl = r["ppl"].get(str(frac), r["ppl"].get(frac))
            if ppl is None:
                continue
            c, mk, lab = STYLE.get(sel, ("#444", ".", sel))
            if r["is_eviction"] or b is None:
                ax.axhline(ppl, ls=":", lw=1.2, color=c, alpha=0.8, label=f"{lab}")
            else:
                ax.scatter(b, ppl, c=c, marker=mk, s=70, label=lab, zorder=3, edgecolors="k", linewidths=0.4)
        ax.axhline(panel["dense_ppl"], color="gray", ls="--", lw=1, label="dense (full attn)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("scoring bytes / key-head")
        ax.set_ylabel("PPL")
        ax.set_title(f"{model}  S={ctx//1024}K  budget k={int(frac*100)}%")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    out = f"{FIG}/fig_study_pareto.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def fig_budget(par, model="qwen0.5b", ctx=16384):
    panel = _panel_lookup(par, model, ctx)
    if panel is None:
        print(f"[budget] no panel for {model}@{ctx}; skipping")
        return
    fracs = sorted(panel["fracs"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in panel["rows"]:
        sel = r["selector"]
        c, mk, lab = STYLE.get(sel, ("#444", ".", sel))
        ys = [r["ppl"].get(str(f), r["ppl"].get(f)) for f in fracs]
        xs = [f * 100 for f, y in zip(fracs, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if not ys:
            continue
        ax.plot(xs, ys, marker=mk, color=c, label=lab, lw=1.3, ms=5)
    ax.axhline(panel["dense_ppl"], color="gray", ls="--", lw=1, label="dense")
    ax.set_xlabel("selection budget (% of keys)")
    ax.set_ylabel("PPL")
    ax.set_title(f"Selection quality vs budget — {model} S={ctx//1024}K")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = f"{FIG}/fig_study_budget.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def fig_profiling(prof):
    bw = prof["P1_P2_bandwidth"]
    dec = prof["P3_decomposition"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) achieved % peak
    labels = ["gather\n(random)", "gather\n(contig)", "gather\n(paged)"]
    keys = ["gather_random", "gather_contiguous", "gather_paged"]
    vals = [bw[k] for k in keys]
    cols = ["#1f77b4", "#1f77b4", "#1f77b4"]
    a1.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        a1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    a1.set_ylabel("% of HBM peak bandwidth")
    a1.set_title("(a) Gather access patterns achieve similar bandwidth")
    a1.set_ylim(0, 100); a1.grid(axis="y", alpha=0.3)
    # (b) latency decomposition
    fr = sorted(dec.keys(), key=lambda k: float(k[1:]))
    x = [float(k[1:]) * 100 for k in fr]
    score = [dec[k]["score_pass_ms"] for k in fr]
    topk = [dec[k]["topk_ms"] for k in fr]
    gather = [dec[k]["gather_ms"] for k in fr]
    int4 = [dec[k]["sparse_int4"] for k in fr]
    w = [xx * 0.18 for xx in x]
    a2.bar(x, score, width=w, label="score pass", color="#1f77b4")
    a2.bar(x, topk, width=w, bottom=score, label="top-k (library operation)", color="#ff7f0e")
    a2.bar(x, gather, width=w, bottom=[s + t for s, t in zip(score, topk)], label="gather", color="#2ca02c")
    a2.plot(x, int4, "rv--", label="packed-int4", ms=6)
    a2.set_xscale("log")
    a2.set_xlabel("selection budget (% of keys)")
    a2.set_ylabel("decode latency (ms) @ S=16K")
    a2.set_title("(b) two-pass decomposition: score pass dominates")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    out = f"{FIG}/fig_study_profiling.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def _summary(values):
    values = np.asarray(values, dtype=float)
    return np.median(values), np.percentile(values, 2.5), np.percentile(values, 97.5)


def fig_crossover(xover):
    contexts, fracs = [4096, 8192, 16384, 32768], [0.01, 0.02, 0.05, 0.10, 0.25]
    ratios = np.full((len(contexts), len(fracs)), np.nan)
    winners = np.empty_like(ratios, dtype=object)
    for i, S in enumerate(contexts):
        for j, frac in enumerate(fracs):
            row = next(x for x in xover if x["S"] == S and x["frac"] == frac)
            fp8, kivi = row["lat_ms"]["sparse_fp8"], row["lat_ms"]["kivi2"]
            ratios[i, j] = fp8 / kivi
            winners[i, j] = "fp8 sparse" if fp8 < kivi else "KIVI-2"
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(np.log2(ratios), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    for i in range(len(contexts)):
        for j in range(len(fracs)):
            ax.text(j, i, f"{winners[i,j]}\n{ratios[i,j]:.2f}×", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(fracs)), [f"{int(x*100)}%" for x in fracs])
    ax.set_yticks(range(len(contexts)), [f"{x//1024}K" for x in contexts])
    ax.set_xlabel("retained-key budget"); ax.set_ylabel("context length")
    ax.set_title("Measured fp8-sparse / KIVI-2 latency ratio (RTX 4000 Ada; <1 favors sparse)")
    fig.colorbar(im, ax=ax, label=r"$\log_2$(fp8 sparse / KIVI-2)")
    fig.tight_layout(); out = f"{FIG}/fig_crossover_regimes.pdf"; fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def fig_capture_robustness(data):
    rows = [r for r in data["rows"] if r["frac"] == 0.05]
    rows.sort(key=lambda r: r["S"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for method, color, marker in (("fp8", "#777777", "s"), ("int4", "#2ca02c", "^"), ("int2", "#9467bd", "<")):
        med, lo, hi = zip(*[_summary(r["latency_ms"][method]) for r in rows])
        x = [r["S"] // 1024 for r in rows]
        ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], label=method,
                    color=color, marker=marker, capsize=3)
    ax.set_xlabel("context length"); ax.set_ylabel("full sparse-decode latency (ms)")
    ax.set_title("Independent CUDA-graph captures (median and 95% percentile interval, k=5%)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    out = f"{FIG}/fig_latency_robustness.pdf"; fig.savefig(out); plt.close(fig); print(f"saved {out}")


def fig_shape_sweep(data):
    rows = data["rows"]
    Ss, Ds = [8192, 16384], [64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, method in zip(axes, ("int4", "int2")):
        ratio = np.zeros((len(Ds), len(Ss)))
        for i, D in enumerate(Ds):
            for j, S in enumerate(Ss):
                r = next(x for x in rows if x["S"] == S and x["D"] == D)
                ratio[i, j] = _summary(r["latency_ms"][method])[0] / _summary(r["latency_ms"]["fp8"])[0]
        im = ax.imshow(ratio, cmap="Reds", vmin=1, vmax=max(2, ratio.max()), aspect="auto")
        for i in range(len(Ds)):
            for j in range(len(Ss)):
                ax.text(j, i, f"{ratio[i,j]:.2f}×", ha="center", va="center")
        ax.set_xticks(range(len(Ss)), [f"{s//1024}K" for s in Ss]); ax.set_yticks(range(len(Ds)), Ds)
        ax.set_xlabel("context length"); ax.set_title(f"{method} / fp8 score latency")
        fig.colorbar(im, ax=ax, label="median ratio")
    axes[0].set_ylabel("head dimension D")
    fig.tight_layout(); out = f"{FIG}/fig_score_shape_sweep.pdf"; fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def main():
    par = load("study_a2_pareto")
    prof = load("study_profiling")
    fig_pareto(par)
    fig_budget(par)
    fig_profiling(prof)
    fig_crossover(load("sparse_crossover"))
    fig_capture_robustness(load("latency_robustness"))
    fig_shape_sweep(load("score_shape_sweep"))


if __name__ == "__main__":
    main()
