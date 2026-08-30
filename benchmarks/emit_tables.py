"""Emit LaTeX tables for the study directly from the consolidated results JSON, so
every number in the paper traces to a tracked results/paper/*.json (integrity rule). Writes into
tables/ which the paper \\input's. Fail-loud on missing data.

Run: python -m benchmarks.emit_tables
"""
import json
import os

os.makedirs("tables", exist_ok=True)

SEL_LABEL = {
    "exact": "exact (fp32 score)", "fp8": "fp8 scalar", "approx": "PQ-LUT (PQCache)",
    "int4": "int4 scalar", "signvq": "sign-VQ (Self-Index.)", "int2": "int2 scalar",
    "sparq8": "SparQ $r{=}8$", "sparq16": "SparQ $r{=}16$", "sparq32": "SparQ $r{=}32$",
    "quest": "Quest (page)", "h2o": "H2O (evict)", "snapkv": "SnapKV (evict)",
}
ORDER = ["exact", "fp8", "approx", "int4", "signvq", "int2",
         "sparq8", "sparq16", "sparq32", "quest", "h2o", "snapkv"]
MODEL_LABEL = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B",
               "qwen3b": "Qwen2.5-3B", "llama3b": "Llama-3.2-3B",
               "qwen7b": "Qwen2.5-7B",
               "qwen14b": "Qwen2.5-14B (NF4)", "qwen32b": "Qwen2.5-32B (NF4)",
               "gemma27b": "Gemma-2-27B (NF4)"}


def load(name):
    p = f"results/paper/{name}.json"
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p} -- run the consolidators first")
    with open(p) as f:
        return json.load(f)


def _ppl(row, frac):
    return row["ppl"].get(str(frac), row["ppl"].get(frac))


def emit_a2(par):
    for panel in par["panels"]:
        model, ctx, D = panel["model"], panel["ctx"], panel["D"]
        corpus = panel.get("corpus", "wikitext2")
        fracs = sorted(panel["fracs"])
        rows = sorted(panel["rows"],
                      key=lambda r: ORDER.index(r["selector"]) if r["selector"] in ORDER else 99)
        cols = "l r " + " ".join("r" for _ in fracs)
        lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
                 "selector & B/key & " + " & ".join(f"$k{{=}}{int(f*100)}\\%$" for f in fracs) + r" \\",
                 r"\midrule"]
        for r in rows:
            b = r["scoring_bytes"]
            bs = "evict" if r["is_eviction"] else (f"{b:.0f}" if b is not None else "--")
            vals = " & ".join("--" if _ppl(r, f) is None else f"{_ppl(r, f):.2f}" for f in fracs)
            lines.append(f"{SEL_LABEL.get(r['selector'], r['selector'])} & {bs} & {vals} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        suffix = "" if corpus == "wikitext2" else f"_{corpus}"
        fn = f"tables/a2_{model}_{ctx}{suffix}.tex"
        with open(fn, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {fn}  ({MODEL_LABEL.get(model, model)} S={ctx} D={D} "
              f"corpus={corpus} dense={panel['dense_ppl']:.2f})")


def emit_quest_pagesweep():
    """F4 fairness control: Quest at page sizes 16/32/64 vs the per-key baselines.
    Reads results/study_a2_qwen0.5b_16384_quest{16,32,64}.json (quest-only runs)."""
    import glob, re
    pages, dense = {}, None
    fracs = None
    for p in sorted(glob.glob("results/paper/study_a2_qwen0.5b_16384_quest*.json")):
        m = re.search(r"_quest(\d+)\.json$", p)
        if not m:
            continue
        page = int(m.group(1))
        d = json.load(open(p))
        dense = d.get("dense_ppl", dense)
        grid = d.get("grid", {})
        byf = {}
        for k, ppl in grid.items():
            mm = re.fullmatch(r"quest_k([0-9.]+)", k)
            if mm:
                byf[float(mm.group(1))] = ppl
        if byf:
            pages[page] = byf
            fracs = sorted(byf) if fracs is None else fracs
    if not pages:
        print("[quest-pagesweep] no results/study_a2_qwen0.5b_16384_quest*.json yet; skipping")
        return
    cols = "l " + " ".join("r" for _ in fracs)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "Quest page & " + " & ".join(f"$k{{=}}{int(f*100)}\\%$" for f in fracs) + r" \\",
             r"\midrule"]
    for page in sorted(pages):
        vals = " & ".join(f"{pages[page][f]:.2f}" for f in fracs)
        lines.append(f"page$=${page} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/f4_quest_pagesweep.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved tables/f4_quest_pagesweep.tex  (Qwen2.5-0.5B S=16K dense={dense:.2f})")


def emit_composition():
    """Composition axis: selection precision x attend-cache precision (2x2).
    Reads the four results/compose_<select>_<attend>.json files (each a standard
    eval_longctx run: selection in {exact,fp8}, attend cache in {bf16,fp8}). fp8 is the
    faithful near-lossless compressed attend cache (1 B/elem vs bf16's 2). Tests whether
    selection-loss and attend-compression-loss compose additively."""
    import re
    cells, fracs, dense = {}, set(), None
    for sel in ("exact", "fp8"):
        for att in ("bf16", "fp8"):
            p = f"results/paper/compose_{sel}_{att}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            # the true (bf16-attend) dense reference comes from the bf16 runs only;
            # fp8-attend runs report an fp8-attend dense, which is not the baseline.
            if att == "bf16":
                dense = d.get("dense_ppl", dense)
            for k, ppl in d.get("grid", {}).items():
                mm = re.fullmatch(rf"{sel}_k([0-9.]+)", k)
                if mm:
                    frac = float(mm.group(1))
                    cells[(sel, att, frac)] = ppl
                    fracs.add(frac)
    if not cells:
        print("[composition] no results/compose_*.json yet; skipping")
        return
    fracs = sorted(fracs)
    sel_lab = {"exact": "exact-select", "fp8": "fp8-select"}
    att_lab = {"bf16": "bf16 attend", "fp8": "fp8 attend"}
    cols = "l l " + " ".join("r" for _ in fracs)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "selection & attend & " + " & ".join(f"$k{{=}}{int(f*100)}\\%$" for f in fracs) + r" \\",
             r"\midrule"]
    for sel in ("exact", "fp8"):
        for att in ("bf16", "fp8"):
            vals = " & ".join(
                f"{cells[(sel, att, f)]:.2f}" if (sel, att, f) in cells else "--" for f in fracs)
            lines.append(f"{sel_lab[sel]} & {att_lab[att]} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/compose.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved tables/compose.tex  (Qwen2.5-0.5B S=16K dense={dense:.2f})")


def emit_profiling(prof):
    bw = prof["P1_P2_bandwidth"]
    rows = [("top-k gather (random)", bw["gather_random"]),
            ("top-k gather (contiguous)", bw["gather_contiguous"]),
            ("top-k gather (paged)", bw["gather_paged"])]
    lines = [r"\begin{tabular}{l r}", r"\toprule",
             r"access pattern & \% of HBM peak \\", r"\midrule"]
    lines += [f"{n} & {v:.0f} " + r"\\" for n, v in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/profiling_bandwidth.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/profiling_bandwidth.tex")

    ml = prof["method_latency_ms"]
    lab = {"kivi2": "KIVI-2", "kivi4": "KIVI-4", "bf16": "BF16 dense", "fp8": "FP8 dense",
           "sparse_fp8_k5": "two-pass sparse (fp8, $k{=}5\\%$)",
           "sparse_int4_k5": "two-pass sparse (int4 score, $k{=}5\\%$)",
           "sparse_int2_k5": "two-pass sparse (int2 score, $k{=}5\\%$)"}
    lines = [r"\begin{tabular}{l r}", r"\toprule",
             r"method & decode latency (ms) \\", r"\midrule"]
    for k, v in sorted(ml.items(), key=lambda kv: kv[1]):
        lines.append(f"{lab.get(k, k)} & {v:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/profiling_latency.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/profiling_latency.tex")


A1_LABEL = {"kivi": "KIVI-4", "kivi_2bit": "KIVI-2", "kvquant": "KVQuant-4",
            "kvquant_3bit": "KVQuant-3", "kvquant_2bit": "KVQuant-2",
            "turbo_quant": "TurboQuant", "turbo_quant_qjl": "TurboQuant-QJL",
            "pq_outlier": "PQ + outlier isolation", "pq_norot": "plain PQ (no rot.)"}
A1_ORDER = ["kivi", "kvquant", "pq_outlier", "pq_norot", "kvquant_3bit",
            "kivi_2bit", "kvquant_2bit", "turbo_quant", "turbo_quant_qjl"]


def emit_a1():
    """Compression quality-per-KEY-bit at a common FP8 value cache (isolates the key axis)."""
    base = load("compression_comparison")
    fp16 = base["dense_reference"]["mean"]
    ppl = dict(base["methods"])
    ppl["pq_norot"] = ppl["norot"]
    lines = [r"\begin{tabular}{l r r}", r"\toprule",
             r"method & key bpw & Wikitext-2 PPL \\", r"\midrule",
             f"FP16 (reference) & 16.0 & {fp16:.2f} " + r"\\", r"\midrule"]
    for k in A1_ORDER:
        if k not in ppl:
            continue
        m, s = ppl[k]["mean"], ppl[k]["std"]
        ps = f"{m:.2f}" + (f"$\\pm${s:.2f}" if s and s > 0.005 else "")
        bpw_key = "norot" if k == "pq_norot" else k
        if bpw_key not in base["bpw"]:
            raise ValueError(f"missing measured key bpw for {k}")
        lines.append(f"{A1_LABEL[k]} & {base['bpw'][bpw_key]:.2f} & {ps} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/a1_compression.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved tables/a1_compression.tex  (FP16={fp16:.2f}, matched FP8 value)")


def emit_capacity():
    cap = load("sparse_capacity")
    ctxs = sorted(cap.keys(), key=int)
    methods = ["BF16", "FP8", "KIVI-4", "Sparse-INT2"]
    lines = [r"\begin{tabular}{l " + " ".join("r" for _ in ctxs) + "}", r"\toprule",
             "method & " + " & ".join(f"$S{{=}}{int(c)//1024}$K" for c in ctxs) + r" \\",
             r"\midrule"]
    for me in methods:
        cells = []
        for c in ctxs:
            mb = cap[c].get(me, {}).get("max_batch")
            cells.append("--" if mb is None else str(mb))
        lines.append(f"{me} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/a4_capacity.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/a4_capacity.tex  (max decode batch on 20GB; sparse LOSES vs KIVI)")


def emit_f6():
    """Score-precision pitfall table from results/pitfall_<model>_<prec>.json."""
    import glob, re
    rows = {}
    for p in glob.glob("results/pitfall_*.json") + glob.glob("results/paper/pitfall_*.json"):
        m = re.fullmatch(r"pitfall_([a-z0-9.]+)_(bf16|fp16|fp32)\.json", os.path.basename(p))
        if not m:
            continue
        model, prec = m.group(1), m.group(2)
        d = json.load(open(p))
        ppl = d["grid"].get("exact_k0.1")
        rows.setdefault(model, {})["dense"] = d.get("dense_ppl")
        rows[model][prec] = ppl
        if prec == "fp32" or "diagnostics" not in rows[model]:
            rows[model]["diagnostics"] = d.get("score_diagnostics", {})
    if not rows:
        print("[f6] no results/pitfall_*.json yet; skipping")
        return
    order = ["qwen0.5b", "qwen1.5b", "qwen3b"]
    lines = [r"\begin{tabular}{l r r r r}", r"\toprule",
             r"model & max $|k|$ & exact (bf16) & exact (fp16) & exact (fp32) \\",
             r" & & \multicolumn{3}{c}{PPL @ $k{=}10\%$, dense in text} \\", r"\midrule"]
    def cell(v):
        if v is None:
            return "--"
        if v != v:                     # NaN: fp16 overflowed on the outlier channels
            return r"\textbf{NaN}$^{\dagger}$"
        return f"{v:.2f}"
    for mdl in [m for m in order if m in rows] + [m for m in rows if m not in order]:
        r = rows[mdl]
        diag = r.get("diagnostics", {})
        maxk = diag.get("max_abs_k", "--")
        maxk = f"{maxk:.0f}" if isinstance(maxk, (int, float)) else maxk
        lines.append(f"{MODEL_LABEL.get(mdl, mdl)} & {maxk} & "
                     f"{cell(r.get('bf16'))} & {cell(r.get('fp16'))} & {cell(r.get('fp32'))} " + r"\\")
    lines += [r"\bottomrule",
              r"\multicolumn{5}{l}{\footnotesize $^{\dagger}$fp16 dot products contain nonfinite values; see released score diagnostics.} \\",
              r"\end{tabular}"]
    with open("tables/f6_precision.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/f6_precision.tex")


def emit_ablation_value_cache():
    """F1 value-cache ablation table from tracked per-seed evidence."""
    p = "results/paper/ablation_value_cache.json"
    if not os.path.exists(p):
        print("[ablation_value_cache] missing; skipping")
        return
    with open(p) as f:
        data = json.load(f)
    
    key_names = {
        "norot": "PQ (No Rot.)",
        "rotated_std": "Rotated PQ",
        "rotated_cal": "Rotated PQ + calibration",
        "rotated_cal_sign": "\\quad + sign-pattern selection",
        "pq_outlier": "\\textbf{\\quad + outlier isolation}"
    }
    order = ["norot", "rotated_std", "rotated_cal", "rotated_cal_sign", "pq_outlier"]
    
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Key configuration} & \textbf{PQ value (2\,bpw)} & \textbf{FP8 value (8\,bpw)} \\",
        r"\midrule"
    ]
    
    for k in order:
        pq_mean = data["value_modes"]["pq"][k]["mean"]
        pq_std = data["value_modes"]["pq"][k]["std"]
        fp8_mean = data["value_modes"]["fp8"][k]["mean"]
        fp8_std = data["value_modes"]["fp8"][k]["std"]
        
        pq_str = f"{pq_mean:.2f}" + (f"$\\pm${pq_std:.2f}" if pq_std > 0.005 else "")
        fp8_str = f"{fp8_mean:.2f}" + (f"$\\pm${fp8_std:.2f}" if fp8_std > 0.005 else "")
        
        name = key_names[k]
        if k == "pq_outlier":
            pq_str = f"\\textbf{{{pq_str}}}"
            fp8_str = f"\\textbf{{{fp8_str}}}"
            
        lines.append(f"{name} & {pq_str} & {fp8_str} \\\\")
        
    lines += [
        r"\bottomrule",
        r"\end{tabular}"
    ]
    
    with open("tables/ablation_value_cache.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/ablation_value_cache.tex")


def emit_niah():
    """NIAH results table. Reads results/niah_8192.json, results/niah_hard_0.5b_8192.json,
    and results/niah_hard_1.5b_8192.json."""
    publication = "results/paper/niah_hard_qwen0.5b_8192.json"
    if os.path.exists(publication):
        d = json.load(open(publication))
        fracs = d["fracs"]
        selectors = [s for s in ("exact", "approx", "int4", "quest") if s in d["selectors"]]
        names = {"exact": "exact", "approx": "PQ-LUT", "int4": "int4 scalar", "quest": "Quest"}
        lines = [r"\begin{tabular}{l " + " ".join("c" for _ in fracs) + "}", r"\toprule",
                 "selector & " + " & ".join(f"$k{{=}}{100*f:g}\\%$" for f in fracs) + r" \\", r"\midrule"]
        for sel in selectors:
            cells = []
            for frac in fracs:
                key = f"{sel}_k{frac}"
                acc = d["grid"][key]
                lo, hi = d["confidence_intervals"][key]
                cells.append(f"{acc:.2f} [{lo:.2f},{hi:.2f}]")
            lines.append(names[sel] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open("tables/niah.tex", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("saved tables/niah.tex (100-case publication NIAH panel)")
        return

    files = {
        "std_0.5b": "results/niah_8192.json",
        "hard_0.5b": "results/niah_hard_0.5b_8192.json",
        "hard_1.5b": "results/niah_hard_1.5b_8192.json"
    }
    data = {}
    for k, p in files.items():
        if os.path.exists(p):
            with open(p) as f:
                data[k] = json.load(f)
        else:
            print(f"[niah] missing {p}")
    
    if len(data) < 3:
        print("[niah] skipping due to missing files")
        return
        
    selectors = ["exact", "approx", "int4", "quest"]
    sel_names = {
        "exact": "exact (fp32 score)",
        "approx": "PQ-LUT (PQCache)",
        "int4": "int4 scalar",
        "quest": "Quest (page)"
    }
    
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"Selector & \multicolumn{3}{c}{Standard (0.5B, $w{=}64$)} & \multicolumn{3}{c}{Hard (0.5B, $w{=}8$)} & \multicolumn{3}{c}{Hard (1.5B, $w{=}8$)} \\",
        r"         & $k{=}1\%$ & $k{=}2\%$ & $k{=}5\%$ & $k{=}0.5\%$ & $k{=}1\%$ & $k{=}2\%$ & $k{=}0.5\%$ & $k{=}1\%$ & $k{=}2\%$ \\",
        r"\midrule"
    ]
    
    for sel in selectors:
        row = [sel_names[sel]]
        for k_val in [0.01, 0.02, 0.05]:
            v = data["std_0.5b"]["grid"].get(f"{sel}_k{k_val}")
            row.append(f"{v:.2f}" if v is not None else "--")
        for k_val in [0.005, 0.01, 0.02]:
            v = data["hard_0.5b"]["grid"].get(f"{sel}_k{k_val}")
            row.append(f"{v:.2f}" if v is not None else "--")
        for k_val in [0.005, 0.01, 0.02]:
            v = data["hard_1.5b"]["grid"].get(f"{sel}_k{k_val}")
            row.append(f"{v:.2f}" if v is not None else "--")
        lines.append(" & ".join(row) + r" \\")
        
    lines += [
        r"\bottomrule",
        r"\end{tabular}"
    ]
    
    with open("tables/niah.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/niah.tex")


def emit_niah_transfer():
    """Emit the independent 7B NIAH panel, including its trial-level intervals.

    This is deliberately a separate table from the 0.5B control: it makes the
    scale-transfer check visible without pretending that it is another replicate of
    the small-model experiment.
    """
    p = "results/paper/niah_hard_qwen7b_8192.json"
    if not os.path.exists(p):
        print("[niah-transfer] missing 7B NIAH result; skipping")
        return
    d = json.load(open(p))
    expected = ("exact", "approx", "int4", "quest")
    selectors = [s for s in expected if s in d["selectors"]]
    names = {"exact": "exact", "approx": "PQ-LUT", "int4": "int4 scalar", "quest": "Quest"}
    fracs = d["fracs"]
    lines = [r"\begin{tabular}{l " + " ".join("c" for _ in fracs) + "}", r"\toprule",
             "selector & " + " & ".join(f"$k{{=}}{100*f:g}\\%$" for f in fracs) + r" \\", r"\midrule"]
    for sel in selectors:
        cells = []
        for frac in fracs:
            key = f"{sel}_k{frac}"
            acc = d["grid"][key]
            lo, hi = d["confidence_intervals"][key]
            cells.append(f"{acc:.2f} [{lo:.2f},{hi:.2f}]")
        lines.append(names[sel] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/niah_transfer.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/niah_transfer.tex (Qwen2.5-7B, 8K, 100 paired cases)")


def emit_presentation_summaries(par):
    """Small manuscript summaries generated from the same publication JSON as the
    full grids.  These are deliberately not hand-maintained copies of table values."""
    def panel(model, ctx):
        for p in par["panels"]:
            if p["model"] == model and p["ctx"] == ctx and p.get("corpus", "wikitext2") == "wikitext2":
                return p
        raise ValueError(f"missing Wikitext-2 panel for {model} at {ctx}")

    def value(p, selector, frac=0.05):
        for row in p["rows"]:
            if row["selector"] == selector:
                return _ppl(row, frac)
        raise ValueError(f"missing {selector} in {p['model']} S={p['ctx']}")

    configurations = [
        ("qwen0.5b", 16384, "Q0.5B"), ("qwen1.5b", 16384, "Q1.5B"),
        ("qwen3b", 16384, "Q3B"), ("llama3b", 16384, "L3B"),
        ("qwen7b", 8192, r"Q7B$^\dagger$"),
    ]
    panels = [panel(model, ctx) for model, ctx, _ in configurations]
    selectors = [("dense", None), ("exact-key", "exact"), ("fp8 scalar", "fp8"),
                 ("PQ-LUT", "approx"), ("int4 scalar", "int4"), ("Quest (page)", "quest")]
    lines = [r"\begin{tabular}{l " + " ".join("r" for _ in panels) + "}", r"\toprule",
             "selector ($k{=}5\\%$) & " + " & ".join(label for _, _, label in configurations) + r" \\",
             r"\midrule"]
    for label, selector in selectors:
        values = [p["dense_ppl"] if selector is None else value(p, selector) for p in panels]
        lines.append(label + " & " + " & ".join(f"{v:.2f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/selection_transfer.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/selection_transfer.tex")

    cap = load("sparse_capacity")
    contexts = sorted(cap, key=int)
    capacity_rows = []
    for method in ("BF16", "FP8", "KIVI-4", "Sparse-INT2"):
        capacity_rows.append((method, [cap[c][method]["max_batch"] for c in contexts]))
    composition = {}
    for selector in ("exact", "fp8"):
        for attend in ("bf16", "fp8"):
            data = load(f"compose_{selector}_{attend}")
            composition[(selector, attend)] = data["grid"][f"{selector}_k0.05"]
    lines = [r"\begin{tabular}{l r r}", r"\toprule",
             r"method & $S{=}16$K & $S{=}32$K \\", r"\midrule"]
    lines += [f"{method} & {values[0]} & {values[1]} " + r"\\" for method, values in capacity_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\hspace{1.4cm}",
              r"\begin{tabular}{l r}", r"\toprule",
              r"selection / attend cache & PPL @ $k{=}5\%$ \\", r"\midrule"]
    for selector, attend in (("exact", "bf16"), ("exact", "fp8"), ("fp8", "bf16"), ("fp8", "fp8")):
        lines.append(f"{selector} / {attend} & {composition[(selector, attend)]:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("tables/capacity_composition.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved tables/capacity_composition.tex")


def main():
    pareto = load("study_a2_pareto")
    emit_a2(pareto)
    emit_profiling(load("study_profiling"))
    emit_a1()
    emit_capacity()
    emit_f6()
    emit_quest_pagesweep()
    emit_composition()
    emit_ablation_value_cache()
    emit_niah()
    emit_niah_transfer()
    emit_presentation_summaries(pareto)


if __name__ == "__main__":
    main()
