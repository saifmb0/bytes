"""Phase A/C3 decider — Needle-in-a-Haystack retrieval under sparse selection.

The C3 PPL test failed: with a fair recent window, cheap INT4 scoring ties PQ-LUT,
because next-token PPL is dominated by local context and doesn't stress selection
fidelity. Retrieval DOES: a secret code is planted DEEP in a real-text haystack
(far outside the recent window), and the model must copy it at the end. A selector
that fails to pick the needle's keys cannot answer. This is the metric that
separates a faithful oracle (PQ-LUT) from a noisy one (INT4) — if it can.

Teacher-forced exact-match: feed [haystack+needle ... query + CODE], predict the
CODE tokens from context (argmax), score exact match. Single forward, tail-only
lm_head (no full-logits OOM). Selectors share a common recent window (fair).

Run: python -m benchmarks.eval_niah_sparse --ctx 16384 --selectors exact,approx,int4,quest
"""
import argparse, json, random
import os
from types import SimpleNamespace
import torch
import benchmarks.eval_ppl as E
from benchmarks.eval_ppl import patch_qwen_attention, run_calibration_and_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from benchmarks.statistics import wilson_interval, paired_accuracy_difference_ci

NEEDLE_TMPL = " The special magic number is {code}. "
QUERY_TMPL = " The special magic number is"


def build_input(hay_tokens, needle_ids, query_ids, code_ids, depth, ctx, dev):
    """Insert needle at `depth` fraction, append query+code; return (ids, ans_start)."""
    budget = ctx - len(needle_ids) - len(query_ids) - len(code_ids)
    hay = hay_tokens[:budget]
    p = int(depth * len(hay))
    prefix = hay[:p] + needle_ids + hay[p:] + query_ids
    full = prefix + code_ids
    ans_start = len(prefix)               # first code token position
    return torch.tensor(full, device=dev).unsqueeze(0), ans_start


def retrieval_correct(model, ids, ans_start, code_ids):
    """Teacher-forced: are all CODE tokens the argmax given context? (exact match)."""
    base = model.model(ids).last_hidden_state                  # [1,S,H]
    # hidden at positions [ans_start-1 .. ans_start+len(code)-2] predict the code tokens
    n = len(code_ids)
    hid = base[:, ans_start - 1: ans_start - 1 + n, :]
    logits = model.lm_head(hid).float()                        # [1,n,vocab]
    pred = logits.argmax(-1)[0].tolist()
    return int(pred == list(code_ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--bits-k", type=int, default=8)
    p.add_argument("--d-sub", type=int, default=2)
    p.add_argument("--outliers", type=int, default=2)
    p.add_argument("--local-w", type=int, default=64)
    p.add_argument("--q-chunk", type=int, default=64)
    p.add_argument("--selectors", default="exact,approx,int4,quest")
    p.add_argument("--fracs", default="0.01,0.02,0.05")
    p.add_argument("--depths", default="0.1,0.5,0.9")
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out_json = a.output_json or f"results/niah_{a.ctx}.json"

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    patch_qwen_attention(model)
    ds_tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib = " ".join(l.strip() for l in ds_tr["text"] if len(l.strip()) > 80)[:6000]
    hay_text = " ".join(l for l in ds_te["text"] if len(l.strip()) > 0)
    hay_tokens = tok(hay_text, return_tensors="pt").input_ids[0].tolist()

    selectors = [s.strip() for s in a.selectors.split(",")]
    if "approx" in selectors:
        args = SimpleNamespace(bits_k=a.bits_k, bits_v=a.bits_k, d_sub=a.d_sub,
                               outliers=a.outliers, no_rot=False, share_lut=False,
                               seed=a.seed, selection_only=True)
        run_calibration_and_training(model, tok, calib, args)
        for idx in E.QUANT_CONFIGS:
            mc = E.QUANT_CONFIGS[idx]["pq_outlier"]
            E.QUANT_CONFIGS[idx]["method"] = "pq_outlier"
            for f in ["pq_k", "sign_pattern", "outlier_indices"]:
                E.QUANT_CONFIGS[idx][f] = mc[f]
    else:
        E.QUANT_CONFIGS = {}
        for idx in range(model.config.num_hidden_layers):
            E.QUANT_CONFIGS[idx] = {
                "pq_k": None, "sign_pattern": "no_rot",
                "outlier_indices": torch.zeros(
                    (model.config.num_key_value_heads, 0), dtype=torch.long, device=model.device),
                "method": "pq_outlier"}
    E.Q_CHUNK = a.q_chunk; E.K_CHUNK = 4096; E.LOCAL_W = a.local_w

    query_ids = tok(QUERY_TMPL, add_special_tokens=False).input_ids
    fracs = [float(x) for x in a.fracs.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    # fixed trial set (same needles/depths for every selector -> paired comparison)
    trials = []
    for trial_id in range(a.n_trials):
        code = "".join(rng.choice("0123456789") for _ in range(7))
        # leading-space variant: in-context the code follows "is " (a space), so the
        # model predicts the space-prefixed digit tokens. Must match for exact-match.
        code_ids = tok(" " + code, add_special_tokens=False).input_ids
        needle_ids = tok(NEEDLE_TMPL.format(code=code), add_special_tokens=False).input_ids
        for d in depths:
            trials.append({"trial_id": trial_id, "code": code, "depth": d,
                           "needle_ids": needle_ids, "code_ids": code_ids})

    res = {"model": a.model, "ctx": a.ctx, "local_w": a.local_w,
           "seed": a.seed, "n_codes": a.n_trials, "n_cases": len(trials),
           "model_revision": getattr(model.config, "_commit_hash", None),
           "dataset_fingerprint": getattr(ds_te, "_fingerprint", None),
           "depths": depths, "fracs": fracs, "selectors": selectors,
           "grid": {}, "confidence_intervals": {}, "trial_records": [], "raw_outcomes": {}}
    if os.path.exists(out_json):
        old = json.load(open(out_json))
        identity = ("model", "ctx", "local_w", "seed", "n_codes", "depths", "fracs", "selectors")
        if all(old.get(k) == res.get(k) for k in identity):
            for key in ("grid", "confidence_intervals", "dense_conditioned",
                        "paired_difference_ci95", "raw_outcomes", "dense_acc", "dense_ci95"):
                if key in old:
                    res[key] = old[key]
    print(f"NIAH ctx={a.ctx} local_w={a.local_w} trials={a.n_trials} depths={depths}")
    print(f"{'selector':>8} | " + " ".join(f"k={int(f*100)}%".rjust(7) for f in fracs))
    print("-" * 50)
    # Dense first: it defines the model-capability ceiling used for conditioned scores.
    E.MODE = "sparse"; E.SPARSE_SELECTOR = "dense"; E.SPARSE_FRAC = 1.0
    dense_outcomes = res["raw_outcomes"].get("dense")
    if dense_outcomes is None:
        dense_outcomes = []
        with torch.no_grad():
            for t in trials:
                ids, ans_start = build_input(hay_tokens, t["needle_ids"], query_ids,
                                             t["code_ids"], t["depth"], a.ctx, model.device)
                dense_outcomes.append(retrieval_correct(model, ids, ans_start, t["code_ids"]))
                torch.cuda.empty_cache()
        res["raw_outcomes"]["dense"] = dense_outcomes
    res["dense_acc"] = sum(dense_outcomes) / len(dense_outcomes)
    res["dense_ci95"] = wilson_interval(sum(dense_outcomes), len(dense_outcomes))
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    outcomes = {}
    with torch.no_grad():
        for sel in selectors:
            E.MODE = "sparse"; E.SPARSE_SELECTOR = sel
            row = []
            for frac in fracs:
                E.SPARSE_FRAC = frac
                key = f"{sel}_k{frac}"
                if key in res["raw_outcomes"]:
                    cell_outcomes = res["raw_outcomes"][key]
                    outcomes[key] = cell_outcomes
                    row.append(f"{sum(cell_outcomes)/len(cell_outcomes):.2f}".rjust(7))
                    continue
                correct = 0
                cell_outcomes = []
                for t in trials:
                    ids, ans_start = build_input(hay_tokens, t["needle_ids"], query_ids,
                                                 t["code_ids"], t["depth"], a.ctx, model.device)
                    outcome = retrieval_correct(model, ids, ans_start, t["code_ids"])
                    correct += outcome
                    cell_outcomes.append(outcome)
                    torch.cuda.empty_cache()
                acc = correct / len(trials)
                outcomes[key] = cell_outcomes
                res["raw_outcomes"][key] = cell_outcomes
                res["grid"][key] = acc
                res["confidence_intervals"][key] = wilson_interval(correct, len(trials))
                dense_ok = [i for i, ok in enumerate(dense_outcomes) if ok]
                res.setdefault("dense_conditioned", {})[key] = (
                    sum(cell_outcomes[i] for i in dense_ok) / len(dense_ok) if dense_ok else None)
                row.append(f"{acc:.2f}".rjust(7))
                os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
                with open(out_json, "w") as f:
                    json.dump(res, f, indent=2)
            print(f"{sel:>8} | " + " ".join(row))

    for i, t in enumerate(trials):
        rec = {"trial_id": t["trial_id"], "code": t["code"], "depth": t["depth"],
               "dense_correct": dense_outcomes[i],
               "outcomes": {key: vals[i] for key, vals in outcomes.items()}}
        res["trial_records"].append(rec)
    if "quest" in selectors:
        for sel in ("exact", "approx", "int4"):
            if sel not in selectors:
                continue
            for frac in fracs:
                akey, qkey = f"{sel}_k{frac}", f"quest_k{frac}"
                res.setdefault("paired_difference_ci95", {})[f"{sel}_minus_quest_k{frac}"] = \
                    paired_accuracy_difference_ci(outcomes[akey], outcomes[qkey], seed=a.seed)
    print(f"{'dense':>8} | full-attention retrieval acc = {res['dense_acc']:.2f}")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
