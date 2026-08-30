"""Phase A2 — long-context sparse selection: does the Quest win survive past 512 ctx?

This is the make-or-break re-test (plan Gate A / claim C2). B1/B2 ran at 512 ctx
where Quest's page metadata is relatively coarse; Quest is *designed* for 16K-128K.
We re-measure ours vs exact-oracle vs Quest at 8K/16K/32K on real long text.

Two OOM-avoidance designs (both required past ~8K on a 20GB Ada):
  - Dense reference = our chunked path at frac=1.0 (= exact full attention, but
    O(Q_CHUNK*K_CHUNK) memory instead of O(S^2)).
  - Loss computed ONLY on the last `eval_window` positions (those have the longest
    context — the meaningful long-context metric) via base-model hidden states +
    lm_head on the tail, so we never materialize [1,S,vocab] logits.

Run: python -m benchmarks.eval_longctx_sparse --ctx 16384 --model Qwen/Qwen2.5-0.5B
"""
import argparse, json, math
from types import SimpleNamespace
import torch
import benchmarks.eval_ppl as E
from benchmarks.eval_ppl import patch_qwen_attention, run_calibration_and_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from benchmarks.statistics import bootstrap_geomean_ci


def longctx_loss(model, input_ids, eval_window):
    """CE loss (nats/token) on the last `eval_window` positions, no full-logits alloc."""
    base = model.model(input_ids).last_hidden_state            # [1, S, H]
    W = eval_window
    hid = base[:, -W - 1:-1, :]                                # predicts tokens -W..-1
    tgt = input_ids[:, -W:]
    logits = model.lm_head(hid).float()                        # [1, W, vocab]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), tgt.reshape(-1)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--bits-k", type=int, default=8)
    p.add_argument("--d-sub", type=int, default=2)
    p.add_argument("--outliers", type=int, default=2)
    p.add_argument("--eval-window", type=int, default=512)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--q-chunk", type=int, default=128)
    p.add_argument("--local-w", type=int, default=0,
                   help="recent-window tokens forced into every selector (fair Quest/H2O/SnapKV protocol)")
    p.add_argument("--k-chunk", type=int, default=4096)
    p.add_argument("--attend-fp8", action="store_true",
                   help="fp8-roundtrip the attend cache (simulate the fp8 gather kernel) — quality check")
    p.add_argument("--attend-dtype", default="bf16", choices=["bf16", "fp8", "int4"],
                   help="precision of the attend cache after selection (composition axis): "
                        "bf16 (exact), fp8, or int4 fakequant. --attend-fp8 is a legacy alias for fp8.")
    p.add_argument("--sparq-r", type=int, default=16,
                   help="SparQ: number of top-|q| query channels used for scoring (r/D scan)")
    p.add_argument("--score-bf16", action="store_true",
                   help="legacy alias for --score-dtype bf16 (exposes the key-outlier pitfall)")
    p.add_argument("--score-dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="selection-score accumulation precision; default fp32 = true oracle. "
                        "fp16 tests whether mantissa (not dynamic range) is the fix for the pitfall.")
    p.add_argument("--score-key-clip", type=float, default=None,
                   help="controlled numerical intervention: cap absolute selection-key channels; "
                        "selection only, never the exact post-selection attention keys/values")
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "pg19"],
                   help="evaluation corpus; pg19 is the long-form held-out second corpus")
    p.add_argument("--load-4bit", action="store_true",
                   help="load model weights in NF4 4-bit (bitsandbytes) for large models on 20GB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selectors", default="exact,approx,quest",
                   help="comma-separated; A3 oracle bake-off: exact,approx,int4,fp8,recent,random")
    p.add_argument("--fracs", default="0.01,0.02,0.05,0.10",
                   help="comma-separated selection fractions")
    p.add_argument("--diagnose-scores", action="store_true",
                   help="record score range/nonfinite/top-k agreement diagnostics (F6)")
    p.add_argument("--output-json", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out_json = a.output_json or f"results/longctx_{a.ctx}.json"

    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=bnb, device_map="auto", attn_implementation="eager")
        print(f"[load-4bit] {a.model} weights loaded as NF4 (compute dtype bf16)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    patch_qwen_attention(model)

    # Calibration + eval text from the chosen corpus. The recipe is identical across
    # corpora (>80-char lines joined into one contiguous stream) so only the source differs.
    if a.dataset == "pg19":
        # emozilla/pg19 is the script-free (parquet) mirror of DeepMind PG19.
        # Books are long; a handful of test books gives plenty for 16K-32K windows.
        ds_te = load_dataset("emozilla/pg19", split="test")
        books = [ds_te[i]["text"] for i in range(min(len(ds_te), 6))]
        calib = books[0][:6000]                       # calibration from the first book's head
        te_text = "\n\n".join(b[6000:] for b in books)  # eval on the remainder (disjoint)
    else:
        ds_tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds_te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        calib = " ".join(l.strip() for l in ds_tr["text"] if len(l.strip()) > 80)[:6000]
        te_text = " ".join(l for l in ds_te["text"] if len(l.strip()) > 0)
    # one long contiguous token stream for the eval windows
    te_ids = tok(te_text, return_tensors="pt").input_ids[0]
    need = a.n_windows * a.ctx
    assert te_ids.numel() >= need, f"need {need} tokens, have {te_ids.numel()}"

    selectors = [s.strip() for s in a.selectors.split(",") if s.strip()]
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
                "pq_k": None,
                "sign_pattern": "no_rot",
                "outlier_indices": torch.zeros((model.config.num_key_value_heads, 0), dtype=torch.long, device=model.device),
                "method": "pq_outlier"
            }

    E.SPARSE_PAGE = a.page; E.Q_CHUNK = a.q_chunk; E.K_CHUNK = a.k_chunk; E.LOCAL_W = a.local_w
    # attend cache precision (composition axis). --score-bf16 is a legacy alias.
    attend_mode = "fp8" if a.attend_fp8 else a.attend_dtype
    E.ATTEND_MODE = attend_mode
    E.ATTEND_FP8 = attend_mode == "fp8"   # back-compat for any reader of the old flag
    import benchmarks.sparse_attention as SA
    SA.SPARQ_R = a.sparq_r
    score_dtype = "bf16" if a.score_bf16 else a.score_dtype
    SA.SCORE_DTYPE = score_dtype
    SA.SCORE_FP32 = score_dtype != "bf16"  # legacy flag stays consistent
    SA.SCORE_KEY_CLIP = a.score_key_clip
    SA.reset_score_diagnostics(enabled=a.diagnose_scores)
    if attend_mode != "bf16":
        print(f"[attend-{attend_mode}] sparse attend cache is {attend_mode}-fakequantized")
    if score_dtype != "fp32":
        print(f"[score-{score_dtype}] selection scores accumulated in {score_dtype}")
    windows = [te_ids[i * a.ctx:(i + 1) * a.ctx].unsqueeze(0).to(model.device)
               for i in range(a.n_windows)]

    def per_window_ppl(selector, frac):
        """Return (mean_ppl, [per-window ppl]) — per-window list enables CI estimation."""
        E.MODE = "sparse"; E.SPARSE_SELECTOR = selector; E.SPARSE_FRAC = frac
        ls = []
        with torch.no_grad():
            for w in windows:
                ls.append(longctx_loss(model, w, a.eval_window))
                torch.cuda.empty_cache()
        ppls = [math.exp(l) for l in ls]
        return math.exp(sum(ls) / len(ls)), ppls, ls

    import os
    meta = {"model": a.model, "ctx": a.ctx, "eval_window": a.eval_window,
            "n_windows": a.n_windows, "page": a.page, "local_w": a.local_w,
            "dataset": a.dataset, "score_dtype": score_dtype, "attend_mode": attend_mode,
            "score_key_clip": a.score_key_clip,
            "load_4bit": a.load_4bit, "diagnose_scores": a.diagnose_scores,
            "seed": a.seed, "fracs": [float(x) for x in a.fracs.split(",") if x.strip()],
            "model_revision": getattr(model.config, "_commit_hash", None),
            "dataset_fingerprint": getattr(ds_te, "_fingerprint", None)}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                res = json.load(f)
            mismatched = [k for k, v in meta.items() if k in res and res[k] != v]
            if mismatched:
                raise ValueError(f"refusing to merge incompatible result file; mismatched {mismatched}")
            res.update(meta)
            if "grid" not in res:
                res["grid"] = {}
            if "windows" not in res:
                res["windows"] = {}
        except ValueError:
            raise
        except Exception:
            res = {**meta, "grid": {}, "windows": {}}
    else:
        res = {**meta, "grid": {}, "windows": {}}
    # dense reference via SDPA (flash, O(S) memory; frac ignored for "dense")
    cached_dense = res.get("losses", {}).get("dense")
    if isinstance(cached_dense, list) and len(cached_dense) == a.n_windows:
        dense_losses = cached_dense
        dense_w = [math.exp(x) for x in dense_losses]
        dense = math.exp(sum(dense_losses) / len(dense_losses))
    else:
        dense, dense_w, dense_losses = per_window_ppl("dense", 1.0)
    res["dense_ppl"] = dense
    res["windows"]["dense"] = dense_w
    res.setdefault("losses", {})["dense"] = dense_losses
    res.setdefault("confidence_intervals", {})["dense"] = bootstrap_geomean_ci(dense_losses, seed=a.seed)
    print(f"\n[ctx={a.ctx}] dense PPL = {dense:.3f}\n")

    fracs = [float(x) for x in a.fracs.split(",") if x.strip()]
    selectors = [s.strip() for s in a.selectors.split(",") if s.strip()]
    print(f"{'selector':>8} | " + " ".join(f"k={int(f*100)}%".rjust(8) for f in fracs))
    print("-" * 52)
    for sel in selectors:
        row = []
        for frac in fracs:
            key = f"{sel}_k{frac}"
            cached = res.get("losses", {}).get(key)
            if isinstance(cached, list) and len(cached) == a.n_windows and key in res.get("grid", {}):
                row.append(f"{res['grid'][key]:.2f}".rjust(8))
                continue
            ppl, ppls, losses = per_window_ppl(sel, frac)
            res["grid"][key] = ppl
            res["windows"][key] = ppls
            res.setdefault("losses", {})[key] = losses
            res.setdefault("confidence_intervals", {})[key] = bootstrap_geomean_ci(losses, seed=a.seed)
            row.append(f"{ppl:.2f}".rjust(8))
            with open(out_json, "w") as f:
                json.dump(res, f, indent=2)
        print(f"{sel:>8} | " + " ".join(row))
    if a.diagnose_scores:
        res["score_diagnostics"] = SA.get_score_diagnostics()
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\ndense={dense:.2f}  saved -> {out_json}")


if __name__ == "__main__":
    main()
