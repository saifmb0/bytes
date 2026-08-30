"""Pilot B2 — does top-k sparse attention preserve quality, and does OUR LUT
selector beat Quest's page bound?

Measures Wikitext-2 PPL under fixed-budget top-k attention for three selectors
(exact oracle / approx LUT (ours) / Quest), vs the dense FP16 reference.
Gates: ours ~ oracle (the 0.89 recall misses are harmless) AND ours <= Quest at
matched budget (finer per-key selection beats page bounds).
"""
import argparse, json
from types import SimpleNamespace
import torch
import benchmarks.eval_ppl as E
from benchmarks.eval_ppl import patch_qwen_attention, run_calibration_and_training, evaluate_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--bits-k", type=int, default=8)
    p.add_argument("--d-sub", type=int, default=2)
    p.add_argument("--outliers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=15)
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default="results/phase0_b2_sparse.json")
    a = p.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    args = SimpleNamespace(bits_k=a.bits_k, bits_v=a.bits_k, d_sub=a.d_sub,
                           outliers=a.outliers, no_rot=False, share_lut=False, seed=a.seed)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(a.model)
    patch_qwen_attention(model)
    ds_tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tr = [l.strip() for l in ds_tr["text"] if len(l.strip()) > 80]
    te = {"text": [l.strip() for l in ds_te["text"] if len(l.strip()) > 80]}
    calib = " ".join(tr[:10])[:6000]
    run_calibration_and_training(model, tok, calib, args)

    for idx in E.QUANT_CONFIGS:
        mc = E.QUANT_CONFIGS[idx]["pq_outlier"]
        E.QUANT_CONFIGS[idx]["method"] = "pq_outlier"
        for f in ["pq_k", "sign_pattern", "outlier_indices"]:
            E.QUANT_CONFIGS[idx][f] = mc[f]

    E.SPARSE_PAGE = a.page

    # dense reference
    E.MODE = "baseline"
    dense_ppl, _ = evaluate_perplexity(model, tok, te, a.max_samples)
    print(f"\nDense FP16 reference PPL = {dense_ppl:.3f}\n")

    fracs = [0.05, 0.10, 0.20]
    selectors = ["exact", "approx", "quest"]
    res = {"dense_ppl": dense_ppl, "page": a.page, "grid": {}}
    print(f"{'selector':>8} | " + " ".join(f"k={int(f*100)}%".rjust(9) for f in fracs))
    print("-" * 48)
    for sel in selectors:
        row = []
        for frac in fracs:
            E.MODE = "sparse"; E.SPARSE_SELECTOR = sel; E.SPARSE_FRAC = frac
            ppl, _ = evaluate_perplexity(model, tok, te, a.max_samples)
            res["grid"][f"{sel}_k{int(frac*100)}"] = ppl
            row.append(f"{ppl:.2f}".rjust(9))
        print(f"{sel:>8} | " + " ".join(row))
    json.dump(res, open(a.output_json, "w"), indent=2)
    print(f"\ndense={dense_ppl:.2f}  saved -> {a.output_json}")


if __name__ == "__main__":
    main()
