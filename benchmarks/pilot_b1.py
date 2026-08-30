"""Pilot B1 — selection fidelity: does LUT-approx scoring pick the same top-k as exact?

Recall = |top-k(approx) ∩ top-k(exact)| / k, measured on post-RoPE attention scores
(the faithful selection space). Gate: recall >= ~0.90 at k≈10% -> the LUT is a usable
relevance oracle for sparse attention; below that, Option B is dead.
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
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default="results/phase0_b1_recall.json")
    a = p.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    args = SimpleNamespace(bits_k=a.bits_k, bits_v=a.bits_k, d_sub=a.d_sub,
                           outliers=a.outliers, no_rot=False, share_lut=False, seed=a.seed)

    print(f"Loading {a.model} (cb{2**a.bits_k}/d_sub={a.d_sub}, payload={a.bits_k/a.d_sub:.2f} bpw)")
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

    # promote pq_outlier + common FP8 value cache
    for idx in E.QUANT_CONFIGS:
        mc = E.QUANT_CONFIGS[idx]["pq_outlier"]
        E.QUANT_CONFIGS[idx]["method"] = "pq_outlier"
        for f in ["pq_k", "pq_v", "sign_pattern", "outlier_indices", "R", "scalar_q",
                  "P_jl", "share_lut", "kivi_q", "kvq"]:
            if f in mc:
                E.QUANT_CONFIGS[idx][f] = mc[f]
            elif f in E.QUANT_CONFIGS[idx]:
                del E.QUANT_CONFIGS[idx][f]
        E.QUANT_CONFIGS[idx]["value_mode"] = "fp8"

    E.MODE = "quantize"
    E.COMPUTE_RECALL = True
    E.RECALL_STATS.clear()
    ppl, kl = evaluate_perplexity(model, tok, te, a.max_samples)

    # aggregate
    out = {"config": f"cb{2**a.bits_k}/d_sub={a.d_sub}", "payload_bpw": a.bits_k / a.d_sub,
           "ppl": ppl, "kl": kl, "recall": {}}
    print(f"\nPPL={ppl:.2f}  KL={kl:.4f}")
    print(f"\n{'ctx':>5} | " + " ".join(f"k={int(f*100)}%".rjust(8) for f in E.RECALL_FRACS))
    print("-" * 60)
    for ctx in E.RECALL_CTXS:
        row = []
        for frac in E.RECALL_FRACS:
            vals = E.RECALL_STATS.get((ctx, frac), [])
            m = sum(vals) / len(vals) if vals else float("nan")
            out["recall"][f"ctx{ctx}_k{int(frac*100)}"] = m
            row.append(f"{m:.3f}".rjust(8))
        print(f"{ctx:>5} | " + " ".join(row))
    json.dump(out, open(a.output_json, "w"), indent=2)
    print(f"\nsaved -> {a.output_json}")


if __name__ == "__main__":
    main()
