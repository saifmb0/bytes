"""Aggregate tracked per-seed compression runs into publication evidence."""
import glob
import json
import math
import os
import statistics


METHODS = ["turbo_quant", "turbo_quant_qjl", "kivi", "kivi_2bit", "kvquant",
           "kvquant_3bit", "kvquant_2bit", "norot", "rotated_std",
           "rotated_cal", "rotated_cal_sign", "pq_outlier"]
ABLATION = ["norot", "rotated_std", "rotated_cal", "rotated_cal_sign", "pq_outlier"]


def stats(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError(f"non-finite or empty observations: {values}")
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values), "observations": values}


def load_mode(mode):
    paths = sorted(glob.glob(f"results/paper/compression_{mode}_seed*.json"))
    if len(paths) != 5:
        raise SystemExit(f"expected 5 {mode} seed files, found {len(paths)}")
    rows = [json.load(open(p)) for p in paths]
    seeds = [r["seed"] for r in rows]
    if sorted(seeds) != list(range(5)):
        raise SystemExit(f"{mode}: expected seeds 0..4, got {seeds}")
    invariant = ("model", "model_revision", "dataset_fingerprint", "max_samples",
                 "bits_k", "bits_v", "d_sub", "outliers", "value_mode")
    for key in invariant:
        if len({str(r.get(key)) for r in rows}) != 1:
            raise SystemExit(f"{mode}: inconsistent {key}")
    return rows


def main():
    fp8, pq = load_mode("fp8"), load_mode("pq")
    meta = {k: fp8[0].get(k) for k in
            ("model", "model_revision", "dataset", "dataset_fingerprint",
             "max_samples", "bits_k", "bits_v", "d_sub", "outliers")}
    baseline = stats([r["fp16_ppl"] for r in fp8])
    matched = {m: stats([r["results"][m] for r in fp8]) for m in METHODS}
    ablation = {
        "schema": "value_cache_ablation_v1", **meta, "seeds": list(range(5)),
        "dense_reference": baseline,
        "value_modes": {
            "pq": {m: stats([r["results"][m] for r in pq]) for m in ABLATION},
            "fp8": {m: stats([r["results"][m] for r in fp8]) for m in ABLATION},
        },
        "source_files": [f"compression_{x}_seed{s}.json" for x in ("fp8", "pq") for s in range(5)]
    }
    compression = {
        "schema": "compression_comparison_v1", **meta, "seeds": list(range(5)),
        "dense_reference": baseline, "methods": matched,
        "bpw": fp8[0]["bpw_baselines"] | {"turbo_quant": 4.0,
            "turbo_quant_qjl": fp8[0]["bpw_qjl"], "norot": 4.0,
            "pq_outlier": 4.25},
        "source_files": [f"compression_fp8_seed{s}.json" for s in range(5)]
    }
    os.makedirs("results/paper", exist_ok=True)
    json.dump(ablation, open("results/paper/ablation_value_cache.json", "w"), indent=2)
    json.dump(compression, open("results/paper/compression_comparison.json", "w"), indent=2)
    print("saved tracked compression aggregates")


if __name__ == "__main__":
    main()
