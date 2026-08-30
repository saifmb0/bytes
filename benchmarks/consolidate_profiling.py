"""Consolidate the decode-latency PROFILING into one per-method roofline + latency-
decomposition table (study axes P1-P3). All numbers trace to tracked results/paper/*.json
measured on the RTX 4000 Ada (CUDA graphs, 20 warmup / 200 runs) at the serving point
B=32, S=16384, D=128, H_q=64, H_k=8.

P1  access pattern: achieved HBM bandwidth of random vs contiguous top-k gather
    (results/paper/access_pattern.json) -> "a gathered byte is ~as cheap as a contiguous one".
P2  per-method achieved % of peak (the 'not all bytes are equal' core):
    gather ~76-83%, BF16 contiguous ~49%, KIVI packed-INT ~33% (=> a KIVI byte costs
    ~2.4x a gathered byte).
P3  latency decomposition of the two-pass sparse decode (score / topk / gather) from
    results/paper/sparse_crossover.json. This establishes component timings, not
    instruction-pipeline or library-algorithm attribution.

Run: python -m benchmarks.consolidate_profiling
"""
import json
import os

S_REF = 16384


def load(name):
    path = f"results/paper/{name}.json"
    with open(path) as f:
        return json.load(f)


def main():
    ap = load("access_pattern")
    xover = load("sparse_crossover")
    kivi = load("kivi_latency")
    peak = ap["_meta"]["peak_gbs"]

    out = {"operating_point": {"B": 32, "S": S_REF, "D": 128, "H_q": 64, "H_k": 8,
                               "peak_gbs": peak, "gpu": "RTX 4000 Ada"},
           "P1_P2_bandwidth": {}, "P3_decomposition": {}, "method_latency_ms": {}}

    # --- P1: achieved bandwidth by access pattern (avg over budgets) ---
    by_pat = {}
    for r in ap["rows"]:
        by_pat.setdefault(r["pattern"], []).append(r["pct_peak"])
    print(f"=== P1/P2 achieved bandwidth (% of {peak:.0f} GB/s peak), gather @ S={ap['_meta']['S']} ===")
    for pat, v in by_pat.items():
        avg = sum(v) / len(v)
        out["P1_P2_bandwidth"][f"gather_{pat}"] = round(avg, 1)
        print(f"  top-k gather ({pat:>10}): {avg:5.1f}%  (range {min(v):.0f}-{max(v):.0f}%)")
    # Do not combine this measurement with an unmeasured KIVI bandwidth.  The KIVI
    # latency file is independently rerun below, but does not expose a comparable
    # bytes-moved counter, so a cross-kernel ``cost per byte'' ratio is not evidence.

    # --- P3: two-pass decomposition + full method latency @ S_REF ---
    cells = [c for c in xover if c["S"] == S_REF]
    print(f"\n=== P3 two-pass sparse decode decomposition @ S={S_REF} (ms) ===")
    print(f"{'frac':>6} | {'score':>7} {'topk':>7} {'gather':>7} | {'sparse_fp8':>10} {'sparse_int4':>11}")
    for c in sorted(cells, key=lambda x: x["frac"]):
        rl = c["roofline"]; lm = c["lat_ms"]
        out["P3_decomposition"][f"k{c['frac']}"] = {
            "score_pass_ms": rl["score_pass_ms"], "topk_ms": rl["topk_only_ms"],
            "gather_ms": rl["gather_only_ms"], "sparse_fp8": lm.get("sparse_fp8"),
            "sparse_int4": lm.get("sparse_int4"), "sparse_int2": lm.get("sparse_int2")}
        print(f"{c['frac']:>6.2f} | {rl['score_pass_ms']:7.2f} {rl['topk_only_ms']:7.2f} "
              f"{rl['gather_only_ms']:7.2f} | {lm.get('sparse_fp8'):10.2f} {lm.get('sparse_int4'):11.2f}")

    # full method latency table @ S_REF (one representative frac=0.05 for sparse)
    ref = next(c for c in cells if abs(c["frac"] - 0.05) < 1e-6)
    ml = {"bf16": ref["lat_ms"]["bf16"], "fp8": ref["lat_ms"]["fp8"],
          "kivi4": kivi["KIVI-4"]["lat_ms"], "kivi2": kivi["KIVI-2"]["lat_ms"],
          "sparse_fp8_k5": ref["lat_ms"]["sparse_fp8"],
          "sparse_int4_k5": ref["lat_ms"]["sparse_int4"],
          "sparse_int2_k5": ref["lat_ms"]["sparse_int2"]}
    out["method_latency_ms"] = ml
    print(f"\n=== method decode latency @ S={S_REF} (sparse @ k=5%) ===")
    for k, v in sorted(ml.items(), key=lambda kv: kv[1]):
        print(f"  {k:>16}: {v:6.2f} ms")

    os.makedirs("results/paper", exist_ok=True)
    with open("results/paper/study_profiling.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/paper/study_profiling.json")


if __name__ == "__main__":
    main()
