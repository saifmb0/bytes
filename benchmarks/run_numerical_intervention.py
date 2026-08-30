"""Run and retain the controlled score-key clipping intervention.

The experiment holds model, windows, selector, budget, and exact post-selection
attention fixed.  It varies only score accumulation dtype and an absolute cap on
selection-key channels.  This tests the proposed key-outlier mechanism directly.
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--caps", default="256,128,64")
    p.add_argument("--output-json", default="results/paper/numerical_intervention.json")
    a = p.parse_args()
    run_dir = os.path.splitext(a.output_json)[0] + "_runs"
    os.makedirs(run_dir, exist_ok=True)
    conditions = [("fp32", None), ("bf16", None)] + [("bf16", float(v)) for v in a.caps.split(",")]
    records = []
    for dtype, cap in conditions:
        tag = f"{dtype}_noclip" if cap is None else f"{dtype}_clip{cap:g}"
        path = os.path.join(run_dir, tag + ".json")
        cmd = [sys.executable, "-m", "benchmarks.eval_longctx_sparse",
               "--model", a.model, "--ctx", str(a.ctx), "--n-windows", str(a.n_windows),
               "--eval-window", "512", "--selectors", "exact", "--fracs", "0.1",
               "--local-w", "32", "--score-dtype", dtype, "--diagnose-scores",
               "--output-json", path]
        if cap is not None:
            cmd += ["--score-key-clip", str(cap)]
        subprocess.run(cmd, check=True)
        result = json.load(open(path))
        records.append({"score_dtype": dtype, "score_key_clip": cap, "raw_file": path,
                        "ppl": result["grid"].get("exact_k0.1"),
                        "losses": result["losses"].get("exact_k0.1"),
                        "score_diagnostics": result.get("score_diagnostics")})
        with open(a.output_json, "w") as f:
            json.dump({"schema": "numerical_intervention_v1", "claim_scope":
                       "selection-only key-channel clipping; exact post-selection attention fixed",
                       "model": a.model, "ctx": a.ctx, "n_windows": a.n_windows,
                       "selector": "exact", "frac": 0.1, "local_w": 32,
                       "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
