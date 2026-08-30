"""Fail-loud validation for publication evidence, including all manuscript axes."""
import argparse
import glob
import json
import math
import os


REQUIRED = {
    "study_a2_qwen7b_8192.json", "niah_hard_qwen0.5b_8192.json",
    "ablation_value_cache.json", "compression_comparison.json", "sparse_capacity.json",
    "access_pattern.json", "kivi_latency.json", "sparse_crossover.json",
    "study_profiling.json", "study_a2_pareto.json",
    "latency_robustness.json", "score_shape_sweep.json", "niah_hard_qwen7b_8192.json",
    "unpack_ablation.json", "topk_nsys.json", "numerical_intervention.json",
}
STUDIES = {"study_a2_qwen0.5b_8192.json", "study_a2_qwen0.5b_16384.json",
           "study_a2_qwen0.5b_32768.json", "study_a2_qwen1.5b_16384.json",
           "study_a2_qwen3b_16384.json", "study_a2_llama3b_16384.json",
           "study_a2_qwen0.5b_16384_pg19.json"}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def need(condition, message, errors):
    if not condition:
        errors.append(message)


def longctx(path, data, errors):
    for key in ("model", "model_revision", "dataset_fingerprint", "ctx", "n_windows",
                "score_dtype", "attend_mode", "grid", "losses", "confidence_intervals"):
        need(data.get(key) not in (None, "", {}), f"{path}: missing {key}", errors)
    need(data.get("score_dtype") == "fp32", f"{path}: selection scoring must be fp32", errors)
    for key, value in data.get("grid", {}).items():
        need(finite(value), f"{path}: non-finite {key}", errors)
        need(len(data.get("losses", {}).get(key, [])) == data.get("n_windows"),
             f"{path}: {key} lacks per-window losses", errors)
        need(key in data.get("confidence_intervals", {}), f"{path}: {key} lacks CI", errors)


def niah(path, data, errors):
    need(data.get("n_cases") == 100, f"{path}: requires 100 NIAH cases", errors)
    need(len(data.get("trial_records", [])) == 100, f"{path}: incomplete trial records", errors)
    need(finite(data.get("dense_acc")), f"{path}: missing dense accuracy", errors)
    need(bool(data.get("paired_difference_ci95")), f"{path}: no paired CIs", errors)
    for key, value in data.get("grid", {}).items():
        need(finite(value), f"{path}: invalid {key}", errors)
        need(key in data.get("confidence_intervals", {}), f"{path}: no CI for {key}", errors)
    if os.path.basename(path) == "niah_hard_qwen7b_8192.json":
        need(data.get("ctx") == 8192 and data.get("local_w") == 8,
             f"{path}: wrong 7B NIAH context/window", errors)
        need(data.get("selectors") == ["exact", "approx", "int4", "quest"],
             f"{path}: wrong 7B selector panel", errors)
        need(data.get("fracs") == [0.01, 0.02],
             f"{path}: wrong 7B retained-key budgets", errors)


def pitfall(path, data, errors):
    need(data.get("diagnose_scores") is True, f"{path}: diagnostics disabled", errors)
    need(bool(data.get("score_diagnostics")), f"{path}: diagnostics missing", errors)
    values = data.get("losses", {}).get("exact_k0.1", [])
    need(len(values) == 2, f"{path}: lacks two raw losses", errors)
    ppl = data.get("grid", {}).get("exact_k0.1")
    if not finite(ppl):
        need(data["score_diagnostics"].get("nonfinite_scores", 0) > 0,
             f"{path}: non-finite PPL lacks non-finite score evidence", errors)
    else:
        need(finite(data.get("grid", {}).get("exact_k0.1")), f"{path}: invalid PPL", errors)


def compression(path, data, errors):
    if "seed" in os.path.basename(path):
        need(data.get("schema") == "compression_seed_v1", f"{path}: wrong schema", errors)
        need(finite(data.get("fp16_ppl")), f"{path}: invalid dense PPL", errors)
        need(bool(data.get("model_revision")) and bool(data.get("dataset_fingerprint")),
             f"{path}: missing provenance", errors)
    else:
        need(data.get("seeds") == list(range(5)), f"{path}: expected seeds 0..4", errors)
        need(len(data.get("dense_reference", {}).get("observations", [])) == 5,
             f"{path}: missing dense observations", errors)
        groups = [data.get("methods", {})] if "methods" in data else list(data.get("value_modes", {}).values())
        for group in groups:
            for method, stat in group.items():
                need(len(stat.get("observations", [])) == 5,
                     f"{path}: {method} lacks five observations", errors)


def captured_latency(path, data, errors, shape=False):
    expected_schema = "score_shape_sweep_v1" if shape else "latency_robustness_v1"
    need(data.get("schema") == expected_schema, f"{path}: wrong schema", errors)
    need(bool(data.get("gpu")), f"{path}: missing GPU provenance", errors)
    rows = data.get("rows", [])
    need(len(rows) == 6, f"{path}: expected 6 sweep cells", errors)
    expected = {"fp8", "int4", "int2"}
    for row in rows:
        need(row.get("captures") == 15, f"{path}: expected 15 captures", errors)
        methods = row.get("latency_ms", {})
        need(expected <= set(methods), f"{path}: missing score methods", errors)
        for method in expected:
            values = methods.get(method, [])
            need(len(values) == 15 and all(finite(v) for v in values),
                 f"{path}: {method} lacks 15 finite capture values", errors)
        if shape:
            need(row.get("frac") == 0.05 and row.get("D") in {64, 128, 256}
                 and row.get("S") in {8192, 16384},
                 f"{path}: invalid shape-sweep row", errors)


def unpack_ablation(path, data, errors):
    need(data.get("schema") == "unpack_ablation_v1", f"{path}: wrong schema", errors)
    rows = data.get("rows", [])
    need(len(rows) == 6, f"{path}: expected six S/D cells", errors)
    required = {"fp8_dot"}
    for bits in ("int4", "int2"):
        required |= {f"{bits}_{stage}" for stage in
                     ("packed_load_reduce", "unpack_reduce", "dequant_reduce", "full_dot")}
    for row in rows:
        need(row.get("captures") == 15, f"{path}: expected 15 captures", errors)
        vals = row.get("latency_ms", {})
        need(required <= set(vals), f"{path}: missing reconstruction stages", errors)
        for key in required:
            x = vals.get(key, [])
            need(len(x) == 15 and all(finite(v) for v in x),
                 f"{path}: {key} lacks 15 finite captures", errors)


def topk_nsys(path, data, errors):
    need(data.get("schema") == "topk_nsys_v1", f"{path}: wrong schema", errors)
    rows = data.get("rows", [])
    need(len(rows) == 9, f"{path}: expected nine S/k trace cells", errors)
    for row in rows:
        text = "\n".join(row.get("named_kernel_lines", []))
        need("mbtopk::computeBlockDigitCounts" in text,
             f"{path}: missing measured top-k digit-count kernel", errors)


def numerical_intervention(path, data, errors):
    need(data.get("schema") == "numerical_intervention_v1", f"{path}: wrong schema", errors)
    records = data.get("records", [])
    need(len(records) == 5, f"{path}: expected fp32/bf16 baseline plus three caps", errors)
    for row in records:
        need(row.get("score_dtype") in {"fp32", "bf16"}, f"{path}: invalid dtype", errors)
        need(len(row.get("losses", [])) == 2, f"{path}: requires two raw losses", errors)
        need(finite(row.get("ppl")), f"{path}: invalid intervention PPL", errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/paper")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    errors = []
    paths = sorted(glob.glob(os.path.join(args.root, "*.json")))
    names = {os.path.basename(path) for path in paths}
    if not args.allow_incomplete:
        for name in sorted(REQUIRED | STUDIES):
            need(name in names, f"missing publication evidence {name}", errors)
        for model in ("qwen0.5b", "qwen1.5b", "qwen3b"):
            for dtype in ("bf16", "fp16", "fp32"):
                need(f"pitfall_{model}_{dtype}.json" in names, f"missing pitfall {model}/{dtype}", errors)
        for mode in ("fp8", "pq"):
            for seed in range(5):
                need(f"compression_{mode}_seed{seed}.json" in names,
                     f"missing compression {mode} seed {seed}", errors)
    for path in paths:
        name = os.path.basename(path)
        if name == "manifest.json" or name.startswith("oom_"):
            continue
        try:
            data = json.load(open(path))
        except Exception as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if name.startswith("study_a2_") and name != "study_a2_pareto.json":
            longctx(path, data, errors)
        elif name.startswith("niah_"):
            niah(path, data, errors)
        elif name.startswith("pitfall_"):
            pitfall(path, data, errors)
        elif name.startswith("compression_") or name == "ablation_value_cache.json":
            compression(path, data, errors)
        elif name == "access_pattern.json":
            rows = data.get("rows", [])
            need(len(rows) >= 21, f"{path}: too few raw rows", errors)
            patterns = {row.get("pattern") for row in rows}
            need("sorted_random" in patterns, f"{path}: missing sorted-random control", errors)
        elif name == "sparse_crossover.json":
            need(len(data) == 25, f"{path}: expected 25 cells", errors)
        elif name == "study_profiling.json":
            need(bool(data.get("P1_P2_bandwidth")) and bool(data.get("P3_decomposition")),
                 f"{path}: incomplete consolidation", errors)
        elif name == "latency_robustness.json":
            captured_latency(path, data, errors)
        elif name == "score_shape_sweep.json":
            captured_latency(path, data, errors, shape=True)
        elif name == "unpack_ablation.json":
            unpack_ablation(path, data, errors)
        elif name == "topk_nsys.json":
            topk_nsys(path, data, errors)
        elif name == "numerical_intervention.json":
            numerical_intervention(path, data, errors)
    if errors:
        raise SystemExit("evidence validation failed:\n- " + "\n- ".join(errors))
    print(f"validated {len(paths)} evidence files in {args.root}")


if __name__ == "__main__":
    main()
