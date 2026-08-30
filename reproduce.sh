#!/usr/bin/env bash
# End-to-end reproduction of the publication artifact.
#
# Prerequisites:
#   - the Python environment described by requirements.txt;
#   - CUDA GPU access (the published measurements use an RTX 4000 Ada);
#   - Hugging Face authentication for model downloads/cache access.
#
# Default behavior is resumable: completed JSON cells are reused. To rerun every
# publication result from scratch, use REPRO_CLEAN=1 bash reproduce.sh. This removes
# only results/paper, never source files, model caches, or external dependencies.
set -euo pipefail

cd "$(dirname "$0")"

CUDA_LIB="/home/202311016/.conda/envs/saif/lib/python3.12/site-packages/nvidia/cu13/lib"
SAIF_BIN="/home/202311016/.conda/envs/saif/bin"
if [[ -x "$SAIF_BIN/python" ]]; then
  export PATH="$SAIF_BIN:$PATH"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == "python" && -x "$SAIF_BIN/python" ]]; then
  PYTHON_BIN="$SAIF_BIN/python"
fi
python() { "$PYTHON_BIN" "$@"; }
if [[ -d "$CUDA_LIB" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}"
fi
if [[ -x /home/202311016/.conda/envs/saif/bin/x86_64-conda-linux-gnu-gcc ]]; then
  export CC=/home/202311016/.conda/envs/saif/bin/x86_64-conda-linux-gnu-gcc
fi

if [[ "${REPRO_CLEAN:-0}" == "1" ]]; then
  rm -rf results/paper
fi
mkdir -p results/paper third_party

# The tracked acmart.cls and ACM-Reference-Format.bst are the official ACM
# distribution used by the ASPLOS submission source.
if [[ ! -f acmart.cls || ! -f ACM-Reference-Format.bst ]]; then
  echo "missing ACM LaTeX assets: acmart.cls and ACM-Reference-Format.bst" >&2
  exit 1
fi

# Official KIVI CUDA kernel: pinned revision is recorded in the final manifest.
export KIVI_REPO="$PWD/third_party/KIVI"
if [[ ! -d "$KIVI_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/jy-yuan/KIVI.git "$KIVI_REPO"
fi
python -m pip install --no-build-isolation -e "$KIVI_REPO/quant"

# Fail before expensive runs if the numerical and allocation foundations are broken.
python -m benchmarks.test_statistics
python -m benchmarks.test_sparse_eval
python -m benchmarks.test_correctness

# Fresh systems evidence (all raw outputs are tracked under results/paper).
python -m benchmarks.bench_access_pattern --output-json results/paper/access_pattern.json
python -m benchmarks.bench_unpack_ablation --output-json results/paper/unpack_ablation.json
# Nsight Systems needs no privileged hardware-counter access.  It records the
# range-filtered CUDA kernel family for top-k; set NSYS_BIN if it is not on PATH.
python -m benchmarks.profile_topk_nsys --output-json results/paper/topk_nsys.json
python -m benchmarks.bench_sparse_capacity --output-json results/paper/sparse_capacity.json
python -m benchmarks.kivi_latency --output-json results/paper/kivi_latency.json
python -m benchmarks.bench_sparse_crossover --output-json results/paper/sparse_crossover.json
python -m benchmarks.bench_latency_robustness --output-json results/paper/latency_robustness.json
python -m benchmarks.bench_score_shape_sweep --output-json results/paper/score_shape_sweep.json
python -m benchmarks.consolidate_profiling

# Five independent seeds under each common value-cache condition. The aggregator
# retains every seed observation and produces the compression and value-cache tables.
for mode in fp8 pq; do
  for seed in 0 1 2 3 4; do
    out="results/paper/compression_${mode}_seed${seed}.json"
    if [[ -s "$out" && "${REPRO_CLEAN:-0}" != "1" ]]; then
      continue
    fi
    python -m benchmarks.eval_ppl --model Qwen/Qwen2.5-0.5B \
      --bits-k 8 --bits-v 4 --d-sub 2 --outliers 2 --max-samples 20 \
      --seed "$seed" --value-mode "$mode" --output-json "$out"
  done
done
python -m benchmarks.consolidate_compression

# Matched selection matrix, cross-corpus/composition controls, larger 7B scale
# extension, hard NIAH, and the nine-model/dtype F6 diagnostic sweep.
bash benchmarks/run_study_a2.sh
bash benchmarks/run_extensions.sh
MODEL="Qwen/Qwen2.5-7B"
SELECTORS="exact,fp8,int4,approx,signvq,quest"
COMMON=(--model "$MODEL" --n-windows 5 --local-w 32 --selectors "$SELECTORS" --fracs 0.01,0.05)
if [[ ! -s results/paper/study_a2_qwen7b_16384.json && ! -s results/paper/study_a2_qwen7b_8192.json ]]; then
  if ! python -m benchmarks.eval_longctx_sparse --ctx 16384 "${COMMON[@]}" \
      --output-json results/paper/study_a2_qwen7b_16384.json; then
    python -m benchmarks.eval_longctx_sparse --ctx 8192 "${COMMON[@]}" \
      --output-json results/paper/study_a2_qwen7b_8192.json
  fi
fi
if [[ ! -s results/paper/niah_hard_qwen0.5b_8192.json ]]; then
  python -m benchmarks.eval_niah_sparse --model Qwen/Qwen2.5-0.5B --ctx 8192 \
    --local-w 8 --selectors exact,approx,int4,quest --fracs 0.005,0.01,0.02 \
    --depths 0.1,0.3,0.6,0.9 --n-trials 25 \
    --output-json results/paper/niah_hard_qwen0.5b_8192.json
fi
# Always invoke the resumable 7B evaluation: it checkpoints the dense condition and
# each selector/budget cell, so a nonempty partial JSON must not be mistaken for
# complete evidence after an interrupted long run.
python -m benchmarks.eval_niah_sparse --model Qwen/Qwen2.5-7B --ctx 8192 \
  --local-w 8 --selectors exact,approx,int4,quest --fracs 0.01,0.02 \
  --depths 0.1,0.3,0.6,0.9 --n-trials 25 \
  --output-json results/paper/niah_hard_qwen7b_8192.json
bash benchmarks/run_pitfall.sh
# Causal check for the numerical-pitfall explanation: cap score keys only, while
# preserving the exact post-selection attention calculation.
python -m benchmarks.run_numerical_intervention --output-json results/paper/numerical_intervention.json

# Derived artifacts may only read tracked publication JSON. Validation is strict:
# every manuscript evidence class must be present and contain its raw observations.
python -m benchmarks.consolidate_study
python -m benchmarks.consolidate_profiling
python -m benchmarks.emit_tables
python -m benchmarks.generate_study_figures
python -m benchmarks.generate_readable_budget_figure
python -m benchmarks.validate_evidence

# The manifest is deliberately last so its hashes and repository state describe the
# final evidence tree. PDF build is optional only when a LaTeX engine is installed.
python -m benchmarks.capture_manifest --output results/paper/manifest.json
python -m benchmarks.validate_evidence
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile --reruns 3 main.tex
else
  echo "No LaTeX engine found; evidence reproduction completed, PDF build skipped." >&2
fi
