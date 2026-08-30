#!/usr/bin/env bash
# W7: needle-in-a-haystack at long context across three models. Strengthens F4
# (per-key vs page-based retrieval) beyond the 8K/0.5B point. The sparse path is
# chunked (no dense [S,S] materialize), so >=16K is feasible on 20GB. If a model's
# DENSE-reference accuracy itself collapses at 32K (model capability, not the harness),
# that row is reported honestly at 16K only.
set -u
cd "$(dirname "$0")/.."
flt() { grep -v "libtinfo\|Loading weights\|Token indices"; }
SELS="exact,approx,int4,quest"
FRACS="0.01,0.02,0.05"
declare -A MID=( [qwen0.5b]="Qwen/Qwen2.5-0.5B" [qwen1.5b]="Qwen/Qwen2.5-1.5B" [qwen3b]="Qwen/Qwen2.5-3B" )

for fr in qwen0.5b qwen1.5b qwen3b; do
  for ctx in 16384 32768; do
    echo "########## NIAH $fr @ $ctx ##########"
    python -m benchmarks.eval_niah_sparse --model "${MID[$fr]}" --ctx $ctx \
      --selectors "$SELS" --fracs "$FRACS" --n-trials 6 \
      --output-json "results/niah_${fr}_${ctx}.json" 2>&1 | flt || \
      echo "########## NIAH $fr @ $ctx failed (OOM/capability) -- skipping ##########"
  done
done
echo "########## NIAH EXT DONE ##########"
