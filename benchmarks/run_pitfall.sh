#!/usr/bin/env bash
# F6 score-precision pitfall: exact-oracle selection scored in bf16 vs fp32, across
# models of differing key-outlier magnitude. Run AFTER the main matrix (shares GPU).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/paper
LW=32; CTX=16384; NW=2
declare -A MID=( [qwen0.5b]="Qwen/Qwen2.5-0.5B" [qwen1.5b]="Qwen/Qwen2.5-1.5B" [qwen3b]="Qwen/Qwen2.5-3B" )
# Three score precisions per model: fp32 (true oracle), fp16 (10 mantissa bits --
# tests whether mantissa precision alone fixes the pitfall), bf16 (7 bits -- broken).
for fr in qwen0.5b qwen1.5b qwen3b; do
  for prec in fp32 fp16 bf16; do
    python -m benchmarks.eval_longctx_sparse --model "${MID[$fr]}" --ctx $CTX \
      --n-windows $NW --local-w $LW --selectors exact --fracs 0.10 \
      --score-dtype $prec --diagnose-scores \
      --output-json "results/paper/pitfall_${fr}_${prec}.json" 2>&1 \
      | grep -v "libtinfo\|Loading weights\|Token indices"
  done
done
echo "########## PITFALL DONE ##########"
