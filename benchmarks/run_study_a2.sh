#!/usr/bin/env bash
# Matched-conditions A2 selection grid across models + contexts (RTX 4000 Ada).
# Protocol held FIXED for every cell: exact bf16 attend cache, local_w=32,
# eval loss on the last 512 positions. SparQ r-sweep goes to its own files so the
# distinct r values don't clobber the shared 'sparq' grid key.
#
# Resumable: eval_longctx_sparse merges into an existing grid JSON, so re-running
# skips nothing but overwrites cells in place. Each invocation is independent, so an
# OOM on one (model,ctx) does not abort the rest.
set -u
cd "$(dirname "$0")/.."
SELS="exact,approx,signvq,int4,int2,fp8,quest,h2o,snapkv"
LW=32
run() {  # model_id  friendly  ctx  nwin
  local mid="$1" fr="$2" ctx="$3" nw="$4"
  echo "########## $fr  ctx=$ctx  nwin=$nw ##########"
  python -m benchmarks.eval_longctx_sparse --model "$mid" --ctx "$ctx" \
    --n-windows "$nw" --local-w "$LW" --selectors "$SELS" \
    --output-json "results/paper/study_a2_${fr}_${ctx}.json" 2>&1 \
    | grep -v "libtinfo\|Loading weights\|Token indices"
  for r in 8 16 32; do
    python -m benchmarks.eval_longctx_sparse --model "$mid" --ctx "$ctx" \
      --n-windows "$nw" --local-w "$LW" --selectors sparq --sparq-r "$r" \
      --output-json "results/paper/study_a2_${fr}_${ctx}_sparq${r}.json" 2>&1 \
      | grep -v "libtinfo\|Loading weights\|Token indices"
  done
}

# headline 0.5B @16K (main grid; fp32 selection scoring)
run "Qwen/Qwen2.5-0.5B" qwen0.5b 16384 3
# cross-model point @16K (D=64 and D=128)
run "Qwen/Qwen2.5-1.5B" qwen1.5b 16384 2
run "Qwen/Qwen2.5-3B"   qwen3b   16384 2
# context sweep on the small model
run "Qwen/Qwen2.5-0.5B" qwen0.5b 8192  3
run "Qwen/Qwen2.5-0.5B" qwen0.5b 32768 2
echo "########## A2 MATRIX DONE ##########"
