#!/usr/bin/env bash
# Rigor-boost extensions to the matched A2 study (small/fast GPU jobs).
# Same fixed protocol as run_study_a2.sh: exact bf16 attend cache, local_w=32, fp32
# selection scoring, loss on the last 512 positions. Each invocation is independent
# (resumable, OOM-isolated). Big quantized models (W4) and NIAH (W7) have own scripts.
set -u
cd "$(dirname "$0")/.."
SELS="exact,approx,signvq,int4,int2,fp8,quest,h2o,snapkv"
LW=32
flt() { grep -v "libtinfo\|Loading weights\|Token indices"; }

# ---- W1: Llama-3.2-3B panel @16K (fixes the paper's internal inconsistency) ----
echo "########## W1 Llama-3.2-3B @16K ##########"
LLAMA="meta-llama/Llama-3.2-3B"
python -m benchmarks.eval_longctx_sparse --model "$LLAMA" --ctx 16384 \
  --n-windows 2 --local-w $LW --selectors "$SELS" \
  --output-json "results/paper/study_a2_llama3b_16384.json" 2>&1 | flt
for r in 8 16 32; do
  python -m benchmarks.eval_longctx_sparse --model "$LLAMA" --ctx 16384 \
    --n-windows 2 --local-w $LW --selectors sparq --sparq-r "$r" \
    --output-json "results/paper/study_a2_llama3b_16384_sparq${r}.json" 2>&1 | flt
done

# ---- W6: Quest page-size sweep (F4 fairness control), 0.5B @16K ----
echo "########## W6 Quest page sweep @16K ##########"
for pg in 16 32 64; do
  python -m benchmarks.eval_longctx_sparse --model Qwen/Qwen2.5-0.5B --ctx 16384 \
    --n-windows 3 --local-w $LW --selectors quest --page "$pg" \
    --output-json "results/paper/study_a2_qwen0.5b_16384_quest${pg}.json" 2>&1 | flt
done

# ---- W3: Composition 2x2 (selection {exact,fp8} x attend {bf16,fp8}), 0.5B @16K ----
echo "########## W3 Composition 2x2 @16K ##########"
for sel in exact fp8; do
  for att in bf16 fp8; do
    python -m benchmarks.eval_longctx_sparse --model Qwen/Qwen2.5-0.5B --ctx 16384 \
      --n-windows 3 --local-w $LW --selectors "$sel" --attend-dtype "$att" \
      --output-json "results/paper/compose_${sel}_${att}.json" 2>&1 | flt
  done
done

# ---- W5: PG19 second-corpus panel, 0.5B @16K (full roster; SparQ omitted, the
#         main roster already establishes corpus-invariance of the ranking) ----
echo "########## W5 PG19 panel @16K ##########"
python -m benchmarks.eval_longctx_sparse --model Qwen/Qwen2.5-0.5B --ctx 16384 \
  --n-windows 3 --local-w $LW --selectors "$SELS" --dataset pg19 \
  --output-json "results/paper/study_a2_qwen0.5b_16384_pg19.json" 2>&1 | flt

echo "########## EXTENSIONS DONE ##########"
