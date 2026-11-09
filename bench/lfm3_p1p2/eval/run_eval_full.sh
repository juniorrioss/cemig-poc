#!/bin/bash
# run_eval_full.sh — BATERIA COMPLETA (trava anti-overfit do capitão) no container da GPU 3:
#   4 células × 1536 (gen_f2.py, system F0 sem tools) + e2e op_d50 × 1536 (formato do APP)
#   + e2e op_d50 × 151 coloquiais da POC. Mesmo sampling da série (0,1/50/rep 1,1).
# Uso: bash run_eval_full.sh TAG ADAPTER_DIR_OR_EMPTY
set -u
TAG=$1
ADAPTER=${2:-}
export HF_HOME=/workspace/cemig/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/workspace/cemig/p1p2/f4-r32-ep2-merged
EVAL=/workspace/cemig/eval
CODE=/workspace/cemig/p1p2/code
AD=""
[ -n "$ADAPTER" ] && AD="--adapter $ADAPTER"
BS=${EVAL_BS:-24}
echo "[full] $TAG start $(date)"
for CELL in distrator ranking oraculo closed_book; do
  (cd $EVAL && python gen_f2.py --cell $CELL --model $BASE $AD --tag p2-$TAG --batch $BS --max-new 3072) 2>&1 | grep -E "^\[ok\]"
done
(cd $CODE/e2e && python gen_e2e.py --model $BASE $AD --tag p2-$TAG-1536 \
    --prompts $CODE/e2e/prompts_op_d50_1536.jsonl --batch $BS --max-new 768) 2>&1 | grep -E "^\[ok\]"
(cd $CODE/e2e && python gen_e2e.py --model $BASE $AD --tag p2-$TAG-151 \
    --prompts $CODE/e2e/prompts_op_d50_151.jsonl --batch $BS --max-new 768) 2>&1 | grep -E "^\[ok\]"
echo "[full] $TAG DONE $(date)"
