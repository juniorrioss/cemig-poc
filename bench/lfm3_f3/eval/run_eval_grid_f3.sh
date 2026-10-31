#!/bin/bash
# run_eval_grid_f3.sh — grade de geração F3 (roda na Spark OU na H100).
# Reusa gen_f2.py da F2 INALTERADO (batch 48, max_new 3072, bateria COMPLETA
# 1536 queries/célula p/ comparabilidade direta com o F0 — ordem do brief).
#
# Uso: bash run_eval_grid_f3.sh <adapter_dir> <tag> [cells]
#   ex.: bash run_eval_grid_f3.sh runs/lfm3-f3-robustez-r32/checkpoint-XXX f3-r32-ep2 "ranking distrator"
# Env: EVAL_DIR (default ~/cemig-poc/lfm3_f2 na Spark; /workspace/cemig/eval na H100)
set -euo pipefail

ADAPTER="$1"
TAG="$2"
CELLS="${3:-oraculo ranking closed_book distrator}"
EVAL_DIR="${EVAL_DIR:-$HOME/cemig-poc/lfm3_f2}"
BATCH="${BATCH:-48}"

cd "$EVAL_DIR"
for cell in $CELLS; do
  echo "=== $TAG $cell ==="
  python gen_f2.py --cell "$cell" --model LiquidAI/LFM3-1.7B \
    --adapter "$ADAPTER" --tag "$TAG" --batch "$BATCH" --max-new 3072
done
echo "DONE $TAG"
