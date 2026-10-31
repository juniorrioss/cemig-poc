#!/bin/bash
# run_eval_f3b.sh — avaliação RÁPIDA padronizada do sweep F3b (roda na Spark
# ou na H100): ranking + distrator APENAS no recorte das 6 normas do F2
# (10,18,34 tratadas + 13,31,33 controle = 344 queries/célula), via gen_f2.py
# da F2 INALTERADO (--norms). O campeão final roda a bateria completa 1536
# nas 4 células com o run_eval_grid_f3.sh da F3.
#
# Uso: bash run_eval_f3b.sh <adapter_dir> <tag> [cells]
# Env: EVAL_DIR (default ~/cemig-poc/lfm3_f2 na Spark; /workspace/cemig/eval na H100)
set -euo pipefail

ADAPTER="$1"
TAG="$2"
CELLS="${3:-ranking distrator}"
EVAL_DIR="${EVAL_DIR:-$HOME/cemig-poc/lfm3_f2}"
BATCH="${BATCH:-48}"
NORMS="10,18,34,13,31,33"

cd "$EVAL_DIR"
for cell in $CELLS; do
  echo "=== $TAG $cell (recorte 6 NRs) ==="
  python gen_f2.py --cell "$cell" --model LiquidAI/LFM3-1.7B \
    --adapter "$ADAPTER" --tag "$TAG" --batch "$BATCH" --max-new 3072 \
    --norms "$NORMS"
done
echo "DONE $TAG"
