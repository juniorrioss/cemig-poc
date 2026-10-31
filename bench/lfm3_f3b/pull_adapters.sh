#!/bin/bash
# pull_adapters.sh — H100 -> local -> Spark (H100->Spark não roteia; rota da F3).
# Uso: bash pull_adapters.sh lfm3-f3b-recusa-35 [checkpoint-glob]
set -euo pipefail
RUN="$1"
GLOB="${2:-checkpoint-*}"
LOCAL=/tmp/f3b_adapters/$RUN
mkdir -p "$LOCAL"
rsync -az --include="$GLOB/***" --include="premises.json" --exclude="*" \
  "aluno_walcy@10.100.0.111:/raid/aluno_walcy/cemig_mobile/checkpoints/$RUN/" \
  "$LOCAL/"
# poda otimizadores (só o adapter interessa p/ avaliação)
find "$LOCAL" -name "optimizer.pt" -delete 2>/dev/null || true
ssh walcyrios@100.79.169.101 "mkdir -p ~/cemig-poc/lfm3_f3b/runs/$RUN"
rsync -az "$LOCAL/" "walcyrios@100.79.169.101:~/cemig-poc/lfm3_f3b/runs/$RUN/"
echo "[ok] $RUN na Spark: ~/cemig-poc/lfm3_f3b/runs/$RUN"
