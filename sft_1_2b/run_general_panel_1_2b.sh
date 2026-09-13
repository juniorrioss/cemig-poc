#!/bin/bash
# run_general_panel_1_2b.sh — Avaliacao GERAL + COMPORTAMENTAL (catastrophic forgetting) do 1.2B.
#
# Roda o painel de capacidade geral + comportamental (finetune2/eval_panel.py, layers
# general+behavioral, conjuntos CONGELADOS ood_general.json/ood_behavioral.json) no 1.2B
# treinado ESCOLHIDO vs o BASE 1.2B (QAD-Q4_0, o embarcado), p/ detectar perda FORA do dominio.
#
# Uso: ./run_general_panel_1_2b.sh [gguf_escolhido_na_Spark] [variant_label]
#   default = melhor variante Q4 do 1.2B (ajuste apos a consolidacao).
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

GGUF="${1:-/home/walcyrios/cemig-poc/models/lfm2.5-1.2b-sft_1_2b_r32-Q4_0.gguf}"
VARIANT="${2:-1_2b_r32_q4}"
PORT="${3:-8494}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FT2="$HERE/../finetune2"
PY="$HERE/../classifier/.venv/bin/python"
JUDGE="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
SPARK="walcyrios@spark-b431"

run_one () {  # $1=gguf $2=variant
  local G="$1"; local V="$2"
  echo "[panel_1.2b] $V <- $G"
  ssh "$SPARK" "nohup ~/cemig-poc/scripts/start_srv.sh '$G' $PORT 8192 4 > ~/cemig-poc/logs/launch_${PORT}.log 2>&1 < /dev/null; sleep 2"
  for i in $(seq 1 40); do
    if ssh "$SPARK" "timeout 4 curl -s http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "  servidor pronto"; break
    fi; sleep 4
  done
  local TUN="tun_${PORT}"
  tmux kill-session -t "$TUN" 2>/dev/null || true
  tmux new-session -d -s "$TUN" "ssh -N -L ${PORT}:127.0.0.1:${PORT} $SPARK"
  sleep 5
  curl -s "http://127.0.0.1:${PORT}/health" | grep -q ok && echo "  tunel ok"
  REGUA_JUDGE_URL="$JUDGE" "$PY" "$FT2/eval_panel.py" --variant "$V" \
    --server-url "http://127.0.0.1:${PORT}" --vllm-url "$JUDGE" \
    --layers general,behavioral --workers 8
  ssh "$SPARK" "pkill -f 'port $PORT' 2>/dev/null || true"
  tmux kill-session -t "$TUN" 2>/dev/null || true
}

# 1. modelo treinado escolhido
run_one "$GGUF" "$VARIANT"
# 2. base 1.2B QAD (referencia p/ o delta de regressao — o embarcado)
run_one "/home/walcyrios/cemig-poc/models/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" "base_1_2b_qad"

echo "[panel_1.2b] concluido. Consolidacao:"
"$PY" "$FT2/eval_panel.py" --consolidate --variants "$VARIANT,base_1_2b_qad" || true
echo "GENERAL_PANEL_1_2B_DONE"
