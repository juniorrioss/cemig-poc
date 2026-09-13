#!/bin/bash
# run_general_panel.sh — PARTE 5: avaliacao GERAL + COMPORTAMENTAL (catastrophic forgetting).
#
# O capitao acha que PEFT protege (provavelmente certo), mas a avaliacao segue pendente. Rodamos
# o painel de capacidade geral + comportamental que JA existe (finetune2/eval_panel.py, layers
# general+behavioral, conjuntos CONGELADOS ood_general.json/ood_behavioral.json) no modelo v3
# ESCOLHIDO vs o BASE 2.6B, para detectar perda FORA do dominio.
#
# NAO usa retrieval (general/behavioral sao prompts diretos). Juiz vLLM 27B via REGUA_JUDGE_URL.
# Sobe UM llama-server por vez na Spark (licao OOM). Reusa integralmente o eval_panel.py.
#
# Uso: ./run_general_panel.sh <gguf_escolhido_na_Spark> <variant_label>
#   ex.: ./run_general_panel.sh ~/cemig-poc/models/lfm2.5-2.6b-sft_v3_r64-bf16.gguf v3_r64_bf16
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

GGUF="$1"; VARIANT="$2"; PORT="${3:-8492}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FT2="$HERE/../finetune2"
PY="$HERE/../classifier/.venv/bin/python"
JUDGE="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
SPARK="walcyrios@spark-b431"

run_one () {  # $1=gguf $2=variant
  local G="$1"; local V="$2"
  echo "[panel] $V <- $G"
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

# 1. modelo escolhido
run_one "$GGUF" "$VARIANT"
# 2. base 2.6B (referencia p/ o delta de regressao)
run_one "/home/walcyrios/cemig-poc/models/LFM2.5-2.6B-Q4_0.gguf" "base_2p6b"

echo "[panel] concluido. Consolidacao:"
"$PY" "$FT2/eval_panel.py" --consolidate --variants "$VARIANT,base_2p6b" || true
echo "GENERAL_PANEL_DONE"
