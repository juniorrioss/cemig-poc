#!/bin/bash
# eval_checkpoint.sh — Avalia um checkpoint GGUF (na Spark) em ORÁCULO e RETRIEVED com a régua.
#
# Sobe um llama-server na Spark para o GGUF, abre túnel deste host, gera respostas (oráculo +
# retrieved v4) e pontua pela régua honesta (juiz 27B). Reusa generate.py + score.py.
#
# Uso: ./eval_checkpoint.sh <gguf_remoto> <label> <port>
#   ex.: ./eval_checkpoint.sh ~/cemig-poc/models/lfm2.5-2.6b-sft_ep1-Q4_0.gguf sft_ep1_q4 8471
#
# Higiene: não-interativo; servidor detached na Spark; túnel em tmux; checkpoint por item.
# Comentários PT-BR; código em inglês.
set -euo pipefail

GGUF="$1"; LABEL="$2"; PORT="${3:-8471}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../classifier/.venv/bin/python"
JUDGE="http://10.100.0.111:8005/v1"
SPARK="walcyrios@spark-b431"

echo "[eval_checkpoint] $LABEL <- $GGUF (porta $PORT)"

# 1) sobe o servidor do checkpoint na Spark
ssh "$SPARK" "nohup ~/cemig-poc/scripts/start_srv.sh '$GGUF' $PORT 20480 4 > ~/cemig-poc/logs/launch_${PORT}.log 2>&1 < /dev/null; sleep 2; cat ~/cemig-poc/logs/launch_${PORT}.log"
# espera o modelo carregar
for i in $(seq 1 30); do
  if ssh "$SPARK" "timeout 4 curl -s http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "  servidor pronto"; break
  fi
  sleep 4
done

# 2) túnel deste host -> Spark
TUN="tun_${PORT}"
tmux kill-session -t "$TUN" 2>/dev/null || true
tmux new-session -d -s "$TUN" "ssh -N -L ${PORT}:127.0.0.1:${PORT} $SPARK"
sleep 5
curl -s "http://127.0.0.1:${PORT}/health" | grep -q ok && echo "  túnel ok"

# 3) gera oráculo + retrieved
for CTX in oracle retrieved; do
  PACK="$HERE/data/${CTX}_pack.json"
  OUT="$HERE/data/gen_${LABEL}_${CTX}.json"
  "$PY" "$HERE/generate.py" --pack "$PACK" --url "http://127.0.0.1:${PORT}/v1" --model "" \
    --label "$LABEL" --variant numeros --out "$OUT" --workers 4 --max-tokens 512
  REGUA_JUDGE_URL="$JUDGE" "$PY" "$HERE/score.py" --resp "$OUT" \
    --out "$HERE/data/score_${LABEL}_${CTX}.json" --workers 10 | \
    grep -E "approval_pct|hallucination_pct|mean_coverage|conv_chunk_hit"
done

# 4) derruba o servidor do checkpoint (libera VRAM)
ssh "$SPARK" "pkill -f 'port $PORT' 2>/dev/null || true"
tmux kill-session -t "$TUN" 2>/dev/null || true
echo "[eval_checkpoint] $LABEL concluído."
