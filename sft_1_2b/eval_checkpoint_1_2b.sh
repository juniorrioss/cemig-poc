#!/bin/bash
# eval_checkpoint_1_2b.sh — Avalia um checkpoint GGUF do 1.2B (na Spark) nas 4 suites.
#
# Reusa integralmente a maquinaria do v2/v3 (generate_v2.py, score_v2.py, common_v2.py e os
# packs oracle/retrieved/val/refusal de sft_v2/data). So os artefatos de saida (gen_/score_)
# vao para sft_1_2b/data. Regua honesta IDENTICA ao v3 (comparacao maca-com-maca).
#
# Uso: ./eval_checkpoint_1_2b.sh <gguf_ABSOLUTO_na_Spark> <label> <port> [system]
#   system default = v2 (prompt que permite recusa util; use 'numeros' p/ base/27b).
# Higiene: nao-interativo; server detached; tunel em tmux; checkpoint por item.
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

GGUF="$1"; LABEL="$2"; PORT="${3:-8483}"; SYSTEM="${4:-v2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
V2="$HERE/../sft_v2"
PY="$HERE/../classifier/.venv/bin/python"
JUDGE="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
SPARK="walcyrios@spark-b431"

echo "[eval_1.2b] $LABEL <- $GGUF (porta $PORT, system=$SYSTEM)"

ssh "$SPARK" "nohup ~/cemig-poc/scripts/start_srv.sh '$GGUF' $PORT 8192 4 > ~/cemig-poc/logs/launch_${PORT}.log 2>&1 < /dev/null; sleep 2; cat ~/cemig-poc/logs/launch_${PORT}.log"
for i in $(seq 1 40); do
  if ssh "$SPARK" "timeout 4 curl -s http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "  servidor pronto"; break
  fi
  sleep 4
done

TUN="tun_${PORT}"
tmux kill-session -t "$TUN" 2>/dev/null || true
tmux new-session -d -s "$TUN" "ssh -N -L ${PORT}:127.0.0.1:${PORT} $SPARK"
sleep 5
curl -s "http://127.0.0.1:${PORT}/health" | grep -q ok && echo "  tunel ok"

URL="http://127.0.0.1:${PORT}/v1"

gen_score () {  # $1=pack $2=suffix $3=system
  local PACK="$V2/data/$1"
  local SUF="$2"; local SYS="$3"
  local GEN="$HERE/data/gen_${LABEL}_${SUF}.json"
  "$PY" "$V2/generate_v2.py" --pack "$PACK" --url "$URL" --model "" \
    --label "${LABEL}_${SUF}" --system "$SYS" --out "$GEN" --workers 4 --max-tokens 512
  REGUA_JUDGE_URL="$JUDGE" "$PY" "$V2/score_v2.py" --resp "$GEN" \
    --out "$HERE/data/score_${LABEL}_${SUF}.json" --workers 10 | \
    grep -E "approval_pct|hallucination_pct|mean_coverage|recusa_correta|recusa_indevida" || true
}

gen_score oracle_pack_151.json    oracle151 "$SYSTEM"
gen_score retrieved_pack_151.json retr151   "$SYSTEM"
gen_score val_pack.json           val       "$SYSTEM"
gen_score refusal_test_pack.json  refusal   "$SYSTEM"

ssh "$SPARK" "pkill -f 'port $PORT' 2>/dev/null || true"
tmux kill-session -t "$TUN" 2>/dev/null || true
echo "[eval_1.2b] $LABEL concluido."
