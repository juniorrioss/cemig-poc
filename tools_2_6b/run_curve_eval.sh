#!/bin/bash
# run_curve_eval.sh — só decisão/args/reuso de um checkpoint (curva de saturação / varredura rank).
# Sobe o GGUF na Spark e roda run_eval_2_6b (sem oráculo/e2e). Usa Q4 (barato, determinístico).
set -uo pipefail
SPARK=walcyrios@spark-b431; SPARK_IP=100.79.169.101
GGUF="$1"; LABEL="$2"; PORT="${3:-8501}"
PY=../classifier/.venv/bin/python
URL="http://$SPARK_IP:$PORT"
ssh "$SPARK" "bash ~/cemig-poc/scripts/start_srv_net.sh $GGUF $PORT 16384 4"
trap "ssh $SPARK \"pkill -f 'port $PORT'\" 2>/dev/null || true" EXIT
for i in $(seq 1 80); do curl -s -m 3 "$URL/health" 2>/dev/null | grep -q ok && break; sleep 3; done
$PY run_eval_2_6b.py --url "$URL" --label "$LABEL" --topk 2 --workers 4
echo "CURVE_EVAL $LABEL DONE"
