#!/bin/bash
# run_panel_2_6b.sh <gguf_spark> <variant> — painel general+behavioral (catastrophic forgetting).
# Server jinja+reasoning-budget 0 bind 0.0.0.0 na Spark; eval_panel roda DESTE host via tailscale.
set -uo pipefail
SPARK=walcyrios@spark-b431; SPARK_IP=100.79.169.101
GGUF="$1"; VARIANT="$2"; PORT="${3:-8503}"
PY=../classifier/.venv/bin/python
FT2=../finetune2
JUDGE="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
URL="http://$SPARK_IP:$PORT"
ssh "$SPARK" "bash ~/cemig-poc/scripts/start_srv_net_jinja.sh $GGUF $PORT 8192 4"
trap "ssh $SPARK \"pkill -f 'port $PORT'\" 2>/dev/null || true" EXIT
for i in $(seq 1 80); do curl -s -m 3 "$URL/health" 2>/dev/null | grep -q ok && break; sleep 3; done
REGUA_JUDGE_URL="$JUDGE" $PY $FT2/eval_panel.py --variant "$VARIANT" \
  --server-url "$URL" --vllm-url "$JUDGE" --layers general,behavioral --workers 8
echo "PANEL $VARIANT DONE"
