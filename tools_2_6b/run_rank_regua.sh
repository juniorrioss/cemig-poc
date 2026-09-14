#!/bin/bash
# régua (oráculo+híbrido+recusa) para um rank em Q4, 3 runs. Pure omitido (já sabemos que perde).
set -uo pipefail
SPARK=walcyrios@spark-b431; SPARK_IP=100.79.169.101
GGUF="$1"; LABEL="$2"; PORT="${3:-8502}"
PY=../classifier/.venv/bin/python
URL="http://$SPARK_IP:$PORT"
export REGUA_JUDGE_URL="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
ssh "$SPARK" "bash ~/cemig-poc/scripts/start_srv_net.sh $GGUF $PORT 16384 4"
trap "ssh $SPARK \"pkill -f 'port $PORT'\" 2>/dev/null || true" EXIT
for i in $(seq 1 80); do curl -s -m 3 "$URL/health" 2>/dev/null | grep -q ok && break; sleep 3; done
for R in 1 2 3; do
  $PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_oracle_q4_run${R} --mode oracle --pack 151 --out data/gen_${LABEL}_oracle_q4_run${R}.json --workers 6
  $PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_hibrido_q4_run${R} --mode hibrido --pack 151 --out data/gen_${LABEL}_hibrido_q4_run${R}.json --workers 6
done
$PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_refusal_q4 --mode oracle --pack refusal --out data/gen_${LABEL}_refusal_q4.json --workers 6
echo "RANK REGUA $LABEL DONE"
