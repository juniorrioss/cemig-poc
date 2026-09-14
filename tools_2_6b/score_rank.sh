#!/bin/bash
set -uo pipefail
LABEL="$1"
export REGUA_JUDGE_URL="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
so(){ local b="$1"; [ -f "data/gen_${b}.json" ] || return; (cd ../sft_v2 && REGUA_JUDGE_URL="$REGUA_JUDGE_URL" ../classifier/.venv/bin/python score_v2.py --resp "../tools_2_6b/data/gen_${b}.json" --out "../tools_2_6b/data/score_${b}.json" --workers 8); }
for R in 1 2 3; do so ${LABEL}_oracle_q4_run${R}; so ${LABEL}_hibrido_q4_run${R}; done
so ${LABEL}_refusal_q4
../classifier/.venv/bin/python ../sft_v3/refusal_metrics.py --scores data/score_${LABEL}_refusal_q4.json:${LABEL}_q4 --out data/refusal_confusion_${LABEL}_q4.json
echo "SCORE RANK $LABEL DONE"
