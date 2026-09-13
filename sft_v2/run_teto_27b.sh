#!/bin/bash
# run_teto_27b.sh — Teto do 27B nas 4 suites (via servidor do juiz, sem llama-server).
# system=numeros (o prompt acionável do teto; o 27B não precisa do prompt de recusa v2 —
# ele já recusa bem). Determinístico + thinking-off. Comentários PT-BR; código em inglês.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="$HERE/../classifier/.venv/bin/python"
JUDGE="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"
MODEL="Qwen/Qwen3.8-27B-FP8"

gen_score () {  # $1=pack $2=suffix
  local PACK="$HERE/data/$1"; local SUF="$2"
  local GEN="$HERE/data/gen_27b_${SUF}.json"
  "$PY" "$HERE/generate_v2.py" --pack "$PACK" --url "$JUDGE" --model "$MODEL" \
    --label "27b_${SUF}" --system numeros --out "$GEN" --workers 16 --max-tokens 512 \
    --deterministic --think-off
  REGUA_JUDGE_URL="$JUDGE" "$PY" "$HERE/score_v2.py" --resp "$GEN" \
    --out "$HERE/data/score_27b_${SUF}.json" --workers 12 | \
    grep -E "approval_pct|hallucination_pct|recusa_correta|recusa_indevida" || true
}

gen_score oracle_pack_151.json    oracle151
gen_score retrieved_pack_151.json retr151
gen_score val_pack.json           val
gen_score refusal_test_pack.json  refusal
echo "TETO_27B_DONE"
