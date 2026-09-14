#!/bin/bash
# score_gens_2_6b.sh — pontua as gerações (oráculo/híbrido/puro/recusa) pela régua honesta.
#
# Reusa sft_v2/score_v2.py (régua: cobertura de fatos + alucinação + tautologia, citação FORA
# do gate) e sft_v3/refusal_metrics.py (matriz de recusa). NÃO gera token novo — só julga o que
# harness_oraculo_2_6b já produziu. Juiz vLLM 27B via REGUA_JUDGE_URL (roteável deste host).
#
# Uso: bash score_gens_2_6b.sh <label> <quant> [runs]
#   ex.: bash score_gens_2_6b.sh tools_2_6b_r64 q4 3
# Comentários PT-BR; código em inglês.
set -uo pipefail
LABEL="$1"; QUANT="${2:-q4}"; RUNS="${3:-3}"
PY=../classifier/.venv/bin/python
export REGUA_JUDGE_URL="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"

score_one () {  # $1 = gen file basename (sem data/gen_ prefix, sem .json)
  local base="$1"
  local gen="data/gen_${base}.json"
  local out="data/score_${base}.json"
  [ -f "$gen" ] || { echo "  [skip] $gen ausente"; return; }
  (cd ../sft_v2 && REGUA_JUDGE_URL="$REGUA_JUDGE_URL" ../classifier/.venv/bin/python score_v2.py \
     --resp "../tools_2_6b/$gen" --out "../tools_2_6b/$out" --workers 8)
}

for R in $(seq 1 "$RUNS"); do
  echo "== score oráculo/híbrido/puro run$R =="
  score_one ${LABEL}_oracle_${QUANT}_run${R}
  score_one ${LABEL}_hibrido_${QUANT}_run${R}
  score_one ${LABEL}_pure_${QUANT}_run${R}
done

echo "== score recusa =="
score_one ${LABEL}_refusal_${QUANT}
$PY ../sft_v3/refusal_metrics.py \
   --scores data/score_${LABEL}_refusal_${QUANT}.json:${LABEL}_${QUANT} \
   --out data/refusal_confusion_${QUANT}.json

echo "SCORE $LABEL ($QUANT) COMPLETO"
