#!/usr/bin/env bash
# run_judge_v3.sh — Julga as 3 novas configs do Retrieval v3 no juiz vLLM 27B (mesmos 4
# eixos/critérios do judge-151 anterior) e gera a tabela 4-células. Idempotente: reusa o
# harness já empacotado. NÃO gera respostas (já feitas por run_v3.sh na 5070).
#
# Pré-requisito: o vLLM do capitão ONLINE em http://10.100.0.111:8005/v1 (juiz 27B).
# Uso: ./run_judge_v3.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/classifier/.venv/bin/python"
JUDGE_URL="${JUDGE_URL:-http://10.100.0.111:8005/v1}"

# Guard: aborta cedo se o juiz não responder (evita rodada pela metade).
if ! curl -s --max-time 8 "$JUDGE_URL/models" >/dev/null 2>&1; then
  echo "ERRO: juiz vLLM inacessível em $JUDGE_URL — abortando." >&2
  exit 1
fi

echo "=== (1) empacota harness v3 (4 configs v1_rigido p/ paridade de prompt) ==="
# Todas as 4 células com o prompt de produção v1_rigido: v3x{1.2B,2.6B} + antigo x{1.2B,2.6B}.
# (responses_lfm2.6b_noth_v1 = antigo x 2.6B thinking-OFF v1_rigido; já gerado na consolidação.)
"$PY" "$HERE/build_judge_input.py" \
  --configs v3_lfm1.2b v3_lfm2.6b_noth old_lfm1.2b_v1 lfm2.6b_noth_v1 \
  --out "$HERE/data/harness_v3.json"

echo "=== (2) juiz vLLM 27B (4 eixos) ==="
"$PY" "$ROOT/bench/judge.py" \
  --input "$HERE/data/harness_v3.json" \
  --output-eval "$HERE/data/judge_evaluations_v3.json" \
  --output-csv  "$HERE/data/judge_summary_v3.csv" --no-calibration

echo "=== (3) tabela 4-células + conversão chunk-certo->aprovada ==="
"$PY" "$HERE/analyze_v3.py"

echo "CONCLUÍDO. Ver bench/judge151/data/analysis_v3.json e README_V3_INTEGRATION.md."
