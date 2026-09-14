#!/bin/bash
# run_all_evals_2_6b.sh — bateria de avaliação do 2.6B tool-trained, IDÊNTICA à do 1.2B.
#
# Sobe o llama-server do candidato 2.6B na Spark (bind 0.0.0.0, acessível via tailscale) e roda
# DESTE host, no classifier/.venv, exatamente as mesmas medições do tools_v1/tools_oraculo — via
# os wrappers que injetam o renderer 2.6B (thinking-OFF). O 2.6B é reasoning: o llama-server é
# irrelevante aqui porque usamos /completion RAW com prompt já renderizado por NÓS (thinking-OFF).
#
# CRÍTICO (memória unificada): mata llama-server antes. --parallel 4 => 4096 tok/slot (o oráculo
# injeta ~1450 tok; --parallel 8 estoura 2048). ctx 16384.
#
# Uso: bash run_all_evals_2_6b.sh <gguf_na_spark> <label> <quant> [port] [runs]
#   ex.: bash run_all_evals_2_6b.sh ~/cemig-poc/models/lfm2.5-2.6b-tools_2_6b_r64-Q4_0.gguf tools_2_6b_r64 q4 8500 3
# Comentários PT-BR; código em inglês.
set -uo pipefail
SPARK=walcyrios@spark-b431
SPARK_IP=100.79.169.101
GGUF="$1"; LABEL="$2"; QUANT="${3:-q4}"; PORT="${4:-8500}"; RUNS="${5:-3}"
PY=../classifier/.venv/bin/python
URL="http://$SPARK_IP:$PORT"
export REGUA_JUDGE_URL="${REGUA_JUDGE_URL:-http://10.100.0.111:8005/v1}"

echo "== sobe llama-server $LABEL ($QUANT) na Spark (0.0.0.0:$PORT, ctx 16384/4) =="
ssh "$SPARK" "bash ~/cemig-poc/scripts/start_srv_net.sh $GGUF $PORT 16384 4"
cleanup() { echo "== derruba servidor =="; ssh "$SPARK" "pkill -f 'port $PORT'" 2>/dev/null || true; }
trap cleanup EXIT

echo "== espera o servidor =="
ok=0
for i in $(seq 1 80); do
  if curl -s -m 3 "$URL/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
  sleep 3
done
[ "$ok" = 1 ] || { echo "servidor não respondeu"; exit 1; }

# ---- decisão / argumentos / reuso (1 run — determinístico onde importa; args usa fala do modelo)
echo "== run_eval (decisão/args/reuso) =="
$PY run_eval_2_6b.py --url "$URL" --label "$LABEL" --topk 2 --workers 4

# ---- e2e retrieval real (aprovação/alucinação com busca do próprio modelo, 90 answerable)
echo "== score_e2e (aprovação/alucinação/conv) =="
$PY score_e2e_2_6b.py --url "$URL" --label "$LABEL" --topk 2 --workers 4

# ---- oráculo / híbrido / puro (151) — RUNS runs para média±desvio
for R in $(seq 1 "$RUNS"); do
  echo "== oráculo run$R =="
  $PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_oracle_${QUANT}_run${R} \
     --mode oracle --pack 151 --out data/gen_${LABEL}_oracle_${QUANT}_run${R}.json --workers 6
  echo "== híbrido run$R =="
  $PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_hibrido_${QUANT}_run${R} \
     --mode hibrido --pack 151 --out data/gen_${LABEL}_hibrido_${QUANT}_run${R}.json --workers 6
  echo "== puro run$R =="
  $PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_pure_${QUANT}_run${R} \
     --mode pure --pack 151 --out data/gen_${LABEL}_pure_${QUANT}_run${R}.json --workers 6
done

# ---- recusa (pack sft_v2, 85) — 1 run (determinístico)
echo "== recusa =="
$PY harness_oraculo_2_6b.py --url "$URL" --label ${LABEL}_refusal_${QUANT} \
   --mode oracle --pack refusal --out data/gen_${LABEL}_refusal_${QUANT}.json --workers 6

echo "AVALIAÇÃO $LABEL ($QUANT) COMPLETA"
