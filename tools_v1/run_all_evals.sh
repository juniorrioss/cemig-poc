#!/bin/bash
# run_all_evals.sh — avalia um GGUF candidato (na Spark) a partir DESTE host, via tailscale.
#
# Fluxo: sobe o llama-server do candidato na Spark bindando em 0.0.0.0 (acessível pela rede
# tailscale, mais estável que túnel SSH sob carga concorrente), roda run_eval.py
# (decisão/args/reuso) + score_e2e.py (aprovação/alucinação/recusa) no classifier/.venv DESTE
# host (que tem a régua e o índice v4), e derruba o servidor no fim. Juiz 27B roteável daqui.
#
# ctx 16384 / --parallel 4 => 4096 tokens/slot (o multiturno chega a ~3000). /completion cru.
#
# Uso: bash run_all_evals.sh <gguf_na_spark> <label> [port] [topk] [--deterministic]
# Comentários PT-BR; código em inglês.
set -uo pipefail   # NÃO -e: pkill/ssh retornam !=0 sem processo (benigno)
SPARK=walcyrios@spark-b431
SPARK_IP=100.79.169.101
GGUF="$1"; LABEL="$2"; PORT="${3:-8500}"; TOPK="${4:-2}"; DET="${5:-}"
PY=../classifier/.venv/bin/python
URL="http://$SPARK_IP:$PORT"

echo "== sobe llama-server $LABEL na Spark (0.0.0.0:$PORT, ctx 16384/4) =="
ssh "$SPARK" "pkill -f 'port $PORT'; sleep 2; cd ~/cemig-poc && setsid bash -c \"nohup ./llama.cpp/build-cuda/bin/llama-server -m '$GGUF' -ngl 99 -c 16384 --parallel 4 --host 0.0.0.0 --port $PORT > logs/server_${PORT}.log 2>&1 &\"; echo done"

cleanup() {
  echo "== derruba servidor =="
  ssh "$SPARK" "pkill -f 'port $PORT'" 2>/dev/null || true
}
trap cleanup EXIT

echo "== espera o servidor responder =="
ok=0
for i in $(seq 1 40); do
  if curl -s -m 3 "$URL/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
  sleep 2
done
[ "$ok" = 1 ] || { echo "servidor não respondeu"; exit 1; }

echo "== run_eval (decisão/args/reuso) =="
$PY run_eval.py --url "$URL" --label "$LABEL" --topk "$TOPK" --workers 4 $DET

echo "== score_e2e (aprovação/alucinação/recusa) =="
$PY score_e2e.py --url "$URL" --label "$LABEL" --topk "$TOPK" --workers 4 $DET

echo "AVALIAÇÃO $LABEL COMPLETA"
