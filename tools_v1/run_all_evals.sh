#!/bin/bash
# run_all_evals.sh — avalia um GGUF candidato (na Spark) a partir DESTE host, via túnel SSH.
#
# Fluxo: sobe o llama-server do candidato na Spark (start_srv), abre um túnel SSH local
# ->  127.0.0.1:$LPORT, roda run_eval.py (decisão/args/reuso) + score_e2e.py (aprovação/
# alucinação/recusa) no classifier/.venv DESTE host (que tem a régua e o índice v4), e derruba
# o servidor no fim. O juiz 27B (10.100.0.111:8005) é roteável daqui.
#
# Uso: bash run_all_evals.sh <gguf_na_spark> <label> [lport] [topk] [--deterministic]
#   ex.: bash run_all_evals.sh ~/cemig-poc/models/lfm2.5-1.2b-tools_1_2b_r32-Q4_0.gguf tools_r32
# Comentários PT-BR; código em inglês.
set -euo pipefail
SPARK=walcyrios@spark-b431
GGUF="$1"; LABEL="$2"; LPORT="${3:-8500}"; TOPK="${4:-2}"; DET="${5:-}"
RPORT=$((LPORT))
PY=../classifier/.venv/bin/python

echo "== sobe llama-server do candidato na Spark (porta $RPORT) =="
ssh "$SPARK" "bash ~/cemig-poc/scripts/start_srv.sh '$GGUF' $RPORT 8192 8" 
sleep 12

echo "== túnel SSH local $LPORT -> Spark 127.0.0.1:$RPORT =="
ssh -f -N -L "$LPORT:127.0.0.1:$RPORT" "$SPARK"
TUNNEL_PID=$(pgrep -f "ssh -f -N -L $LPORT:127.0.0.1:$RPORT" | head -1 || true)
sleep 3

cleanup() {
  echo "== derruba túnel + servidor =="
  [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  ssh "$SPARK" "pkill -f 'port $RPORT'" 2>/dev/null || true
}
trap cleanup EXIT

# espera o servidor responder
for i in $(seq 1 30); do
  if curl -s -m 3 "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "== run_eval (decisão/args/reuso) =="
$PY run_eval.py --url "http://127.0.0.1:$LPORT" --label "$LABEL" --topk "$TOPK" --workers 6 $DET

echo "== score_e2e (aprovação/alucinação/recusa) =="
$PY score_e2e.py --url "http://127.0.0.1:$LPORT" --label "$LABEL" --topk "$TOPK" --workers 6 $DET

echo "AVALIAÇÃO $LABEL COMPLETA"
