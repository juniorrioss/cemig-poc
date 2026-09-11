#!/usr/bin/env bash
# run_v3.sh — Gera respostas E2E das 151 com o RETRIEVAL v3 (fusão RRF 3-sinais) para as
# duas configs novas da PARTE 1 do brief, na RTX 5070 (llama-server CUDA):
#   v3_lfm1.2b       -> LFM2.5-1.2B-Instruct-QAD-Q4_0 (embarcado), prompt v1_rigido
#   v3_lfm2.6b_noth  -> LFM2.5-2.6B-Q4_0 thinking-OFF (--reasoning-budget 0), prompt v1_rigido
# Para paridade de prompt na tabela 4-células, também gera:
#   old_lfm1.2b_v1   -> retrieval ANTIGO × 1.2B × v1_rigido (a célula antiga com o prompt de produção)
# (old × 2.6B × v1_rigido já existe: data/responses_lfm2.6b_noth_v1.json)
#
# Uso: ./run_v3.sh [config ...]   default: as 3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA="$HOME/llama.cpp/build-cuda/bin"
MODELS="$HOME/models-poc"
PY="$ROOT/classifier/.venv/bin/python"
PORT="${JUDGE151_PORT:-8398}"
URL="http://127.0.0.1:$PORT"
export LD_LIBRARY_PATH="$LLAMA:${LD_LIBRARY_PATH:-}"

CONFIGS=("${@:-v3_lfm1.2b v3_lfm2.6b_noth old_lfm1.2b_v1}")
# shellcheck disable=SC2206
CONFIGS=(${CONFIGS[@]})

model_file_for() {
  case "$1" in
    v3_lfm1.2b)       echo "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" ;;
    v3_lfm2.6b_noth)  echo "LFM2.5-2.6B-Q4_0.gguf" ;;
    old_lfm1.2b_v1)   echo "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}
extra_args_for() { case "$1" in v3_lfm2.6b_noth) echo "--reasoning-budget 0" ;; *) echo "" ;; esac; }
reasoning_for()  { case "$1" in v3_lfm2.6b_noth) echo "off" ;; *) echo "on" ;; esac; }
retriever_for()  { case "$1" in old_lfm1.2b_v1) echo "main" ;; *) echo "v3" ;; esac; }

wait_health() {
  for _ in $(seq 1 90); do
    if curl -s --max-time 3 "$URL/health" | grep -q '"ok"'; then return 0; fi
    sleep 2
  done
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  mf="$(model_file_for "$cfg")"; extra="$(extra_args_for "$cfg")"
  reasoning="$(reasoning_for "$cfg")"; retr="$(retriever_for "$cfg")"
  out="$HERE/data/responses_${cfg}.json"; log="/tmp/judge151v3_srv_${cfg}.log"
  echo "=== [$cfg] modelo=$mf retriever=$retr reasoning=$reasoning extra='$extra' ==="
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "[$cfg] ERRO: porta $PORT já em uso; abortando p/ não contaminar."; exit 1
  fi
  # shellcheck disable=SC2086
  "$LLAMA/llama-server" -m "$MODELS/$mf" -ngl 99 -c 4096 --jinja \
    --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 &
  SRV_PID=$!
  trap 'kill $SRV_PID 2>/dev/null || true' EXIT
  if ! wait_health; then echo "[$cfg] health falhou:"; tail -20 "$log"; kill $SRV_PID 2>/dev/null||true; exit 1; fi
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "[$cfg] servidor morreu (colisão?)"; tail -20 "$log"; exit 1; fi
  echo "[$cfg] servidor pronto (pid $SRV_PID) porta $PORT."

  DB_ARG=(); [ "$retr" = "main" ] && DB_ARG=(--db "$ROOT/corpus/index_hf_36nr.db")
  "$PY" "$HERE/pipeline.py" \
    --config-key "$cfg" --model-file "$mf" --reasoning "$reasoning" \
    --retriever "$retr" --prompt-key v1_rigido \
    --url "$URL" --out "$out" --max-tokens 1024 --timeout 180 "${DB_ARG[@]}"

  kill $SRV_PID 2>/dev/null || true; wait $SRV_PID 2>/dev/null || true; trap - EXIT; sleep 3
  echo "[$cfg] concluído -> $out"
done
echo "TODAS AS CONFIGS v3 CONCLUÍDAS."
