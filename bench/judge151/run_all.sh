#!/usr/bin/env bash
# run_all.sh — Orquestra as 3 configs de sintetizador na RTX 5070 (llama-server CUDA) e
# gera as respostas E2E para as 151 perguntas. Cada config sobe o servidor, aguarda health,
# roda o pipeline (checkpoint incremental) e derruba o servidor antes da próxima.
#
# Uso: ./run_all.sh [config ...]   (default: as 3)  |  configs: lfm1.2b lfm2.6b lfm2.6b_noth
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA="$HOME/llama.cpp/build-cuda/bin"
MODELS="$HOME/models-poc"
PY="$ROOT/classifier/.venv/bin/python"
# Porta dedicada e ALTA para não colidir com servidores de outras lanes (ex.: 8090 é
# ocupado por outro processo do host). O guard abaixo aborta se a porta não estiver livre.
PORT="${JUDGE151_PORT:-8397}"
URL="http://127.0.0.1:$PORT"
export LD_LIBRARY_PATH="$LLAMA:${LD_LIBRARY_PATH:-}"

CONFIGS=("${@:-lfm1.2b lfm2.6b lfm2.6b_noth}")
# shellcheck disable=SC2206
CONFIGS=(${CONFIGS[@]})

model_file_for() {
  case "$1" in
    lfm1.2b)       echo "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" ;;
    lfm2.6b)       echo "LFM2.5-2.6B-Q4_0.gguf" ;;
    lfm2.6b_noth)  echo "LFM2.5-2.6B-Q4_0.gguf" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}

extra_args_for() {
  # reasoning OFF via --reasoning-budget 0 (não há flag enable_thinking no template LFM2.5)
  case "$1" in
    lfm2.6b_noth) echo "--reasoning-budget 0" ;;
    *) echo "" ;;
  esac
}

reasoning_for() {
  case "$1" in
    lfm2.6b_noth) echo "off" ;;
    *) echo "on" ;;
  esac
}

wait_health() {
  for _ in $(seq 1 60); do
    if curl -s --max-time 3 "$URL/health" | grep -q '"ok"'; then return 0; fi
    sleep 2
  done
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  mf="$(model_file_for "$cfg")"
  extra="$(extra_args_for "$cfg")"
  reasoning="$(reasoning_for "$cfg")"
  out="$HERE/data/responses_${cfg}.json"
  log="/tmp/judge151_srv_${cfg}.log"

  echo "=== [$cfg] modelo=$mf reasoning=$reasoning extra='$extra' ==="
  # GUARD: aborta se a porta já estiver ocupada (evita falar com servidor de outra lane).
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "[$cfg] ERRO: porta $PORT já está em uso por outro processo; abortando p/ não contaminar."; exit 1
  fi
  # shellcheck disable=SC2086
  "$LLAMA/llama-server" -m "$MODELS/$mf" -ngl 99 -c 4096 --jinja \
    --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 &
  SRV_PID=$!
  trap 'kill $SRV_PID 2>/dev/null || true' EXIT

  if ! wait_health; then
    echo "[$cfg] servidor não respondeu health; log:"; tail -20 "$log"; kill $SRV_PID 2>/dev/null || true; exit 1
  fi
  # GUARD 2: confirma que o servidor que responde é O NOSSO (pid vivo e dono da porta).
  if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "[$cfg] ERRO: nosso servidor (pid $SRV_PID) morreu; provável colisão de porta."; tail -20 "$log"; exit 1
  fi
  echo "[$cfg] servidor pronto (pid $SRV_PID) na porta $PORT."

  # Pipeline com checkpoint incremental (retoma se cair).
  "$PY" "$HERE/pipeline.py" \
    --config-key "$cfg" --model-file "$mf" --reasoning "$reasoning" \
    --url "$URL" --out "$out" --max-tokens 1024 --timeout 180

  kill $SRV_PID 2>/dev/null || true
  wait $SRV_PID 2>/dev/null || true
  trap - EXIT
  sleep 3
  echo "[$cfg] concluído -> $out"
done

echo "TODAS AS CONFIGS CONCLUÍDAS."
