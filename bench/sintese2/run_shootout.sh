#!/usr/bin/env bash
# run_shootout.sh — Orquestra o shootout de síntese v2 na RTX 5070 (llama-server CUDA).
# Para cada CANDIDATO sobe 1 servidor e roda as CONDIÇÕES pedidas (default: baseline full),
# reaproveitando o carregamento. Guard de porta duplo (não contaminar outra lane).
#
# Candidatos: lfm1.2b lfm2.6b_noth minicpm2b minicpm1b
# Uso: ./run_shootout.sh [--conditions "baseline full"] [candidato ...]
#      (sem candidatos = todos os 4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA="$HOME/llama.cpp/build-cuda/bin"
MODELS="$HOME/models-poc"
PY="$ROOT/classifier/.venv/bin/python"
PORT="${SINTESE2_PORT:-8399}"
URL="http://127.0.0.1:$PORT"
export LD_LIBRARY_PATH="$LLAMA:${LD_LIBRARY_PATH:-}"

CONDITIONS="baseline full"
if [ "${1:-}" = "--conditions" ]; then CONDITIONS="$2"; shift 2; fi

CANDS=("$@")
if [ ${#CANDS[@]} -eq 0 ]; then CANDS=(lfm1.2b lfm2.6b_noth minicpm2b minicpm1b); fi

model_file_for() {
  case "$1" in
    lfm1.2b)      echo "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf" ;;
    lfm2.6b_noth) echo "LFM2.5-2.6B-Q4_0.gguf" ;;
    minicpm2b)    echo "MiniCPM5-2B-Q4_K_M.gguf" ;;
    minicpm1b)    echo "MiniCPM5-1B-Q4_K_M.gguf" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}
family_for()   { case "$1" in lfm*) echo "lfm" ;; minicpm*) echo "minicpm" ;; esac; }
thinking_for() { case "$1" in *) echo "off" ;; esac; }  # todos thinking-OFF
# LFM2.5-2.6B desliga thinking via flag do servidor; MiniCPM via payload (harness).
server_extra_for() { case "$1" in lfm2.6b_noth) echo "--reasoning-budget 0" ;; *) echo "" ;; esac; }

wait_health() {
  for _ in $(seq 1 120); do
    if curl -s --max-time 3 "$URL/health" | grep -q '"ok"'; then return 0; fi
    sleep 2
  done
  return 1
}

for cand in "${CANDS[@]}"; do
  mf="$(model_file_for "$cand")"; fam="$(family_for "$cand")"
  th="$(thinking_for "$cand")"; extra="$(server_extra_for "$cand")"
  log="/tmp/sintese2_srv_${cand}.log"
  echo "=================================================================="
  echo "=== CANDIDATO $cand | modelo=$mf family=$fam thinking=$th extra='$extra' ==="
  echo "=================================================================="
  if [ ! -f "$MODELS/$mf" ]; then echo "[$cand] ERRO: modelo ausente $MODELS/$mf"; exit 1; fi
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "[$cand] ERRO: porta $PORT já em uso; abortando p/ não contaminar."; exit 1
  fi
  # shellcheck disable=SC2086
  "$LLAMA/llama-server" -m "$MODELS/$mf" -ngl 99 -c 4096 --jinja \
    --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 &
  SRV_PID=$!
  trap 'kill $SRV_PID 2>/dev/null || true' EXIT
  if ! wait_health; then echo "[$cand] health falhou:"; tail -20 "$log"; kill $SRV_PID 2>/dev/null||true; exit 1; fi
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "[$cand] servidor morreu (colisão?)"; tail -20 "$log"; exit 1; fi
  # Verifica que o servidor vivo é O NOSSO modelo (procedência).
  served="$(curl -s --max-time 5 "$URL/v1/models" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "?")"
  echo "[$cand] servidor pronto (pid $SRV_PID) porta $PORT | serve: $served"

  for cond in $CONDITIONS; do
    out="$HERE/data/responses_${cand}_${cond}.json"
    echo "--- [$cand/$cond] -> $out ---"
    "$PY" "$HERE/harness.py" \
      --config-key "$cand" --condition "$cond" --model-file "$mf" \
      --family "$fam" --thinking "$th" --url "$URL" --out "$out" \
      --max-tokens 1024 --timeout 180
  done

  kill $SRV_PID 2>/dev/null || true; wait $SRV_PID 2>/dev/null || true; trap - EXIT; sleep 3
  echo "[$cand] concluído."
done
echo "SHOOTOUT CONCLUÍDO: candidatos=${CANDS[*]} condições='$CONDITIONS'."
