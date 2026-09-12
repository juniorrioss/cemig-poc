#!/usr/bin/env bash
# run_all.sh — orquestra o pipeline finetune2 (treino de síntese do LFM2.5-2.6B).
#
# Etapas (idempotentes; cada uma pula se a saída já existe):
#   1. gen_sft            -> data/sft.jsonl (~1500 pares; vLLM 27B)
#   2. gen_dpo_questions  -> data/dpo_questions.jsonl (~800 perguntas CLEAN)
#   3. gen_dpo            -> data/dpo.jsonl (~400 pares; 2.6B local + juiz 27B + editor 27B)
#   4. gen_ood            -> data/ood_general.json + ood_behavioral.json (CONGELADOS)
#   5. train              -> ~/train-poc (FASE A SFT + FASE B DPO + export GGUF Q4)
#   6. eval_panel         -> data/panel_<variant>.json p/ base/sft/dpo + consolidação
#
# Higiene: não-interativo, checkpoint incremental, vLLM com retry. NADA do holdout no treino.
# Servidores GGUF (base/sft/dpo) sobem via run_eval_server() abaixo (RTX 5070, thinking-OFF).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY_GEN="$ROOT/classifier/.venv/bin/python"   # busca/juiz/geração (sklearn + requests)
PY_TRAIN="$ROOT/.venv-train/bin/python"      # treino (torch cu128 + trl + peft)
LLAMA="$HOME/llama.cpp/build-cuda/bin/llama-server"
LLAMA_LD="$HOME/llama.cpp/build-cuda/bin"
MODELS="$HOME/models-poc"
TRAINP="$HOME/train-poc"

mkdir -p "$HERE/logs" "$HERE/data"

log() { echo "[run_all $(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# 1-4: datasets
# ---------------------------------------------------------------------------
gen_data() {
  if [ ! -s "$HERE/data/sft.jsonl" ] || [ "$(grep -vc '^#' "$HERE/data/sft.jsonl")" -lt 1400 ]; then
    log "gen_sft..."; "$PY_GEN" "$HERE/gen_sft.py" --n 1500
  else log "sft.jsonl OK, pulando"; fi

  if [ ! -s "$HERE/data/dpo_questions.jsonl" ]; then
    log "gen_dpo_questions..."; "$PY_GEN" "$HERE/gen_dpo_questions.py" --n 800
  else log "dpo_questions.jsonl OK, pulando"; fi

  if [ ! -s "$HERE/data/ood_general.json" ]; then
    log "gen_ood..."; "$PY_GEN" "$HERE/gen_ood.py"
  else log "ood_general.json OK (CONGELADO), pulando"; fi

  # gen_dpo exige o servidor 2.6B base no ar (porta 8401, thinking-OFF).
  if [ ! -s "$HERE/data/dpo.jsonl" ]; then
    log "subindo servidor base 2.6B p/ gen_dpo..."
    start_server "$MODELS/LFM2.5-2.6B-Q4_0.gguf" 8401 "--reasoning-budget 0"
    log "gen_dpo..."; "$PY_GEN" "$HERE/gen_dpo.py" --target 400 --server-url http://127.0.0.1:8401
    stop_server
  else log "dpo.jsonl OK, pulando"; fi
}

# ---------------------------------------------------------------------------
# 5: treino (FASE A + FASE B)
# ---------------------------------------------------------------------------
train() {
  log "train FASE A (SFT) + FASE B (DPO)..."
  HF_HUB_OFFLINE=1 "$PY_TRAIN" "$HERE/train.py" --phase both
}

# ---------------------------------------------------------------------------
# servidores GGUF p/ avaliação
# ---------------------------------------------------------------------------
SERVER_PID=""
start_server() {  # $1=gguf $2=port $3=extra_flags
  # --parallel 4: slots concorrentes p/ o eval_panel (8 workers de juiz + 4 gerações simultâneas).
  LD_LIBRARY_PATH="$LLAMA_LD" "$LLAMA" -m "$1" -ngl 99 -c 8192 --parallel 4 --port "$2" \
    --host 127.0.0.1 ${3:-} > "$HERE/logs/server_$2.log" 2>&1 &
  SERVER_PID=$!
  log "server pid=$SERVER_PID ($1 :$2), aguardando..."
  for _ in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$2/health" | grep -q ok; then log "server pronto"; return 0; fi
    sleep 2
  done
  log "ERRO: server não subiu"; return 1
}
stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; sleep 3; }

# ---------------------------------------------------------------------------
# 6: painel (base / sft / dpo)
# ---------------------------------------------------------------------------
eval_panel() {
  # base
  start_server "$MODELS/LFM2.5-2.6B-Q4_0.gguf" 8402 "--reasoning-budget 0"
  "$PY_GEN" "$HERE/eval_panel.py" --variant base --server-url http://127.0.0.1:8402
  stop_server
  # sft
  local sft_gguf="$TRAINP/merged_sft/lfm2.5-2.6b-sft-Q4_0.gguf"
  if [ -s "$sft_gguf" ]; then
    start_server "$sft_gguf" 8402 "--reasoning-budget 0"
    "$PY_GEN" "$HERE/eval_panel.py" --variant sft --server-url http://127.0.0.1:8402
    stop_server
  else log "AVISO: GGUF SFT ausente ($sft_gguf)"; fi
  # dpo
  local dpo_gguf="$TRAINP/merged_dpo/lfm2.5-2.6b-dpo-Q4_0.gguf"
  if [ -s "$dpo_gguf" ]; then
    start_server "$dpo_gguf" 8402 "--reasoning-budget 0"
    "$PY_GEN" "$HERE/eval_panel.py" --variant dpo --server-url http://127.0.0.1:8402
    stop_server
  else log "AVISO: GGUF DPO ausente ($dpo_gguf)"; fi
  # consolidação
  "$PY_GEN" "$HERE/eval_panel.py" --consolidate --variants base,sft,dpo
}

trap 'stop_server' EXIT

case "${1:-all}" in
  data) gen_data ;;
  train) train ;;
  eval) eval_panel ;;
  all) gen_data; train; eval_panel ;;
  *) echo "uso: $0 {data|train|eval|all}"; exit 1 ;;
esac
log "concluído: ${1:-all}"
