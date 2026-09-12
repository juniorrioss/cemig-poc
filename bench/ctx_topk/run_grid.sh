#!/usr/bin/env bash
# run_grid.sh — Gera respostas das 8 células de qualidade (topK×trecho) nas 151 reais.
# Sintetizador embarcado (1.2B QAD) no llama-server CUDA :8410 (--parallel 4).
# Encoder denso GGUF em :8399. Higiene: guard de porta, timeout, checkpoint incremental.
# Comentários PT-BR. Procedência: task poc-ctx-topk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../classifier/.venv/bin/python"
LLM_URL="http://127.0.0.1:8410"
EMB_URL="http://127.0.0.1:8399"

# Guard: os dois servidores precisam estar no ar (latência GPU não vale, mas geração sim).
curl -sf "$LLM_URL/health" >/dev/null || { echo "ERRO: llama-server 1.2B (:8410) fora do ar"; exit 1; }
curl -sf "$EMB_URL/health" >/dev/null || { echo "ERRO: embedding GGUF (:8399) fora do ar"; exit 1; }

for topk in 2 3 4 5; do
  for trecho in full trim; do
    key="k${topk}_${trecho}"
    echo "=== Gerando $key ==="
    timeout 1800 "$PY" "$HERE/pipeline.py" \
      --config-key "$key" --top-k "$topk" --trecho "$trecho" \
      --url "$LLM_URL" --emb-url "$EMB_URL" \
      --out "$HERE/data/responses_${key}.json" --workers 4
  done
done
echo "=== Grade de qualidade concluída ==="
