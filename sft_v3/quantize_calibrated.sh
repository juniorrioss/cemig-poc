#!/bin/bash
# quantize_calibrated.sh — PARTE 4: quantiza o modelo escolhido COM e SEM calibracao (imatrix).
#
# Roda na Spark. Recebe o bf16 GGUF do modelo escolhido e o corpus de calibracao (calib_v3.txt).
# Produz 4 artefatos p/ comparar quanto a calibracao recupera da perda bf16->Q4:
#   1. <tag>-Q4_0.gguf            (SEM imatrix — baseline, == pipeline do v2)
#   2. <tag>-Q4_0-imatrix.gguf    (COM imatrix — calibrado)
#   3. <tag>-Q4_K_M.gguf          (SEM imatrix)
#   4. <tag>-Q4_K_M-imatrix.gguf  (COM imatrix)
# Q4_K_M costuma ganhar MAIS com imatrix (usa a matriz na atribuicao de bits por bloco);
# Q4_0 e mais simples. Comparamos os 4 na regua p/ escolher o formato final com evidencia.
#
# Uso (na Spark): bash ~/cemig-poc/slm_scripts_v3/quantize_calibrated.sh \
#                   ~/cemig-poc/models/lfm2.5-2.6b-sft_v3_r64-bf16.gguf sft_v3_r64 \
#                   ~/cemig-poc/train_v3/calib_v3.txt
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

BF16="$1"; TAG="$2"; CALIB="${3:-$HOME/cemig-poc/train_v3/calib_v3.txt}"
BIN=~/cemig-poc/llama.cpp/build-cuda/bin
M=~/cemig-poc/models
IMAT="$M/imatrix_${TAG}.dat"

echo "== [1/3] gerando imatrix de calibracao (dominio NR/campo) =="
# -ngl 99: computa a imatrix na GPU; --chunks limita o custo (corpus ja e representativo).
"$BIN/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IMAT" -ngl 99 --chunks 400 2>&1 | tail -8

echo "== [2/3] quantizando SEM calibracao (baseline) =="
"$BIN/llama-quantize" "$BF16" "$M/lfm2.5-2.6b-${TAG}-Q4_0.gguf"   Q4_0   2>&1 | tail -3
"$BIN/llama-quantize" "$BF16" "$M/lfm2.5-2.6b-${TAG}-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -3

echo "== [3/3] quantizando COM calibracao (imatrix) =="
"$BIN/llama-quantize" --imatrix "$IMAT" "$BF16" "$M/lfm2.5-2.6b-${TAG}-Q4_0-imatrix.gguf"   Q4_0   2>&1 | tail -3
"$BIN/llama-quantize" --imatrix "$IMAT" "$BF16" "$M/lfm2.5-2.6b-${TAG}-Q4_K_M-imatrix.gguf" Q4_K_M 2>&1 | tail -3

echo "== artefatos =="
ls -la "$M"/lfm2.5-2.6b-${TAG}-Q4*.gguf
echo "QUANT_CALIB_DONE"
