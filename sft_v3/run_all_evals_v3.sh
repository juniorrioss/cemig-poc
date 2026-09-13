#!/bin/bash
# run_all_evals_v3.sh — Parte 4: avalia v3_r64, v3_r128 (bf16+Q4) UMA por vez (licao OOM).
#
# Cada variante 2.6B sobe UM llama-server na Spark. O v2_r64 (bf16+Q4) ja foi avaliado no
# sft_v2 e sera comparado maca-com-maca reusando os score_v2_r64_* existentes (nao reroda).
# Comentarios PT-BR; codigo em ingles.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PORT=8481

run () {  # $1=gguf $2=label $3=system
  echo "############ EVAL $2 ############"
  ./eval_checkpoint_v3.sh "$1" "$2" "$PORT" "$3"
}

M=/home/walcyrios/cemig-poc/models

# Treinados v3 (Q4 = embarque; bf16 = separa efeito quant do efeito treino).
run $M/lfm2.5-2.6b-sft_v3_r64-Q4_0.gguf   v3_r64_q4    v2
run $M/lfm2.5-2.6b-sft_v3_r64-bf16.gguf   v3_r64_bf16  v2
run $M/lfm2.5-2.6b-sft_v3_r128-Q4_0.gguf  v3_r128_q4   v2
run $M/lfm2.5-2.6b-sft_v3_r128-bf16.gguf  v3_r128_bf16 v2

echo "ALL_V3_EVALS_DONE"
