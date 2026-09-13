#!/bin/bash
# run_all_evals.sh — Roda a Parte 3 completa (4 suites x variantes), UMA por vez.
#
# Cada variante 2.6B sobe UM llama-server na Spark (lição OOM do v1: nunca 2 servers). O 27B
# (teto) roda pelo servidor do juiz direto (sem llama-server). Sequencial p/ não competir por
# memória unificada. Comentários PT-BR; código em inglês.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PORT=8473

# 2.6B: (gguf_remoto, label, system). system=v2 p/ base e treinados (mesmo prompt, isola treino).
run () {  # $1=gguf $2=label $3=system
  echo "############ EVAL $2 ############"
  ./eval_checkpoint_v2.sh "$1" "$2" "$PORT" "$3"
}

M=/home/walcyrios/cemig-poc/models

# Treinados v2 (Q4 = embarque; bf16 = separa efeito quant do efeito treino).
run $M/lfm2.5-2.6b-sft_v2_r16-Q4_0.gguf  v2_r16_q4   v2
run $M/lfm2.5-2.6b-sft_v2_r16-bf16.gguf  v2_r16_bf16 v2
run $M/lfm2.5-2.6b-sft_v2_r32-Q4_0.gguf  v2_r32_q4   v2
run $M/lfm2.5-2.6b-sft_v2_r32-bf16.gguf  v2_r32_bf16 v2
run $M/lfm2.5-2.6b-sft_v2_r64-Q4_0.gguf  v2_r64_q4   v2
run $M/lfm2.5-2.6b-sft_v2_r64-bf16.gguf  v2_r64_bf16 v2

# Base 2.6B (mesmo prompt v2 -> isola o efeito do TREINO na recusa).
run $M/LFM2.5-2.6B-Q4_0.gguf   base_q4_v2   v2
run $M/LFM2.5-2.6B-bf16.gguf   base_bf16_v2 v2

# v1 r=16 (SEM treino de recusa) — mede a recusa de um modelo treinado só em oráculo.
run $M/lfm2.5-2.6b-sft_ep1-Q4_0.gguf  v1_r16_q4  v2

echo "ALL_2P6B_EVALS_DONE"
