#!/bin/bash
# run_all_evals_1_2b.sh — Avalia toda a varredura do 1.2B + as 3 referencias de base.
#
# Cada variante sobe UM llama-server na Spark, UMA por vez (licao OOM da memoria unificada).
# O 27B (teto) e o 2.6B v3_r64 (melhor atual) sao reusados/comparados na consolidacao:
#   - 27B: scores ja existem em sft_v2/data/score_27b_* (reuso, nao reroda).
#   - 2.6B v3_r64: scores ja existem em sft_v3/data/score_v3_r64_* (reuso, nao reroda).
#
# system: 'v2' p/ os treinados (prompt que permite recusa util, o que treinaram);
#         'numeros' p/ as bases NAO-treinadas (base nao sabe o prompt v2; usa o acionavel,
#          igual o v2/v3 fizeram com base/27b).
# Comentarios PT-BR; codigo em ingles.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PORT=8483
M=/home/walcyrios/cemig-poc/models

run () {  # $1=gguf $2=label $3=system
  echo "############ EVAL $2 ############"
  ./eval_checkpoint_1_2b.sh "$1" "$2" "$PORT" "$3"
}

# --- Bases (referencia p/ o delta e p/ o risco QAD) ---
run $M/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf   base_qad_q4      numeros   # EMBARCADO de fabrica
run $M/LFM2.5-1.2B-Instruct-base-bf16.gguf  base_bf16        numeros   # base HF alta precisao
run $M/LFM2.5-1.2B-Instruct-base-Q4_0.gguf  base_q4          numeros   # base requant simples (sem QAD)

# --- Treinados (Q4 = embarque; bf16 = separa efeito quant do efeito treino) ---
for R in 16 32 64 128; do
  run $M/lfm2.5-1.2b-sft_1_2b_r${R}-Q4_0.gguf   sft_r${R}_q4    v2
  run $M/lfm2.5-1.2b-sft_1_2b_r${R}-bf16.gguf   sft_r${R}_bf16  v2
done

echo "ALL_1_2B_EVALS_DONE"
