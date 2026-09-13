#!/bin/bash
# run_quant_evals.sh — PARTE 4: avalia as 4 quantizacoes do v3_r64 (COM e SEM calibracao).
#
# Mede quanto a calibracao (imatrix) recupera da perda bf16->Q4 (no v2 foram ~9 p.p.):
#   Q4_0 sem/com imatrix; Q4_K_M sem/com imatrix. Contra o bf16 (teto do treino) ja avaliado.
# Reusa o eval_checkpoint_v3.sh (4 suites, regua honesta). UMA por vez (licao OOM).
# Comentarios PT-BR; codigo em ingles.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PORT=8483
M=/home/walcyrios/cemig-poc/models

run () { echo "############ EVAL $2 ############"; ./eval_checkpoint_v3.sh "$1" "$2" "$PORT" v2; }

# Q4_0 sem calibracao == v3_r64_q4 ja avaliado (reusa score_v3_r64_q4_*); nao reroda.
run $M/lfm2.5-2.6b-sft_v3_r64-Q4_0-imatrix.gguf    v3_r64_q4_calib
run $M/lfm2.5-2.6b-sft_v3_r64-Q4_K_M.gguf          v3_r64_q4km_nocalib
run $M/lfm2.5-2.6b-sft_v3_r64-Q4_K_M-imatrix.gguf  v3_r64_q4km_calib

echo "ALL_QUANT_EVALS_DONE"
