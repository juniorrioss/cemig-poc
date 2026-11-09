#!/bin/bash
# run_eval_p2.sh — geração da AVALIAÇÃO do P2 dentro do container da GPU 3
# (cemig_p1p2_walcy; Qwen segue no ar ao lado, só o JULGAMENTO usa ele depois).
#
# Para cada TAG (modelo base mesclado + adapter opcional), MESMO harness/sampling
# da série LFM3 (0,1/50/rep 1,1):
#   1. 4 células padrão do benchmark_mobile (gen_f2.py; system F0 SEM tools):
#      ranking, distrator, oraculo, closed_book — NORMS vazio = 1536, ou
#      NORMS=10,18,34,13,31,33 = recorte 344
#   2. e2e op_d50 do recorte 344 (prompts CONGELADOS da F6, formato do APP com
#      turno tool, top-5 -> 50%): mede cita item-ouro + alucinação no app
#   3. probe geral (130 itens congelados)
# Uso: bash run_eval_p2.sh TAG ADAPTER_DIR_OR_EMPTY [NORMS]
set -u
TAG=$1
ADAPTER=${2:-}
NORMS=${3:-}
export HF_HOME=/workspace/cemig/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/workspace/cemig/p1p2/f4-r32-ep2-merged
EVAL=/workspace/cemig/eval
CODE=/workspace/cemig/p1p2/code
AD=""
[ -n "$ADAPTER" ] && AD="--adapter $ADAPTER"
NA=""
[ -n "$NORMS" ] && NA="--norms $NORMS"

echo "[eval] $TAG start $(date)"
for CELL in distrator ranking oraculo closed_book; do
  (cd $EVAL && python gen_f2.py --cell $CELL --model $BASE $AD --tag p2-$TAG $NA \
      --batch ${EVAL_BS:-16} --max-new 3072) 2>&1 | grep -E "^\[(gen|ok)\]" | tail -n 2
done
(cd $CODE/e2e && python gen_e2e.py --model $BASE $AD --tag p2-$TAG \
    --prompts $CODE/e2e/prompts_op_d50.jsonl --batch ${EVAL_BS:-16} --max-new 768) 2>&1 | grep -E "^\[(gen|ok)\]" | tail -n 2
(cd $CODE/probe && python run_probe.py gen --model $BASE $AD --tag p2-$TAG --batch ${EVAL_BS:-16}) 2>&1 | tail -n 2
echo "[eval] $TAG DONE $(date)"
