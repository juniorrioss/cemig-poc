#!/bin/bash
# queue_full.sh — bateria completa do ckpt vencedor da poda (v5s400): 4 células ×1536
# (gen_f2 é reentrante: estende os 344 do recorte) + e2e op_d50 ×1536 + ×151; depois
# retoma o rollout P1 (reentrante). NÃO editar com a fila rodando.
set -u
L=/workspace/cemig/p2v5/logs
CODE=/workspace/cemig/p1p2/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/cemig/hf_cache
FULL_CELLS="distrator ranking oraculo closed_book" EVAL_BS=24 bash $CODE/run_eval_full.sh v5s400 \
  /workspace/cemig/checkpoints/lfm3-p2-rlvr-v5/checkpoint-400 >> $L/queue_full.log 2>&1
echo FULL_DONE >> $L/queue_full.log
cd $CODE && python p1_pipeline.py rollout --work /workspace/cemig/p1p2 --sources data2x f4 \
  --model /workspace/cemig/p1p2/f4-r32-ep2-merged --tag f4ep2 --batch 48 --token-budget 220000 \
  >> $L/rollout_sft.log 2>&1
echo ROLLOUT_DONE >> $L/queue_full.log
