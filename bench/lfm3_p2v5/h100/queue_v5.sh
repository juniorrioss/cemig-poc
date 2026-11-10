#!/bin/bash
# queue_v5.sh — fila da GPU 3 (container cemig_p1p2_walcy; Qwen segue ao lado):
#   1. treino v5 = pool v3 (p2_prompts_v3.jsonl) + contrato v4 escalonado, 800 steps, ckpt/100
#      (parada antecipada = matar o train_p2_rlvr.py; a fila segue com os ckpts existentes)
#   2. recorte 344 (4 células + e2e + probe) dos ckpts 400/600/800 (poda)
#   3. pendências: closed_book ×1536 de base e v3s400 (completa os 344 já gerados)
#   4. pendência P1: rollout data2x+f4 completo (reentrante)
# NÃO editar este arquivo com a fila rodando (o bash relê o script).
set -u
L=/workspace/cemig/p2v5/logs; mkdir -p $L
CODE=/workspace/cemig/p1p2/code
CK=/workspace/cemig/checkpoints/lfm3-p2-rlvr-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/cemig/hf_cache
cd $CODE
WANDB_MODE=online python train_p2_rlvr.py --run lfm3-p2-rlvr-v5 \
  --prompts /workspace/cemig/p1p2/p2_prompts_v3.jsonl --contract v4 --refuse-oversample 1 \
  --max-steps 800 --lr 5e-6 --save-steps 100 --micro-batch 4 --max-prompt-tokens 2200 > $L/train_v5.log 2>&1
echo TRAIN_END >> $L/queue_v5.log
chmod -R a+rX $CK
for S in 400 600 800; do
  [ -d $CK/checkpoint-$S ] && EVAL_BS=24 bash $CODE/run_eval_p2.sh v5s$S $CK/checkpoint-$S 10,18,34,13,31,33 >> $L/queue_v5.log 2>&1
done
echo RECORTE_DONE >> $L/queue_v5.log
cd /workspace/cemig/eval
python gen_f2.py --cell closed_book --model /workspace/cemig/p1p2/f4-r32-ep2-merged \
  --adapter /workspace/cemig/checkpoints/lfm3-p2-rlvr-v3/checkpoint-400 --tag p2-v3s400 --batch 24 --max-new 3072 2>&1 | grep -E "^\[ok\]" >> $L/queue_v5.log
python gen_f2.py --cell closed_book --model /workspace/cemig/p1p2/f4-r32-ep2-merged \
  --tag p2-base --batch 24 --max-new 3072 2>&1 | grep -E "^\[ok\]" >> $L/queue_v5.log
echo CLOSED_DONE >> $L/queue_v5.log
cd $CODE && python p1_pipeline.py rollout --work /workspace/cemig/p1p2 --sources data2x f4 \
  --model /workspace/cemig/p1p2/f4-r32-ep2-merged --tag f4ep2 --batch 48 --token-budget 220000 \
  > $L/rollout_sft.log 2>&1
echo ROLLOUT_DONE >> $L/queue_v5.log
