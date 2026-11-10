#!/bin/bash
# queue_cells.sh — recuperação: as 4 células ×1536 do v5s400 caíram (rollout P1 ainda
# segurava a GPU ao início da queue_full). Espera a queue_full terminar o e2e (FULL_DONE),
# interrompe o rollout que ela encadeia, gera as células (reentrante) e retoma o rollout.
set -u
L=/workspace/cemig/p2v5/logs
CODE=/workspace/cemig/p1p2/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/cemig/hf_cache
while ! grep -q FULL_DONE $L/queue_full.log 2>/dev/null; do sleep 30; done
pkill -f queue_full.sh; pkill -f "p1_pipeline.py rollout"; sleep 60
cd /workspace/cemig/eval
for CELL in distrator ranking oraculo closed_book; do
  python gen_f2.py --cell $CELL --model /workspace/cemig/p1p2/f4-r32-ep2-merged \
    --adapter /workspace/cemig/checkpoints/lfm3-p2-rlvr-v5/checkpoint-400 --tag p2-v5s400 \
    --batch 24 --max-new 3072 > $L/cells_$CELL.log 2>&1
  tail -n 1 $L/cells_$CELL.log >> $L/queue_cells.log
done
echo CELLS_DONE >> $L/queue_cells.log
cd $CODE && python p1_pipeline.py rollout --work /workspace/cemig/p1p2 --sources data2x f4 \
  --model /workspace/cemig/p1p2/f4-r32-ep2-merged --tag f4ep2 --batch 48 --token-budget 220000 \
  >> $L/rollout_sft.log 2>&1
echo ROLLOUT_DONE >> $L/queue_cells.log
