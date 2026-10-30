#!/bin/bash
# run_train_f3.sh — lança o treino F3 na H100 (container cemig_mobile_walcy,
# GPU 2 do nó = cuda:0 interno). Rodar NO HOST da H100 (aluno_walcy@10.100.0.111):
#   bash run_train_f3.sh
# Processo desacoplado (docker exec -d + nohup); monitorar por tail do log.
set -euo pipefail

CONT=cemig_mobile_walcy
WORK=/workspace/cemig/f3

docker exec -d "$CONT" bash -c "
  cd $WORK &&
  nohup python train_f3.py \
    --rank 32 --epochs 4 \
    --train $WORK/train.jsonl --val $WORK/val.jsonl \
    --out /workspace/cemig/checkpoints/lfm3-f3-robustez-r32 \
    --run lfm3-f3-robustez-r32 \
    > $WORK/train_f3.log 2>&1 &
  echo \$! > $WORK/train_f3.pid
"
echo "lancado; log: docker exec $CONT tail -f $WORK/train_f3.log"
