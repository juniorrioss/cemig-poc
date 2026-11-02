#!/bin/bash
# run_train_f4.sh — lança o treino F4 na H100 (container cemig_mobile_walcy,
# GPU 2 do nó = cuda:0 interno). Rodar NO HOST da H100 (aluno_walcy@10.100.0.111):
#   bash run_train_f4.sh
# Pré-requisitos no /workspace/cemig/f4/ (rsync antes):
#   train_f4.py, train.jsonl, val.jsonl, lfm3-1.7b-data2x-ep4-merged/
# Processo desacoplado (docker exec -d + nohup); monitorar por tail do log.
set -euo pipefail

CONT=cemig_mobile_walcy
WORK=/workspace/cemig/f4

docker exec -d "$CONT" bash -c "
  export HF_HOME=/workspace/cemig/hf_cache CUDA_VISIBLE_DEVICES=0 &&
  cd $WORK &&
  nohup python train_f4.py \
    --rank 32 --epochs 3 \
    --model $WORK/lfm3-1.7b-data2x-ep4-merged \
    --train $WORK/train.jsonl --val $WORK/val.jsonl \
    --out /workspace/cemig/checkpoints/lfm3-f4-tools-r32 \
    --run lfm3-f4-tools-r32 \
    > $WORK/train_f4.log 2>&1 &
  echo \$! > $WORK/train_f4.pid
"
echo "lancado; log: docker exec $CONT tail -f $WORK/train_f4.log"
