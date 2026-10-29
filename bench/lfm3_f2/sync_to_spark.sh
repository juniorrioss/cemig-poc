#!/bin/bash
# Sincroniza o necessário para a Spark (~/cemig-poc/lfm3_f2/).
set -euo pipefail
cd "$(dirname "$0")"
DEST=walcyrios@spark-b431:cemig-poc/lfm3_f2/

scp -q train/train_lora_f2.py train/task_vector.py train/prep_train_data.py \
    eval/gen_f2.py \
    k2v/train_grpo_k2v.py \
    ../lfm3_f0/common_f0.py \
    "$DEST"
# dados de treino (gerados pelo prep_train_data localmente)
ssh walcyrios@spark-b431 'mkdir -p cemig-poc/lfm3_f2/data/train cemig-poc/lfm3_f2/data/k2v'
scp -q data/train/train.jsonl data/train/val.jsonl "$DEST"data/train/ 2>/dev/null || echo "(train data ainda não gerado)"
scp -q data/synth/train_mix.jsonl "$DEST"data/ 2>/dev/null || echo "(mix ainda não gerado)"
scp -q k2v/data/k2v_qa_checklist.jsonl "$DEST"data/k2v/ 2>/dev/null || echo "(checklist ainda não)"
echo ok
