#!/bin/bash
# Celulas closed_book dos artefatos novos (H100 GPU 2), Spark saturada.
set -uo pipefail
cd /workspace/cemig/eval
PY=python
NORMS=10,18,34,13,31,33
CK=/workspace/cemig/checkpoints
gen() { $PY gen_f2.py --cell closed_book --model "$1" ${2:+--adapter "$2"} --tag "$3" --norms $NORMS --batch 48 --max-new 3072 >> /workspace/cemig/eval/logs_eval_h100.txt 2>&1; }

for v in sum-r8 sum-r32 prop-r8 prop-r32; do
  echo "=== taskvec-$v closed_book ===" >> /workspace/cemig/eval/logs_eval_h100.txt
  gen "$CK/taskvec-$v" "" "f2-taskvec-$v"
done
echo "=== C-k2v-grpo final closed_book ===" >> /workspace/cemig/eval/logs_eval_h100.txt
gen LiquidAI/LFM3-1.7B "$CK/lfm3-f2-C-k2v-grpo/final" "f2-C-k2v-final"
echo "=== C-k2v-grpo ckpt200 closed_book ===" >> /workspace/cemig/eval/logs_eval_h100.txt
gen LiquidAI/LFM3-1.7B "$CK/lfm3-f2-C-k2v-grpo/checkpoint-200" "f2-C-k2v-ck200"
echo DONE > /workspace/cemig/eval/EVAL_H100_DONE
