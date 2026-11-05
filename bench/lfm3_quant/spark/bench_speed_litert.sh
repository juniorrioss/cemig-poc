#!/usr/bin/env bash
# bench_speed_litert.sh — Spark: decode/prefill t/s single-stream do LiteRT-LM (CLI benchmark,
# p=1100 ~ prompt médio do op_d50, d=128) para int4/int8 × CPU(6 threads)/GPU, + hf_speed.py.
# Rodar SEM outra carga na máquina (após as gerações). Saída: speed_litert_*.log, speed_hf_bf16.json
set -uo pipefail
Q=~/cemig-poc/lfm3_quant
LRT=~/cemig-poc/lfm3_litert/.venv/bin/litert-lm
cd $Q
for qt in mixed_int4 int8; do
  M=$Q/lrt/f4-r32-ep2_${qt}_ekv2048.litertlm
  $LRT benchmark $M --backend gpu -p 1100 -d 128 --max-num-tokens 2048 --runs 3 --cache no > speed_litert_${qt}_gpu.log 2>&1
  $LRT benchmark $M --backend cpu --cpu-thread-count 6 -p 1100 -d 128 --max-num-tokens 2048 --runs 2 --cache no > speed_litert_${qt}_cpu6.log 2>&1
done
HF_MODULES_CACHE=$HOME/cemig-poc/lfm3_f7/hfmod HF_HOME=$HOME/cemig-poc/lfm3_f7/hfhome \
  ~/cemig-poc/lfm3_f2/.venv/bin/python hf_speed.py --n 20 > speed_hf.log 2>&1
touch SPEED_DONE
