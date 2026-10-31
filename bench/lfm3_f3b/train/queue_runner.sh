#!/bin/bash
# queue_runner.sh — fila de treinos do sweep F3b na H100 (container
# cemig_mobile_walcy, GPU 2 = cuda:0 interno). Roda DENTRO do container,
# desacoplado (nohup). Consome /workspace/cemig/f3b/queue/*.job em ordem
# lexicográfica; cada .job é um shell script (bash -e). Job concluído vira
# .done, falho vira .fail (fila continua).
#
# Lançamento (no host da H100):
#   docker exec -d cemig_mobile_walcy bash -c \
#     'nohup bash /workspace/cemig/f3b/queue_runner.sh > /workspace/cemig/f3b/queue.log 2>&1 &'
set -u
Q=/workspace/cemig/f3b/queue
mkdir -p "$Q"
export HF_HOME=/workspace/cemig/hf_cache CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=lfm3-1.7b-cemig

echo "[queue] start $(date)"
while true; do
  job=$(ls "$Q"/*.job 2>/dev/null | head -1 || true)
  if [ -z "$job" ]; then
    # fim da fila conhecida? espera novos jobs por até 8h, depois sai
    idle=$(( ${idle:-0} + 60 ))
    if [ "$idle" -ge 28800 ]; then echo "[queue] idle 8h, saindo"; break; fi
    sleep 60
    continue
  fi
  idle=0
  echo "[queue] >>> $job $(date)"
  if bash -e "$job" > "${job%.job}.log" 2>&1; then
    mv "$job" "${job%.job}.done"
    echo "[queue] OK  $job $(date)"
  else
    mv "$job" "${job%.job}.fail"
    echo "[queue] FAIL $job $(date) (veja ${job%.job}.log)"
  fi
done
