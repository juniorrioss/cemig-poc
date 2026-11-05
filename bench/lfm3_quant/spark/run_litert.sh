#!/usr/bin/env bash
# run_litert.sh — Spark: LiteRT-LM (Python 0.19.0, venv do scout lfm3-litert) no recorte 344 op_d50.
# Uso: setsid nohup bash run_litert.sh <arquivo.litertlm> <qtag> <backend cpu|gpu> > log 2>&1
#   2 rodadas (seed 42/43). CPU: 2 shards x 6 threads em paralelo; GPU (WebGPU/Vulkan): 1 processo.
set -uo pipefail
Q=~/cemig-poc/lfm3_quant
PY=~/cemig-poc/lfm3_litert/.venv/bin/python
MODEL=$1; QT=$2; BE=$3
cd $Q
for s in 42 43; do
  r=$([ $s = 42 ] && echo r1 || echo r2)
  tag=litert_${QT}_${BE}-$r
  if [ $BE = cpu ]; then
    $PY gen_litert.py --model $MODEL --backend cpu --fmt litert_${QT}_cpu --tag $tag --seed $s --threads 6 --shard 0/2 > lg_$tag.s0.log 2>&1 &
    p0=$!
    $PY gen_litert.py --model $MODEL --backend cpu --fmt litert_${QT}_cpu --tag $tag --seed $s --threads 6 --shard 1/2 > lg_$tag.s1.log 2>&1 &
    p1=$!
    wait $p0 $p1
  else
    $PY gen_litert.py --model $MODEL --backend gpu --fmt litert_${QT}_gpu --tag $tag --seed $s > lg_$tag.s0.log 2>&1
  fi
  cat gen/gen_$tag.s*.jsonl > gen/gen_$tag.jsonl
done
touch LRT_${QT}_${BE}_DONE
echo "[done] $(date)"
