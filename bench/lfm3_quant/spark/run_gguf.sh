#!/usr/bin/env bash
# run_gguf.sh — Spark: GGUF bf16 / Q8_0 / Q4_0 do f4-r32-ep2 no recorte 344 op_d50, 2 rodadas
# (seed 42, 43) + controle Q4_0 rep=1.0 (isola a ausência de repetition penalty no LiteRT).
# llama-server CUDA (llama.cpp v0.9.0 da F7), --parallel 4. Depois: llama-bench single-stream
# (GPU e CPU 6 threads) p/ decode t/s. Desacoplado: setsid nohup bash run_gguf.sh > run_gguf.log
set -uo pipefail
Q=~/cemig-poc/lfm3_quant
F7=~/cemig-poc/lfm3_f7
BIN=$F7/llama.cpp/build-cuda/bin
PY=~/cemig-poc/lfm3_f2/.venv/bin/python
cd $Q
declare -A M=([bf16]=$F7/f4-r32-ep2-bf16.gguf [q8_0]=$Q/f4-r32-ep2-Q8_0.gguf [q4_0]=$F7/f4-r32-ep2-Q4_0.gguf)
port=8430
for f in bf16 q8_0 q4_0; do
  $BIN/llama-server -m ${M[$f]} -ngl 99 -c 16384 --parallel 4 --host 127.0.0.1 --port $port > srv_$f.log 2>&1 &
  pid=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:$port/health >/dev/null && break; sleep 2; done
  for s in 42 43; do
    r=$([ $s = 42 ] && echo r1 || echo r2)
    $PY gen_gguf.py --url http://127.0.0.1:$port --fmt gguf_$f --seed $s --tag gguf_$f-$r
  done
  if [ $f = q4_0 ]; then
    $PY gen_gguf.py --url http://127.0.0.1:$port --fmt gguf_q4_0_rep1.0 --seed 42 --tag gguf_q4_0_rep1.0-r1 --rep 1.0
  fi
  kill $pid; wait $pid 2>/dev/null
  port=$((port+1))
done
echo "[bench] $(date)"
for f in bf16 q8_0 q4_0; do
  $BIN/llama-bench -m ${M[$f]} -ngl 99 -p 1100 -n 128 -r 3 -o json > bench_gpu_$f.json 2>/dev/null
  $BIN/llama-bench -m ${M[$f]} -ngl 0 -t 6 -p 1100 -n 128 -r 2 -o json > bench_cpu6_$f.json 2>/dev/null
done
echo "[done] $(date)"
touch GGUF_DONE
