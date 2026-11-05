#!/usr/bin/env bash
# run_q4km.sh — Spark: ponto extra GGUF Q4_K_M (mesmo protocolo do run_gguf.sh: 2 rodadas seeds 42/43 + llama-bench).
set -uo pipefail
Q=~/cemig-poc/lfm3_quant; F7=~/cemig-poc/lfm3_f7; BIN=$F7/llama.cpp/build-cuda/bin; PY=~/cemig-poc/lfm3_f2/.venv/bin/python
cd $Q
[ -f f4-r32-ep2-Q4_K_M.gguf ] || $BIN/llama-quantize $F7/f4-r32-ep2-bf16.gguf f4-r32-ep2-Q4_K_M.gguf Q4_K_M > quant_q4km.log 2>&1
$BIN/llama-server -m f4-r32-ep2-Q4_K_M.gguf -ngl 99 -c 16384 --parallel 4 --host 127.0.0.1 --port 8440 > srv_q4_k_m.log 2>&1 &
pid=$!
for i in $(seq 120); do curl -sf http://127.0.0.1:8440/health >/dev/null && break; sleep 2; done
$PY gen_gguf.py --url http://127.0.0.1:8440 --fmt gguf_q4_k_m --seed 42 --tag gguf_q4_k_m-r1
$PY gen_gguf.py --url http://127.0.0.1:8440 --fmt gguf_q4_k_m --seed 43 --tag gguf_q4_k_m-r2
kill $pid; wait $pid 2>/dev/null
$BIN/llama-bench -m f4-r32-ep2-Q4_K_M.gguf -ngl 99 -p 1100 -n 128 -r 3 -o json > bench_gpu_q4_k_m.json 2>/dev/null
$BIN/llama-bench -m f4-r32-ep2-Q4_K_M.gguf -ngl 0 -t 6 -p 1100 -n 128 -r 2 -o json > bench_cpu6_q4_k_m.json 2>/dev/null
ls -l f4-r32-ep2-Q4_K_M.gguf; md5sum f4-r32-ep2-Q4_K_M.gguf
touch Q4KM_DONE
