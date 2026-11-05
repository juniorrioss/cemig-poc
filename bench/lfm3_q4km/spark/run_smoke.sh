#!/usr/bin/env bash
# run_smoke.sh — Spark: sobe llama-server (v0.9.0 CUDA) com o Q4_K_M e roda smoke_q4km.py. Detached.
set -uo pipefail
O=~/cemig-poc/lfm3_q4km; F7=~/cemig-poc/lfm3_f7; BIN=$F7/llama.cpp/build-cuda/bin
PY=~/cemig-poc/lfm3_f2/.venv/bin/python
export HF_MODULES_CACHE=$F7/hfmod HF_HOME=$F7/hfhome
mkdir -p $O; cd $O
$BIN/llama-server -m ~/cemig-poc/lfm3_quant/f4-r32-ep2-Q4_K_M.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8450 > srv_q4km.log 2>&1 &
pid=$!
for i in $(seq 120); do curl -sf http://127.0.0.1:8450/health >/dev/null && break; sleep 2; done
$PY smoke_q4km.py > smoke.log 2>&1
kill $pid; wait $pid 2>/dev/null
touch SMOKE_DONE
