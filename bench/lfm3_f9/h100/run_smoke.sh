#!/usr/bin/env bash
# run_smoke.sh — container (GPU 3; sem curl → health via python): sobe 2 llama-server (bf16, Q4_0) do v3s400 e roda smoke_gguf.py.
set -uo pipefail
W=/workspace/cemig/f9; BIN=/workspace/cemig/p1p2/llama.cpp-cd959ba/build-cuda/bin
cd $W
$BIN/llama-server -m v3s400-bf16.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8460 > srv_bf16.log 2>&1 & p1=$!
$BIN/llama-server -m v3s400-Q4_0.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8461 > srv_q40.log 2>&1 & p2=$!
python - <<PY
import time,requests
for _ in range(240):
    try:
        if all(requests.get(f"http://127.0.0.1:{p}/health",timeout=2).ok for p in (8460,8461)): break
    except Exception: pass
    time.sleep(1)
PY
python /workspace/cemig/p1p2/code/smoke_gguf.py > smoke.log 2>&1
kill $p1 $p2; wait 2>/dev/null
touch SMOKE_DONE
