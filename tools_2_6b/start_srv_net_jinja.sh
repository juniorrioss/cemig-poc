#!/bin/bash
# start_srv_net_jinja.sh <model> <port> <ctx> <parallel> — llama-server bind 0.0.0.0 c/ jinja + reasoning-budget 0
MODEL="$1"; PORT="$2"; CTX="${3:-8192}"; PAR="${4:-4}"
cd ~/cemig-poc
pkill -f "port $PORT"; sleep 3
setsid ./llama.cpp/build-cuda/bin/llama-server -m "$MODEL" -ngl 99 -c "$CTX" \
  --parallel "$PAR" --reasoning-budget 0 --jinja --host 0.0.0.0 --port "$PORT" \
  > logs/server_${PORT}.log 2>&1 < /dev/null &
echo "launched on $PORT (jinja, reasoning-budget 0)"
