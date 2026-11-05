#!/usr/bin/env bash
# run_bench.sh <adb-serial> [rodadas=3] [cooldown_s=90]
# Mede decode/prefill do LFM3 f4-r32-ep2 Q4_0 (APK atual) vs Q4_K_M no S24+ com o llama-bench
# compilado do MESMO llama.cpp do app (submódulo android/llama.cpp cd959ba, flags do
# android/llama/build.gradle.kts: armv8.4-a+dotprod+i8mm+fp16, KleidiAI ON, sem OpenMP/llamafile).
# Parâmetros = JNI do app: 6 threads, n_batch=n_ubatch=512. Cenário: prefill 450 tok
# (contexto comprimido) + geração 300 tok. Rodadas INTERCALADAS (Q4_0, Q4_K_M) com cooldown
# e ordem alternada para não favorecer um formato pelo throttling térmico.
set -euo pipefail
S="$1"; R="${2:-3}"; CD="${3:-90}"
A="/home/rios/android-sdk/platform-tools/adb -s $S"
D=/data/local/tmp/lfm3q
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/device"
mkdir -p "$OUT"
# pré-condição: sem economia de energia e tela acordada (low_power=1/Doze derruba o clock e invalida a medida)
$A shell "settings put global low_power 0; input keyevent KEYCODE_WAKEUP"
# keepalive: o S24+ entrou em Doze no meio da série mesmo com timeout de 30 min
( while true; do $A shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 15; done ) &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
for r in $(seq 1 "$R"); do
  # ordem alternada por rodada (AB, BA, AB...) para não dar sempre ao mesmo formato o aparelho mais frio
  if [ $((r % 2)) -eq 1 ]; then ORDER="Q4_0 Q4_K_M"; else ORDER="Q4_K_M Q4_0"; fi
  for q in $ORDER; do
    echo "[r$r $q] $(date +%T) low_power=$($A shell settings get global low_power | tr -d '\r') $($A shell dumpsys power | grep -m1 mWakefulness= | tr -d ' \r') AP=$($A shell dumpsys thermalservice | grep -m1 'mName=AP' | grep -o 'mValue=[0-9.]*')"
    $A shell "cd $D && ./llama-bench -m f4-r32-ep2-$q.gguf -t 6 -b 512 -ub 512 -p 450 -n 300 -r 1 -o json" \
      > "$OUT/bench_${q}_r$r.json" 2> "$OUT/bench_${q}_r$r.err" || echo "falhou r$r $q"
    grep -E '"(n_prompt|n_gen|avg_ts)"' "$OUT/bench_${q}_r$r.json" | tr -d ' \n'; echo
    sleep "$CD"
  done
done
