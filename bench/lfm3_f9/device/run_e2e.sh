#!/usr/bin/env bash
# run_e2e.sh <adb-serial> <apk> <gguf> <tag> [extra-broadcast-args]
# F9: E2E das 8 perguntas-exemplo do APK v1.1 (v3s400 Q4_0) no S24+ via
# bench/lfm3_f7/device/deploy_and_measure.sh, com a pré-condição de energia/keepalive da lfm3_q4km
# (low_power=0 + KEYCODE_WAKEUP a cada 15 s). Saída em bench/lfm3_f9/data/device/.
set -euo pipefail
S="$1"; APK_IN="$2"; GGUF="$3"; TAG="$4"; EXTRA="${5:-}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
ADB=/home/rios/android-sdk/platform-tools/adb
A="$ADB -s $S"
OUTD="$REPO/bench/lfm3_f9/data/device"
$A shell "settings put global low_power 0; input keyevent KEYCODE_WAKEUP"
( while true; do $A shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 15; done ) &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
OUT="$OUTD" ADB="$ADB" APK="$APK_IN" LFM3_MODEL=v3s400 LFM3_QUANT=Q4_0 LFM3_GGUF="$GGUF" \
  "$REPO/bench/lfm3_f7/device/deploy_and_measure.sh" "$S" "$EXTRA" "_$TAG"
mv "$OUTD/meminfo_after.txt" "$OUTD/meminfo_$TAG.txt"
python3 "$REPO/bench/lfm3_f7/device_summary.py" "$OUTD/f7_e2e_results_$TAG.json" | tee "$OUTD/summary_$TAG.txt"
