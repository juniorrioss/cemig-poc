#!/usr/bin/env bash
# run_e2e.sh <adb-serial> <Q4_K_M|Q4_0> <apk> <gguf> <tag>
# E2E das 8 perguntas-exemplo (broadcast RUN_F7_E2E) via bench/lfm3_f7/device/deploy_and_measure.sh,
# com pré-condição de energia (low_power=0) e keepalive de tela (o S24+ entra em Doze e derruba
# o clock; o app também precisa ficar em foreground). Saída em bench/lfm3_q4km/data/device/e2e/.
set -euo pipefail
S="$1"; Q="$2"; APK_IN="$3"; GGUF="$4"; TAG="$5"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
ADB=/home/rios/android-sdk/platform-tools/adb
A="$ADB -s $S"
$A shell "settings put global low_power 0; input keyevent KEYCODE_WAKEUP"
( while true; do $A shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 15; done ) &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
OUT="$REPO/bench/lfm3_q4km/data/device/e2e" ADB="$ADB" APK="$APK_IN" LFM3_QUANT="$Q" LFM3_GGUF="$GGUF" \
  "$REPO/bench/lfm3_f7/device/deploy_and_measure.sh" "$S" "" "_$TAG"
mv "$REPO/bench/lfm3_q4km/data/device/e2e/meminfo_after.txt" "$REPO/bench/lfm3_q4km/data/device/e2e/meminfo_$TAG.txt"
