#!/usr/bin/env bash
# ==============================================================================
# bench/device-liquid/run-liquid-bench.sh
# Launcher de benchmark para modelos Liquid LFM2.5 no Galaxy S24+
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERIAL="${DEVICE_SERIAL:-192.168.0.6:41073}"
ADB_BIN="${ADB_BIN:-$HOME/android-sdk/platform-tools/adb}"
MODELS_DIR="${MODELS_DIR:-$HOME/models-poc}"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "================================================================="
echo "CEMIG POC — Benchmark Liquid On-Device (Galaxy S24+)"
echo "Dispositivo: $SERIAL"
echo "Modelos:     $MODELS_DIR"
echo "Resultados:  $RESULTS_DIR"
echo "================================================================="

# Trap de limpeza
cleanup() {
    echo "Interrupção detectada. Limpando /data/local/tmp/poc..."
    "$ADB_BIN" -s "$SERIAL" shell "rm -rf /data/local/tmp/poc; svc power stayon false" 2>/dev/null || true
}
trap cleanup INT TERM

python3 "$SCRIPT_DIR/device_liquid_bench.py" \
    --serial "$SERIAL" \
    --adb-bin "$ADB_BIN" \
    --models-dir "$MODELS_DIR" \
    --results-dir "$RESULTS_DIR" \
    "$@"
