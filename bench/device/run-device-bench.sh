#!/usr/bin/env bash
# ==============================================================================
# bench/device/run-device-bench.sh
# Orquestrador de benchmark para os 6 modelos Tier 1 no Galaxy S24+ via ADB
#
# Uso:
#   ./bench/device/run-device-bench.sh [--serial IP:PORT] [--models MODEL...] [--threads N]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIAL="${DEVICE_SERIAL:-192.168.0.6:41073}"
ADB_BIN="${ADB_BIN:-$HOME/android-sdk/platform-tools/adb}"
MODELS_DIR="${MODELS_DIR:-$HOME/models-poc}"
BIN_DIR="$SCRIPT_DIR/build-android/bin"
RESULTS_DIR="$SCRIPT_DIR/results"
THREADS="${THREADS:-6}"
REPETITIONS="${REPETITIONS:-3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1500}"
GEN_TOKENS="${GEN_TOKENS:-128}"

echo "================================================================="
echo "CEMIG POC — Benchmark de SLMs no Dispositivo (Galaxy S24+)"
echo "Dispositivo:  $SERIAL"
echo "Modelos:      $MODELS_DIR"
echo "Binários:     $BIN_DIR"
echo "Resultados:   $RESULTS_DIR"
echo "Threads:      $THREADS (cluster de performance Exynos 2400)"
echo "Configuração: pp$PROMPT_TOKENS / tg$GEN_TOKENS / r$REPETITIONS"
echo "================================================================="

# Trap para garantir limpeza no aparelho se o script for interrompido
cleanup() {
    echo "Interrupção detectada. Limpando /data/local/tmp/poc no aparelho..."
    "$ADB_BIN" -s "$SERIAL" shell "rm -rf /data/local/tmp/poc" 2>/dev/null || true
}
trap cleanup INT TERM

# 1. Verificar binários compilados
if [ ! -f "$BIN_DIR/llama-bench" ] || [ ! -f "$BIN_DIR/llama-cli" ]; then
    echo "Binários não encontrados em $BIN_DIR. Executando build-android.sh..."
    "$SCRIPT_DIR/build-android.sh"
fi

# 2. Executar runner Python com parâmetros passados
python3 "$SCRIPT_DIR/device_bench.py" \
    --serial "$SERIAL" \
    --adb-bin "$ADB_BIN" \
    --models-dir "$MODELS_DIR" \
    --bin-dir "$BIN_DIR" \
    --results-dir "$RESULTS_DIR" \
    --threads "$THREADS" \
    --repetitions "$REPETITIONS" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --gen-tokens "$GEN_TOKENS" \
    "$@"
