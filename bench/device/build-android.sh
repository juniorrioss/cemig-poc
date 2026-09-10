#!/usr/bin/env bash
# ==============================================================================
# bench/device/build-android.sh
# Compila llama-bench e llama-cli para Android arm64 via CMake + Android NDK
#
# Configurações de arquitetura e flags:
# - ABI: arm64-v8a
# - Plataforma Android: android-31
# - Flags ARM: -march=armv8.4-a+dotprod+i8mm+fp16 (habilita DotProd, Int8 Matmul e FP16)
# - KleidiAI: ON (otimizações Arm KleidiAI para núcleos Cortex / Neoverse)
# - Dependências: BUILD_SHARED_LIBS=OFF para binários estáticos independentes de libs externas
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Diretórios padrão
ANDROID_SDK="${ANDROID_SDK:-$HOME/android-sdk}"
ANDROID_NDK="${ANDROID_NDK:-$ANDROID_SDK/ndk/26.1.10909125}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
OUTPUT_DIR="$SCRIPT_DIR/build-android"
BUILD_DIR="$OUTPUT_DIR/build"

echo "================================================================="
echo "CEMIG POC — Compilação llama.cpp para Android arm64"
echo "NDK:        $ANDROID_NDK"
echo "llama.cpp:  $LLAMA_CPP_DIR"
echo "Saída:      $OUTPUT_DIR/bin"
echo "================================================================="

# 1. Validações prévias
if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERRO: Android NDK não encontrado em: $ANDROID_NDK" >&2
    echo "Execute android/setup-sdk.sh ou defina ANDROID_NDK." >&2
    exit 1
fi

TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "ERRO: Arquivo toolchain não encontrado em: $TOOLCHAIN_FILE" >&2
    exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "ERRO: Diretório llama.cpp não encontrado em: $LLAMA_CPP_DIR" >&2
    exit 1
fi

# Configurar ferramentas do host (CMake / Ninja / Strip)
CMAKE_BIN="cmake"
if ! command -v cmake &>/dev/null; then
    if [ -x "$ANDROID_SDK/cmake/3.22.1/bin/cmake" ]; then
        CMAKE_BIN="$ANDROID_SDK/cmake/3.22.1/bin/cmake"
        export PATH="$ANDROID_SDK/cmake/3.22.1/bin:$PATH"
    else
        echo "ERRO: cmake não encontrado." >&2
        exit 1
    fi
fi

NINJA_ARG=""
if command -v ninja &>/dev/null || [ -x "$ANDROID_SDK/cmake/3.22.1/bin/ninja" ]; then
    export PATH="$ANDROID_SDK/cmake/3.22.1/bin:$PATH"
    NINJA_ARG="-G Ninja"
fi

STRIP_BIN="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"

# 2. Configuração do CMake
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR/bin"

# Flags de instrução ARM testadas e verificadas no Exynos 2400 (Galaxy S24+):
# -march=armv8.4-a+dotprod+i8mm+fp16 ativa HAVE_DOTPROD, HAVE_MATMUL_INT8 e HAVE_FP16_VECTOR_ARITHMETIC
ARM_FLAGS="-march=armv8.4-a+dotprod+i8mm+fp16"

echo "Configurando CMake..."
"$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" $NINJA_ARG \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_C_FLAGS="$ARM_FLAGS" \
    -DCMAKE_CXX_FLAGS="$ARM_FLAGS" \
    -DGGML_CPU_KLEIDIAI=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DBUILD_SHARED_LIBS=OFF

# 3. Compilação dos alvos llama-bench e llama-cli
echo "Compilando alvos llama-bench e llama-cli..."
"$CMAKE_BIN" --build "$BUILD_DIR" --target llama-bench llama-cli --parallel "$(nproc)"

# 4. Copiar e preparar binários de saída
echo "Copiando binários para $OUTPUT_DIR/bin/..."
cp "$BUILD_DIR/bin/llama-bench" "$OUTPUT_DIR/bin/llama-bench"
cp "$BUILD_DIR/bin/llama-cli" "$OUTPUT_DIR/bin/llama-cli"

if [ -x "$STRIP_BIN" ]; then
    echo "Aplicando llvm-strip nos binários..."
    "$STRIP_BIN" -s "$OUTPUT_DIR/bin/llama-bench"
    "$STRIP_BIN" -s "$OUTPUT_DIR/bin/llama-cli"
fi

chmod +x "$OUTPUT_DIR/bin/llama-bench" "$OUTPUT_DIR/bin/llama-cli"

echo "================================================================="
echo "Compilação concluída com sucesso!"
ls -lh "$OUTPUT_DIR/bin/llama-bench" "$OUTPUT_DIR/bin/llama-cli"
file "$OUTPUT_DIR/bin/llama-bench"
echo "================================================================="
