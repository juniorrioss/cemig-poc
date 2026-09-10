#!/usr/bin/env bash
# ==============================================================================
# setup-sdk.sh — Instalação headless do OpenJDK 17 e Android SDK no WSL2
# Idempotente: executa apenas o necessário se os componentes já estiverem instalados.
# Instala em ~/android-sdk (fora do repositório).
# ==============================================================================
set -euo pipefail

SDK_ROOT="${ANDROID_HOME:-$HOME/android-sdk}"
JDK_DIR="$SDK_ROOT/jdk-17"
CMDLINE_TOOLS_DIR="$SDK_ROOT/cmdline-tools/latest"

echo "=== Configurando ambiente Android SDK em: $SDK_ROOT ==="
mkdir -p "$SDK_ROOT"

# 1. Instalação do OpenJDK 17 (Eclipse Temurin) se não estiver presente
if [ -x "$JDK_DIR/bin/java" ] && "$JDK_DIR/bin/java" -version 2>&1 | grep -q '17\.'; then
    echo "✓ OpenJDK 17 já instalado em $JDK_DIR"
else
    echo "→ Baixando e instalando OpenJDK 17 (Temurin)..."
    JDK_TMP=$(mktemp -d)
    JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
    curl -fsSL -L "$JDK_URL" -o "$JDK_TMP/openjdk17.tar.gz"
    mkdir -p "$JDK_DIR"
    tar -xzf "$JDK_TMP/openjdk17.tar.gz" --strip-components=1 -C "$JDK_DIR"
    rm -rf "$JDK_TMP"
    echo "✓ OpenJDK 17 instalado com sucesso em $JDK_DIR"
fi

export JAVA_HOME="$JDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

# 2. Instalação do Android Commandline Tools se não estiver presente
if [ -x "$CMDLINE_TOOLS_DIR/bin/sdkmanager" ]; then
    echo "✓ Android cmdline-tools já instalado em $CMDLINE_TOOLS_DIR"
else
    echo "→ Baixando e instalando Android cmdline-tools..."
    CMDLINE_TMP=$(mktemp -d)
    CMDLINE_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
    curl -fsSL "$CMDLINE_URL" -o "$CMDLINE_TMP/cmdline-tools.zip"
    
    python3 -c "
import zipfile
with zipfile.ZipFile('$CMDLINE_TMP/cmdline-tools.zip', 'r') as z:
    z.extractall('$CMDLINE_TMP')
"
    mkdir -p "$SDK_ROOT/cmdline-tools"
    rm -rf "$CMDLINE_TOOLS_DIR"
    mv "$CMDLINE_TMP/cmdline-tools" "$CMDLINE_TOOLS_DIR"
    chmod +x "$CMDLINE_TOOLS_DIR/bin/"*
    rm -rf "$CMDLINE_TMP"
    echo "✓ Android cmdline-tools instalado com sucesso em $CMDLINE_TOOLS_DIR"
fi

chmod +x "$CMDLINE_TOOLS_DIR/bin/"* 2>/dev/null || true
export ANDROID_HOME="$SDK_ROOT"
export PATH="$CMDLINE_TOOLS_DIR/bin:$SDK_ROOT/platform-tools:$PATH"

# 3. Aceitar licenças
echo "→ Aceitando licenças do Android SDK..."
mkdir -p "$SDK_ROOT/licenses"
(yes 2>/dev/null || true) | sdkmanager --sdk_root="$SDK_ROOT" --licenses > /dev/null 2>&1 || true
echo "✓ Licenças aceitas"

# 4. Instalar pacotes necessários se não estiverem presentes
PACKAGES=(
    "platform-tools"
    "platforms;android-34"
    "build-tools;34.0.0"
    "ndk;26.1.10909125"
    "cmake;3.22.1"
)

TO_INSTALL=()
for pkg in "${PACKAGES[@]}"; do
    case "$pkg" in
        "platform-tools")
            [ -d "$SDK_ROOT/platform-tools" ] || TO_INSTALL+=("$pkg")
            ;;
        "platforms;android-34")
            [ -d "$SDK_ROOT/platforms/android-34" ] || TO_INSTALL+=("$pkg")
            ;;
        "build-tools;34.0.0")
            [ -d "$SDK_ROOT/build-tools/34.0.0" ] || TO_INSTALL+=("$pkg")
            ;;
        "ndk;26.1.10909125")
            [ -d "$SDK_ROOT/ndk/26.1.10909125" ] || TO_INSTALL+=("$pkg")
            ;;
        "cmake;3.22.1")
            [ -d "$SDK_ROOT/cmake/3.22.1" ] || TO_INSTALL+=("$pkg")
            ;;
        *)
            TO_INSTALL+=("$pkg")
            ;;
    esac
done

if [ ${#TO_INSTALL[@]} -eq 0 ]; then
    echo "✓ Todos os pacotes SDK já estão instalados (${PACKAGES[*]})"
else
    echo "→ Instalando pacotes pendentes: ${TO_INSTALL[*]}..."
    sdkmanager --sdk_root="$SDK_ROOT" "${TO_INSTALL[@]}"
    echo "✓ Pacotes instalados com sucesso"
fi

echo ""
echo "=== Ambiente configurado com sucesso! ==="
echo "Para carregar as variáveis no ambiente atual:"
echo "  export JAVA_HOME=\"$JDK_DIR\""
echo "  export ANDROID_HOME=\"$SDK_ROOT\""
echo "  export PATH=\"\$JAVA_HOME/bin:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$PATH\""
