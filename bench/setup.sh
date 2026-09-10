#!/usr/bin/env bash
# ==============================================================================
# bench/setup.sh — Setup do ambiente de benchmark de SLMs Tier 1 para CEMIG POC
# Compila llama.cpp (CPU) e baixa modelos GGUF Q4_K_M para ~/models-poc (fora do repo)
# ==============================================================================

set -euo pipefail

# Diretórios fora do repositório
MODELS_DIR="${MODELS_DIR:-$HOME/models-poc}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/cemig-bench}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================="
echo "CEMIG POC — Setup de Benchmark SLM (Tier 1)"
echo "Repo:       $REPO_DIR"
echo "Modelos:    $MODELS_DIR"
echo "llama.cpp:  $LLAMA_CPP_DIR"
echo "Venv:       $VENV_DIR"
echo "================================================================="

mkdir -p "$MODELS_DIR"

# 1. Garantir cmake e ferramentas de compilação
if ! command -v cmake &>/dev/null; then
    if [ -x "$HOME/android-sdk/cmake/3.22.1/bin/cmake" ]; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$HOME/android-sdk/cmake/3.22.1/bin/cmake" "$HOME/.local/bin/cmake"
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "ERRO: cmake não encontrado. Instale cmake antes de prosseguir." >&2
        exit 1
    fi
fi

# 2. Clonar e compilar llama.cpp (se necessário)
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Clonando llama.cpp em $LLAMA_CPP_DIR..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-server" ] || [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-bench" ]; then
    echo "Compilando llama.cpp (CPU com AVX2 / OpenMP)..."
    cd "$LLAMA_CPP_DIR"
    # Isolar gcc nativo para evitar conflito de libs do Homebrew
    PATH="/usr/bin:/bin:$HOME/.local/bin:${PATH}" \
    CC="/usr/bin/gcc" CXX="/usr/bin/g++" \
    cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=ON
    
    PATH="/usr/bin:/bin:$HOME/.local/bin:${PATH}" \
    cmake --build build --config Release -j 8 --target llama-cli llama-server llama-bench
    echo "Compilação concluída com sucesso."
else
    echo "llama.cpp já compilado em $LLAMA_CPP_DIR/build/bin/."
fi

# 3. Configurar ambiente virtual Python fora do repositório
PYTHON_BIN="$(which python3.12 2>/dev/null || which python3)"
if [ ! -d "$VENV_DIR" ]; then
    echo "Criando virtualenv Python em $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Instalando dependências Python..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install huggingface_hub requests pypdf gguf tqdm --quiet

# 4. Download dos modelos Tier 1 em GGUF Q4_K_M
echo "Baixando modelos Tier 1 (GGUF Q4_K_M) para $MODELS_DIR..."
"$VENV_DIR/bin/python" - <<EOF
import os
from huggingface_hub import hf_hub_download

MODELS = [
    ('unsloth/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q4_K_M.gguf', 'qwen3-0.6b-q4_k_m.gguf'),
    ('unsloth/Qwen3.5-0.8B-GGUF', 'Qwen3.5-0.8B-Q4_K_M.gguf', 'qwen3.5-0.8b-q4_k_m.gguf'),
    ('bartowski/google_gemma-3-1b-it-GGUF', 'google_gemma-3-1b-it-Q4_K_M.gguf', 'gemma-3-1b-it-q4_k_m.gguf'),
    ('LiquidAI/LFM2-1.2B-RAG-GGUF', 'LFM2-1.2B-RAG-Q4_K_M.gguf', 'lfm2-1.2b-rag-q4_k_m.gguf'),
    ('LiquidAI/LFM2.5-350M-GGUF', 'LFM2.5-350M-Q4_K_M.gguf', 'lfm2.5-350m-q4_k_m.gguf'),
    ('unsloth/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'llama-3.2-1b-instruct-q4_k_m.gguf'),
]

dest_dir = os.path.expanduser('$MODELS_DIR')
os.makedirs(dest_dir, exist_ok=True)

for repo, remote_file, local_alias in MODELS:
    local_path = os.path.join(dest_dir, local_alias)
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  [OK] {local_alias:34s} ({size_mb:.1f} MB)")
        continue
    print(f"  [BAIXANDO] {local_alias} de {repo}...")
    dl_path = hf_hub_download(repo_id=repo, filename=remote_file, local_dir=dest_dir)
    if os.path.basename(dl_path) != local_alias:
        target = os.path.join(dest_dir, local_alias)
        if not os.path.exists(target):
            os.link(dl_path, target)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  [CONCLUÍDO] {local_alias} ({size_mb:.1f} MB)")
EOF

# 5. Gerar banco mínimo local de fallback caso não exista corpus/index.db
if [ ! -f "$REPO_DIR/corpus/index.db" ] && [ ! -f "$REPO_DIR/bench/data/minimal_index.db" ]; then
    echo "Gerando banco de fallback bench/data/minimal_index.db..."
    "$VENV_DIR/bin/python" "$REPO_DIR/bench/build_minimal_index.py"
fi

echo "================================================================="
echo "Setup concluído com sucesso!"
echo "Modelos prontos em: $MODELS_DIR"
echo "Binários prontos em: $LLAMA_CPP_DIR/build/bin/"
echo "Para executar o benchmark: python3 -m bench.harness"
echo "================================================================="
