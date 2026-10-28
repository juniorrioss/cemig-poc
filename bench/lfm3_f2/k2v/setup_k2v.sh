#!/bin/bash
# Clona o K2V pinado e aplica os patches locais (thinking-OFF no cliente OpenAI).
set -euo pipefail
cd "$(dirname "$0")"
PIN=fbd8bf92544e4758bf86393e713d80d9036b01e0
if [ ! -d K2V ]; then
  git clone https://github.com/SeedScientist/K2V K2V
fi
git -C K2V checkout "$PIN"
git -C K2V apply --check ../patches.diff 2>/dev/null && git -C K2V apply ../patches.diff || echo "patch já aplicado"
uv venv .venv --python 3.12 -q 2>/dev/null || true
VIRTUAL_ENV=$PWD/.venv uv pip install -q openai python-dotenv tenacity tiktoken tqdm networkx numpy pandas jieba nltk pyyaml gradio transformers wikipedia
echo ok
