#!/usr/bin/env bash
# run_from_scratch.sh — teste de replicação DO ZERO (Spark): venv novo, HF_HOME vazio,
# só o que está publicado no HF (inference.py + requirements baixados do model repo).
# Saídas em $W/out: llama_seed42_r{1,2}.jsonl, hf_seed42_r{1,2}.jsonl, parity.json, env.txt.
# Uso: bash release/validation/run_from_scratch.sh   (W=<dir> opcional)
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
W=${W:-$HOME/cemig-poc/release_test/scratch}
REPO=juniorrios/cemig-lfm3-1.7b-nr
export PATH=$HOME/.local/bin:$PATH
rm -rf "$W"; mkdir -p "$W/out"; cd "$W"
export HF_HOME="$W/hf_home" HF_XET_CACHE="$W/hf_home/xet"   # cache vazio = download real
curl -fsSLO "https://huggingface.co/$REPO/resolve/main/inference.py"
curl -fsSLO "https://huggingface.co/$REPO/resolve/main/requirements.txt"
curl -fsSLO "https://huggingface.co/$REPO/resolve/main/requirements-hf.txt"
cp "$SRC/questions10.jsonl" "$SRC/ref_prompts10.jsonl.gz" "$SRC/../tools/parity_check.py" .
uv venv -q -p 3.12 .venv
. .venv/bin/activate
t0=$(date +%s); uv pip install -q --no-cache -r requirements.txt; echo "install_llama_s=$(( $(date +%s)-t0 ))" > out/env.txt
python inference.py --json --file questions10.jsonl > out/llama_seed42_r1.jsonl
python inference.py --json --file questions10.jsonl > out/llama_seed42_r2.jsonl
python inference.py "Quando a gente troca de turno, tem que trocar os bloqueios e as etiquetas?" > out/cli_example.txt
# paridade busca+compressão+prompt vs prompts congelados da F6 (encoder = llama-cpp-python CPU)
mkdir -p tools && mv parity_check.py tools/
python tools/parity_check.py --prompts ref_prompts10.jsonl.gz --out out/parity.json > /dev/null
# via transformers bf16 (torch cu130 p/ a GB10)
t0=$(date +%s)
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -q -r requirements-hf.txt
echo "install_hf_s=$(( $(date +%s)-t0 ))" >> out/env.txt
python inference.py --backend hf --json --file questions10.jsonl > out/hf_seed42_r1.jsonl
python inference.py --backend hf --json --file questions10.jsonl > out/hf_seed42_r2.jsonl
uv pip freeze > out/freeze.txt
python -c "import platform,llama_cpp,transformers,torch;print('python',platform.python_version(),'llama_cpp',llama_cpp.__version__,'transformers',transformers.__version__,'torch',torch.__version__,'arch',platform.machine())" >> out/env.txt
du -sh "$HF_HOME" >> out/env.txt
echo DONE > out/DONE
