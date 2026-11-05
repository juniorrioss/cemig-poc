#!/usr/bin/env bash
# run_convert.sh — Spark: merge LoRA -> HF full, build llama.cpp v0.9.0 (CUDA sm_121),
# convert -> GGUF bf16, quantize -> Q4_0. Desacoplado (setsid nohup); log em convert.log.
set -euo pipefail
W=~/cemig-poc/lfm3_f7
PY=~/cemig-poc/lfm3_f2/.venv/bin/python
export HF_MODULES_CACHE=$HOME/cemig-poc/lfm3_f7/hfmod HF_HOME=$HOME/cemig-poc/lfm3_f7/hfhome
LC=$W/llama.cpp
cd $W
echo "[1] merge $(date)"
if [ ! -f $W/merged/md5.json ]; then
  $PY $W/merge_f4.py --model ~/cemig-poc/lfm3_f4/lfm3-1.7b-data2x-ep4-merged \
     --adapter ~/cemig-poc/lfm3_f5/checkpoint-284 --out $W/merged
fi
echo "[2] build $(date)"
if [ ! -x $LC/build-cuda/bin/llama-quantize ]; then
  cmake -S $LC -B $LC/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > build_cfg.log 2>&1
  cmake --build $LC/build-cuda -j 16 --target llama-server llama-quantize llama-cli llama-tokenize > build_make.log 2>&1
fi
echo "[3] convert $(date)"
if [ ! -f $W/f4-r32-ep2-bf16.gguf ]; then
  $PY -m pip --version >/dev/null 2>&1 || true
  PYTHONPATH=$LC/gguf-py $PY $LC/convert_hf_to_gguf.py $W/merged --outtype bf16 --outfile $W/f4-r32-ep2-bf16.gguf
fi
echo "[4] quantize $(date)"
if [ ! -f $W/f4-r32-ep2-Q4_0.gguf ]; then
  $LC/build-cuda/bin/llama-quantize $W/f4-r32-ep2-bf16.gguf $W/f4-r32-ep2-Q4_0.gguf Q4_0
fi
md5sum $W/f4-r32-ep2-*.gguf
ls -la $W/*.gguf
echo "[done] $(date)"
touch $W/CONVERT_DONE
