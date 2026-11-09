#!/usr/bin/env bash
# run_promote.sh — container cemig_p1p2_walcy (H100, GPU 3 = cuda:0 interno):
# merge do LoRA v3s400 no f4-r32-ep2 mesclado -> HF bf16 full; build llama.cpp v0.9.0
# (mesmo cd959ba do submódulo do app) CUDA sm_90; convert -> GGUF bf16; quantize -> Q4_0.
# Reentrante (pula etapas prontas). Log em /workspace/cemig/f9/promote.log.
set -euo pipefail
W=/workspace/cemig/f9
LC=/workspace/cemig/p1p2/llama.cpp-cd959ba
BASE=/workspace/cemig/p1p2/f4-r32-ep2-merged
ADP=/workspace/cemig/checkpoints/lfm3-p2-rlvr-v3/checkpoint-400
mkdir -p $W; cd $W
echo "[1] merge $(date -u)"
if [ ! -f $W/v3s400-merged/merge_md5.json ]; then
  python /workspace/cemig/p1p2/code/merge_policy.py --base $BASE --adapter $ADP --out $W/v3s400-merged
  # template do treino (idêntico ao do base; viaja junto p/ o convert)
  cp $ADP/chat_template.jinja $W/v3s400-merged/chat_template.jinja
fi
echo "[2] build $(date -u)"
if [ ! -x $LC/build-cuda/bin/llama-quantize ]; then
  cmake -S $LC -B $LC/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF \
     -DCMAKE_BUILD_TYPE=Release > build_cfg.log 2>&1
  cmake --build $LC/build-cuda -j 32 --target llama-server llama-quantize llama-tokenize > build_make.log 2>&1
fi
echo "[3] convert $(date -u)"
if [ ! -f $W/v3s400-bf16.gguf ]; then
  pip install -q --no-deps "sentencepiece" >/dev/null 2>&1 || true
  PYTHONPATH=$LC/gguf-py python $LC/convert_hf_to_gguf.py $W/v3s400-merged --outtype bf16 --outfile $W/v3s400-bf16.gguf
fi
echo "[4] quantize $(date -u)"
if [ ! -f $W/v3s400-Q4_0.gguf ]; then
  $LC/build-cuda/bin/llama-quantize $W/v3s400-bf16.gguf $W/v3s400-Q4_0.gguf Q4_0 > quant.log 2>&1
fi
md5sum $W/v3s400-*.gguf; sha256sum $W/v3s400-Q4_0.gguf
ls -la $W/*.gguf
chmod -R a+rX $W
echo "[done] $(date -u)"; touch $W/PROMOTE_DONE
