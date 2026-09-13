#!/bin/bash
# make_base_ggufs.sh — Gera os GGUFs do BASE 1.2B NAO-treinado (roda na Spark).
#
# Precisamos de 3 referencias de base p/ isolar os efeitos (ordem do capitao):
#   1. QAD-Q4_0 (LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf) — o EMBARCADO de fabrica (pushed por scp).
#   2. Instruct bf16 — o base HF convertido, alta precisao (teto do base).
#   3. Instruct Q4_0 — o base HF requantizado SEM QAD (plain requant).
# Comparar (2) vs (3) vs (1) mede o quanto o QAD-de-fabrica ganha sobre a requantizacao simples,
# e serve de baseline honesto p/ o treinado (treino bf16 -> Q4_0 tambem e requant simples).
#
# Uso (na Spark): bash ~/cemig-poc/slm_scripts_1_2b/make_base_ggufs.sh
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail
cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
BASE=~/cemig-poc/base_hf_1.2b
LLAMA=~/cemig-poc/llama.cpp
M=~/cemig-poc/models
BF16=$M/LFM2.5-1.2B-Instruct-base-bf16.gguf
Q4=$M/LFM2.5-1.2B-Instruct-base-Q4_0.gguf

if [ ! -f "$BF16" ]; then
  echo "== convert base HF -> bf16 GGUF =="
  $PY $LLAMA/convert_hf_to_gguf.py "$BASE" --outfile "$BF16" --outtype bf16
fi
if [ ! -f "$Q4" ]; then
  echo "== quantize base bf16 -> Q4_0 (plain requant, sem QAD) =="
  $LLAMA/build-cuda/bin/llama-quantize "$BF16" "$Q4" Q4_0
fi
echo "BASE GGUFs prontos: $BF16 / $Q4"
ls -la "$BF16" "$Q4" ~/cemig-poc/models/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf 2>&1
