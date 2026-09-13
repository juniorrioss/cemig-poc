#!/bin/bash
# run_train_v2.sh — Varredura de RANK do SFT v2 na DGX Spark (r=16, 32, 64; 1 epoca cada).
#
# Ordem do capitao: 1 EPOCA, 3 ranks (16/32/64, alpha=2r), MESMOS dados + MESMA seed, unica
# variavel = rank. Antes de treinar: mata llama-servers (o v1 travou por memoria unificada
# esgotada com servers residentes). O train_v2.py imprime as premissas verificadas
# (targeted_module_names + trainable params) e ABORTA por assert se o LoRA errar os targets.
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/slm_scripts_v2/run_train_v2.sh \
#                   > ~/cemig-poc/logs/train_v2.log 2>&1 &
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-sft-v2
mkdir -p "$WANDB_DIR"

DATA=~/cemig-poc/train_v2/train_v2.jsonl

echo "== [1/3] liberando memoria: matando llama-server residente =="
pkill -f llama-server 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

echo "== [2/3] verificando integridade do dataset v2 =="
$PY - <<EOF
import json
from collections import Counter
n_ok = n_bad = 0
fam = Counter(); docs = set()
for line in open("$DATA", encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    try:
        d = json.loads(line)
        assert len(d["messages"]) == 3
        assert d["messages"][2]["role"] == "assistant"
        assert len(d["messages"][2]["content"]) >= 20
        fam[d["meta"]["family"]] += 1
        docs.add(d["meta"]["src_doc"])
        n_ok += 1
    except Exception:
        n_bad += 1
print(f"dataset v2: {n_ok} pares validos, {n_bad} invalidos | familias={dict(fam)} | {len(docs)} NRs")
reserved = {"nr-33", "nr-16", "nr-26"} & docs
assert not reserved, f"CONTAMINACAO: NR reservada no treino: {reserved}"
assert n_bad == 0, "dataset tem linhas invalidas — abortando"
assert n_ok >= 2500, f"esperava ~3000 pares, achei {n_ok} — abortando"
assert set(fam) >= {"oracle","recusa","parcial","distrator"}, f"faltam familias: {set(fam)}"
print("integridade OK (4 familias presentes, 0 NR reservada, 0 invalida)")
EOF

echo "== [3/3] varredura de rank (16, 32, 64), 1 epoca cada =="
for R in 16 32 64; do
  echo "===== TREINO r=$R (alpha=$((2*R))) ====="
  $PY slm_scripts_v2/train_v2.py \
    --base ~/cemig-poc/base_hf \
    --data "$DATA" \
    --out-root ~/cemig-poc/train_v2 \
    --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $R --max-length 2048 --seed 42
  echo "TRAIN_EXIT_r${R}=$?"
  # higiene de memoria entre ranks
  pkill -f llama-server 2>/dev/null || true
  sleep 3
done
echo "VARREDURA COMPLETA"
