#!/bin/bash
# run_train.sh — Retomada do SFT na DGX Spark.
#
# Ordem do capitao (msg 003): sem cautela excessiva no batch — a causa do travamento foi
# treino + 2 llama-servers residentes competindo pelos 121 GB unificados, NAO o treino ser
# grande. Este launcher mantem so o basico de higiene de memoria:
#   1. MATA qualquer llama-server residente (o basico — geracao e treino nunca coexistem);
#   2. verifica a integridade dos 3.000 pares antes de treinar;
#   3. treina LoRA config ORIGINAL (bs=8 x ga=4, max_length=2048), checkpoint por epoca,
#      export GGUF bf16+Q4_0 por epoca;
#   4. WANDB_MODE=offline (telemetria completa em disco; sync retroativo quando houver chave).
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/slm_scripts/run_train.sh \
#                   > ~/cemig-poc/logs/train_sft.log 2>&1 &
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Telemetria wandb em disco (chave nao configurada na Spark; sync retroativo depois).
export WANDB_MODE=offline
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-slm-oraculo
mkdir -p "$WANDB_DIR"

DATA=~/cemig-poc/train/train_sft.jsonl

echo "== [1/3] liberando memoria: matando llama-server residente =="
pkill -f llama-server 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

echo "== [2/3] verificando integridade do dataset =="
$PY - <<EOF
import json, sys
n_ok = n_bad = 0
docs = set()
for line in open("$DATA", encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    try:
        d = json.loads(line)
        assert len(d["messages"]) == 3
        assert d["messages"][2]["role"] == "assistant"
        assert len(d["messages"][2]["content"]) >= 20
        docs.add(d["meta"]["src_doc"])
        n_ok += 1
    except Exception as e:
        n_bad += 1
print(f"dataset: {n_ok} pares validos, {n_bad} invalidos, {len(docs)} NRs")
# Muralha 2: nenhuma NR reservada pode estar no treino.
reserved = {"nr-33", "nr-16", "nr-26"} & docs
assert not reserved, f"CONTAMINACAO: NR reservada no treino: {reserved}"
assert n_bad == 0, "dataset tem linhas invalidas — abortando"
assert n_ok >= 2900, f"esperava ~3000 pares, achei {n_ok} — abortando"
print("integridade OK (0 NR reservada, 0 invalida)")
EOF

echo "== [3/3] treino LoRA (config original bs=8/max_len=2048) =="
$PY slm_scripts/train_sft.py \
  --base ~/cemig-poc/base_hf \
  --data "$DATA" \
  --out-root ~/cemig-poc/train \
  --models-dir ~/cemig-poc/models \
  --llama-dir ~/cemig-poc/llama.cpp \
  --epochs 3 --lr 1e-4 --bs 8 --grad-accum 4 --lora-r 16 --lora-alpha 32 \
  --max-length 2048 --seed 42
echo "TRAIN_EXIT=$?"
