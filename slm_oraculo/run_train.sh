#!/bin/bash
# run_train.sh — Retomada FOOLPROOF do SFT na DGX Spark (memory-safe).
#
# Ordem do capitao: nao derrubar a maquina de novo por OOM. Este launcher:
#   1. MATA qualquer llama-server residente (libera a memoria unificada — a causa provavel
#      do travamento: bf16+Q4 servers + LoRA competindo pelos 121 GB);
#   2. verifica a integridade dos 3.000 pares antes de treinar;
#   3. usa config memory-safe (bs=4 x ga=8, max_length=1536, group_by_length,
#      expandable_segments) — ver analise no README;
#   4. treina LoRA (3 epocas, checkpoint por epoca) e exporta GGUF por epoca.
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

echo "== [3/3] treino LoRA memory-safe =="
$PY slm_scripts/train_sft.py \
  --base ~/cemig-poc/base_hf \
  --data "$DATA" \
  --out-root ~/cemig-poc/train \
  --models-dir ~/cemig-poc/models \
  --llama-dir ~/cemig-poc/llama.cpp \
  --epochs 3 --lr 1e-4 --bs 4 --grad-accum 8 --lora-r 16 --lora-alpha 32 \
  --max-length 1536 --seed 42
echo "TRAIN_EXIT=$?"
