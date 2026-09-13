#!/bin/bash
# run_train_v3.sh — Treino do SFT v3 na DGX Spark (r=64 e r=128; 1 epoca cada).
#
# Ordem do capitao (apos o v2): manter a receita (1 epoca, bs8xga4, max_len 2048, lr 1e-4,
# cosine, bf16, grad ckpt, targets por regex, assistant_only_loss, val interno), variar so o
# RANK (r64 e r128, alpha=2r) sobre os dados REBALANCEADOS (recusa <= 25%). wandb ONLINE.
#
# Antes de treinar: mata llama-servers (licao OOM do v1/v2 na memoria unificada). O train_v3
# imprime as premissas verificadas + ABORTA por assert se o LoRA errar os targets.
#
# INSTABILIDADE do r128: este runner captura grad_norm/eval_loss no log. Se r128 divergir
# (grad_norm explodindo ou eval_loss pior que r64), NAO baixa o lr sozinho -> o relatorio
# reporta e propoe lr menor (ordem do brief).
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/slm_scripts_v3/run_train_v3.sh \
#                   > ~/cemig-poc/logs/train_v3.log 2>&1 &
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-sft-v3
mkdir -p "$WANDB_DIR" ~/cemig-poc/train_v3

DATA=~/cemig-poc/train_v3/train_v3.jsonl

echo "== [1/3] liberando memoria: matando llama-server residente =="
pkill -f llama-server 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

echo "== [2/3] verificando integridade do dataset v3 (recusa <= 25%) =="
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
tot = sum(fam.values())
ref = tot - fam.get("oracle", 0)
print(f"dataset v3: {n_ok} pares validos, {n_bad} invalidos | familias={dict(fam)} | {len(docs)} NRs")
print(f"recusa-familias: {100*ref/tot:.1f}%")
reserved = {"nr-33", "nr-16", "nr-26"} & docs
assert not reserved, f"CONTAMINACAO: NR reservada no treino: {reserved}"
assert n_bad == 0, "dataset tem linhas invalidas — abortando"
assert n_ok >= 2900, f"esperava ~3000 pares, achei {n_ok} — abortando"
assert set(fam) >= {"oracle","recusa","parcial","distrator"}, f"faltam familias: {set(fam)}"
assert 100*ref/tot <= 25.0, f"recusa-familias {100*ref/tot:.1f}% ACIMA do teto de 25% — abortando"
print("integridade OK (4 familias, 0 NR reservada, 0 invalida, recusa <= 25%)")
EOF

echo "== [3/3] treino r=64 e r=128, 1 epoca cada =="
# train_v2.py e rank/tag/data-configuravel (motor unico v2/v3); so os dados e o rank mudam.
for R in 64 128; do
  echo "===== TREINO r=$R (alpha=$((2*R))) ====="
  $PY slm_scripts_v2/train_v2.py \
    --base ~/cemig-poc/base_hf \
    --data "$DATA" \
    --out-root ~/cemig-poc/train_v3 \
    --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $R --max-length 2048 --seed 42 \
    --tag sft_v3_r$R
  echo "TRAIN_EXIT_r${R}=$?"
  pkill -f llama-server 2>/dev/null || true
  sleep 3
done
echo "VARREDURA V3 COMPLETA"
