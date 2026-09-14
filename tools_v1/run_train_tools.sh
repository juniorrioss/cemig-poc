#!/bin/bash
# run_train_tools.sh — treino do LFM2.5-1.2B com TOOL-CALLING na DGX Spark.
#
# Duas fases (ordem do brief + emenda 1):
#   FASE A (curva de saturação): rank FIXO r32, 1 época, nos DEGRAUS aninhados
#     750/1500/3000/4500 (mesma seed, cada degrau ⊂ maior). 1 run por degrau (barra de erro
#     NÃO medida aqui — medida só no candidato final, ordem da emenda 1).
#   FASE B (varredura de rank): no degrau COMPLETO (4500), r16/r32/r64 (+r128 se couber),
#     1 época. Escolhe o melhor rank pela régua.
#
# Reusa o motor train_1_2b.py (== train_v2.main, rank-configurável) SEM alterá-lo: os 3
# asserts (mlp✓ attn✓ conv✗, 72 alvos no 1.2B) e o assistant_only_loss nativo valem igual.
# Os dados são MULTITURNO com tool_calls; o chat_template oficial renderiza a região de
# geração ({% generation %}) mascarando o turno 'tool'.
#
# Antes de treinar: mata llama-servers (memória unificada). max_len: ver measure_truncation
# (aumentado p/ 3072 se o multiturno truncar > ~2%). wandb online se a chave existir.
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/tools_v1/run_train_tools.sh \
#                   > ~/cemig-poc/logs/train_tools.log 2>&1 &
# Comentários PT-BR; código em inglês.
set -euo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-tools-1.2b
mkdir -p "$WANDB_DIR" ~/cemig-poc/train_tools ~/cemig-poc/logs

BASE=~/cemig-poc/base_hf_1.2b
DATADIR=~/cemig-poc/tools_v1/data
MAXLEN="${MAXLEN:-3072}"           # medido por measure_truncation (multiturno)
SAT_RANK="${SAT_RANK:-32}"          # rank fixo da curva de saturação
RANKS="${RANKS:-16 32 64 128}"      # varredura de rank no degrau completo
STEPS="${STEPS:-750 1500 3000}"     # degraus < completo (4500 vem da varredura r32)

echo "== [0] matando llama-server residente (memória unificada) =="
pkill -f llama-server 2>/dev/null || true
sleep 3

echo "== [1] integridade dos datasets (tools no system, 0 NR reservada, multiturno) =="
$PY - <<EOF
import json, glob
from collections import Counter
for path in sorted(glob.glob("$DATADIR/train_step*.jsonl")) + ["$DATADIR/train_full.jsonl"]:
    fam=Counter(); docs=set(); n_ok=n_bad=0; has_tools=0
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        try:
            d=json.loads(line); m=d["messages"]
            assert m[0]["role"]=="system" and "List of tools:" in m[0]["content"]
            assert m[-1]["role"]=="assistant"
            has_tools+=1; fam[d["meta"]["family"]]+=1
            if d["meta"].get("src_doc"): docs.add(d["meta"]["src_doc"])
            n_ok+=1
        except Exception as e:
            n_bad+=1
    assert n_bad==0, f"{path}: {n_bad} invalidos"
    assert not ({"nr-33","nr-16","nr-26"} & docs), f"{path}: NR reservada no treino"
    print(f"{path.split('/')[-1]}: {n_ok} ok, tools_in_system={has_tools}, familias={dict(fam)}")
print("integridade OK")
EOF

echo "== [2] FASE A: curva de saturação (rank fixo r$SAT_RANK) nos degraus $STEPS =="
for S in $STEPS; do
  echo "===== SATURACAO degrau=$S (r$SAT_RANK) ====="
  $PY tools_v1/train_1_2b_tools.py \
    --base "$BASE" --data "$DATADIR/train_step${S}.jsonl" \
    --out-root ~/cemig-poc/train_tools --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $SAT_RANK --max-length $MAXLEN --seed 42 \
    --tag tools_1_2b_step${S}_r${SAT_RANK}
  echo "SAT_EXIT_step${S}=$?"
  pkill -f llama-server 2>/dev/null || true; sleep 3
done

echo "== [3] FASE B: varredura de rank no degrau completo (4500) =="
for R in $RANKS; do
  echo "===== RANK r=$R (degrau 4500) ====="
  $PY tools_v1/train_1_2b_tools.py \
    --base "$BASE" --data "$DATADIR/train_full.jsonl" \
    --out-root ~/cemig-poc/train_tools --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $R --max-length $MAXLEN --seed 42 \
    --tag tools_1_2b_r${R}
  echo "RANK_EXIT_r${R}=$?"
  pkill -f llama-server 2>/dev/null || true; sleep 3
done

echo "TREINO TOOLS COMPLETO"
