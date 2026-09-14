#!/bin/bash
# run_train_tools_2_6b.sh — treino do LFM2.5-2.6B com TOOL-CALLING na DGX Spark.
#
# Espelha o run_train_tools.sh do tools_v1 (1.2B) para a bateria ser MAÇÃ-COM-MAÇÃ:
#   FASE A (curva de saturação): rank FIXO r32, 1 época, degraus aninhados 750/1500/3000
#     (o completo 4236 sai da FASE B). 1 run por degrau (barra de erro só no candidato final).
#     Objetivo: confirmar se o 2.6B satura no MESMO ponto do 1.2B (decisão ~1500, sintaxe ~750,
#     argumento nunca melhora) OU se, por ter mais capacidade, aproveita mais dado / a consulta
#     reescrita passa a funcionar (achado importante — mudaria a arquitetura).
#   FASE B (varredura de rank): degrau COMPLETO (4236), r32/r64/r128, 1 época.
#
# Reusa o motor sft_v2/train_v2 (via train_2_6b_tools.py) SEM alterá-lo: os 3 asserts
# (mlp✓ attn✓ conv✗, 122 alvos no 2.6B) e o assistant_only_loss nativo valem igual. Dados =
# os MESMOS 4.236 do tools_v1 (comparabilidade). Antes de treinar: mata llama-servers (a Spark
# já travou por memória unificada). MAXLEN medido por measure_truncation_2_6b (2.6B tokeniza
# ~18% menos que o 1.2B; default 2048, subir a 3072 só se truncar >5%). wandb online se a
# chave existir.
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/tools_2_6b/run_train_tools_2_6b.sh \
#                   > ~/cemig-poc/logs/train_tools_2_6b.log 2>&1 < /dev/null &
# Comentários PT-BR; código em inglês.
set -uo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-tools-2.6b
mkdir -p "$WANDB_DIR" ~/cemig-poc/train_tools_2_6b ~/cemig-poc/logs

BASE=~/cemig-poc/base_hf                 # 2.6B (LiquidAI/LFM2.5-2.6B)
DATADIR=~/cemig-poc/tools_v1/data        # MESMO dataset do tools_v1 (comparabilidade)
MAXLEN="${MAXLEN:-3072}"                 # medido: 2.6B trunca 5.36% a 2048 / 0% a 3072 (usa 3072, = tools_v1)
SAT_RANK="${SAT_RANK:-32}"
RANKS="${RANKS:-32 64 128}"
STEPS="${STEPS:-750 1500 3000}"

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
  $PY tools_2_6b/train_2_6b_tools.py \
    --base "$BASE" --data "$DATADIR/train_step${S}.jsonl" \
    --out-root ~/cemig-poc/train_tools_2_6b --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $SAT_RANK --max-length $MAXLEN --seed 42 \
    --tag tools_2_6b_step${S}_r${SAT_RANK}
  echo "SAT_EXIT_step${S}=$?"
  pkill -f llama-server 2>/dev/null || true; sleep 3
done

echo "== [3] FASE B: varredura de rank no degrau completo (4236) =="
for R in $RANKS; do
  echo "===== RANK r=$R (degrau 4236) ====="
  $PY tools_2_6b/train_2_6b_tools.py \
    --base "$BASE" --data "$DATADIR/train_full.jsonl" \
    --out-root ~/cemig-poc/train_tools_2_6b --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $R --max-length $MAXLEN --seed 42 \
    --tag tools_2_6b_r${R}
  echo "RANK_EXIT_r${R}=$?"
  pkill -f llama-server 2>/dev/null || true; sleep 3
done

echo "TREINO TOOLS 2.6B COMPLETO"
