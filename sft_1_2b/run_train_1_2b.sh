#!/bin/bash
# run_train_1_2b.sh — Treino do SFT do LFM2.5-1.2B-Instruct na DGX Spark (varredura de rank).
#
# Ordem do capitao: aplicar ao 1.2B a MESMA receita do v3 do 2.6B, MANTENDO os dados
# (train_v3.jsonl) e variando o RANK (r16, r32, r64 e r128). O 1.2B tem menos capacidade,
# entao o ponto de saturacao do rank pode ser OUTRO (por isso a varredura ampla).
#
# Receita IDENTICA ao v3 (1 epoca, bs8xga4, max_len 2048, lr 1e-4, cosine, bf16, grad ckpt,
# targets por regex, assistant_only_loss, val interno). Base = LFM2.5-1.2B-Instruct (bf16);
# NAO existe checkpoint QAD em HF -> o risco QAD e medido depois (bf16 vs Q4 vs QAD original).
#
# Antes de treinar: mata llama-servers (licao OOM da memoria unificada). O train_1_2b imprime
# as premissas verificadas (targets=72 esperado: 6*4 attn + 16*3 MLP, 0 conv) e ABORTA por
# assert se o LoRA errar os targets.
#
# Se a loss/eval indicar que 1 epoca e INSUFICIENTE p/ o modelo menor (underfitting -> eval_loss
# ainda caindo forte / muito acima do 2.6B), o relatorio REPORTA com evidencia e propoe 2
# epocas — NAO se decide sozinho (ordem do brief).
#
# Uso (na Spark): setsid nohup bash ~/cemig-poc/slm_scripts_1_2b/run_train_1_2b.sh \
#                   > ~/cemig-poc/logs/train_1_2b.log 2>&1 &
# Comentarios PT-BR; codigo em ingles.
set -euo pipefail

cd ~/cemig-poc
PY=~/jupyterlab/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=~/cemig-poc/wandb
export WANDB_PROJECT=cemig-sft-1.2b
mkdir -p "$WANDB_DIR" ~/cemig-poc/train_1_2b

BASE=~/cemig-poc/base_hf_1.2b
DATA=~/cemig-poc/train_v3/train_v3.jsonl   # MESMOS dados do v3 (ordem do capitao)
RANKS="${RANKS:-16 32 64 128}"

echo "== [1/3] liberando memoria: matando llama-server residente =="
pkill -f llama-server 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

echo "== [2/3] verificando arquitetura do base 1.2B + integridade do dataset =="
$PY - <<EOF
import json, re, glob
from collections import Counter
from safetensors import safe_open
# --- PASSO 0: arquitetura do base (conta modulos, NAO assume nomes do 2.6B) ---
base="$BASE"
cfg=json.load(open(base+"/config.json"))
lt=cfg.get("layer_types",[])
print(f"base 1.2B: arch={cfg.get('architectures')} n_layers={len(lt)} {dict(Counter(lt))}")
pat=Counter()
for f in glob.glob(base+"/*.safetensors"):
    with safe_open(f,"pt") as h:
        for k in h.keys():
            pat[re.sub(r'\.\d+\.','.N.',k)]+=1
n_attn=pat.get("model.layers.N.self_attn.q_proj.weight",0)
n_mlp=pat.get("model.layers.N.feed_forward.w1.weight",0)
n_conv=pat.get("model.layers.N.conv.conv.weight",0)
print(f"modulos: attn(q_proj)={n_attn} mlp(w1)={n_mlp} conv={n_conv}")
assert n_attn>0 and n_mlp>0 and n_conv>0, "arquitetura inesperada (faltam attn/mlp/conv)"
exp_targets=n_attn*4 + n_mlp*3
print(f"targets LoRA esperados (regex do v2) = {n_attn}*4 + {n_mlp}*3 = {exp_targets}")
# --- integridade do dataset v3 (recusa <= 25%, 0 NR reservada) ---
n_ok=n_bad=0; fam=Counter(); docs=set()
for line in open("$DATA", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    try:
        d=json.loads(line); assert len(d["messages"])==3
        assert d["messages"][2]["role"]=="assistant"
        fam[d["meta"]["family"]]+=1; docs.add(d["meta"]["src_doc"]); n_ok+=1
    except Exception: n_bad+=1
tot=sum(fam.values()); ref=tot-fam.get("oracle",0)
print(f"dataset v3: {n_ok} pares, {n_bad} invalidos | familias={dict(fam)} | {len(docs)} NRs | recusa={100*ref/tot:.1f}%")
assert not ({"nr-33","nr-16","nr-26"} & docs), "CONTAMINACAO: NR reservada no treino"
assert n_bad==0 and n_ok>=2900, "dataset invalido"
assert 100*ref/tot <= 25.0, "recusa-familias acima de 25%"
print("PASSO0 + integridade OK")
EOF

echo "== [3/3] treino varredura de rank ($RANKS), 1 epoca cada =="
for R in $RANKS; do
  echo "===== TREINO 1.2B r=$R (alpha=$((2*R))) ====="
  $PY slm_scripts_1_2b/train_1_2b.py \
    --base "$BASE" \
    --data "$DATA" \
    --out-root ~/cemig-poc/train_1_2b \
    --models-dir ~/cemig-poc/models \
    --llama-dir ~/cemig-poc/llama.cpp \
    --epochs 1 --lr 1e-4 --bs 8 --grad-accum 4 \
    --lora-r $R --max-length 2048 --seed 42 \
    --tag sft_1_2b_r$R
  echo "TRAIN_EXIT_r${R}=$?"
  pkill -f llama-server 2>/dev/null || true
  sleep 3
done
echo "VARREDURA 1.2B COMPLETA"
