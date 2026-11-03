#!/bin/bash
# container_runner.sh — pipeline AUTÔNOMO do F4 dentro do container da H100
# (ordem do capitão: zero dependência de rota externa; juiz 27B no mesmo nó
# via gateway docker 172.17.0.1:8005).
#
# Sequência (roda horas sem supervisão):
#   1. espera o treino terminar (checkpoints 3 épocas em OUT);
#   2. por época: gen ranking+distrator (1536, gen_f2.py) -> julgamento local
#      (scorepack: score_f4_container.py);
#   3. por época: gen oraculo+closed_book -> julgamento;
# Logs: /workspace/cemig/f4/runner.log. Reentrante (gen/score têm checkpoint).
#
# Lançamento (no host):
#   docker exec -d cemig_mobile_walcy bash -c \
#     'nohup bash /workspace/cemig/f4/container_runner.sh > /workspace/cemig/f4/runner.log 2>&1 &'
set -u
export HF_HOME=/workspace/cemig/hf_cache CUDA_VISIBLE_DEVICES=0
export F2_GEN_URL=http://172.17.0.1:8005/v1
export REGUA_JUDGE_URL=http://172.17.0.1:8005/v1

F4=/workspace/cemig/f4
OUT=/workspace/cemig/checkpoints/lfm3-f4-tools-r32
EVAL=/workspace/cemig/eval
MERGED=$F4/merged-rebuilt

echo "[runner] start $(date)"

# 1. espera o treino acabar (final/ salvo pelo train_f4.py)
while [ ! -d "$OUT/final" ]; do
  if ! ls $OUT/checkpoint-* >/dev/null 2>&1 && ! pgrep -f train_f4.py >/dev/null; then
    echo "[runner] treino nem começou nem está rodando? aguardando mesmo assim"
  fi
  sleep 300
done
echo "[runner] treino concluído $(date); checkpoints:"; ls -d $OUT/checkpoint-*

# mapeia época -> checkpoint (ordem crescente de step)
CKPTS=$(ls -d $OUT/checkpoint-* | sort -t- -k2 -n)
EP=0
for CK in $CKPTS; do
  EP=$((EP+1))
  TAG="f4-r32-ep$EP"
  for CELL in ranking distrator; do
    echo "[runner] === gen $TAG $CELL $(date)"
    (cd $EVAL && python gen_f2.py --cell $CELL --model $MERGED \
        --adapter $CK --tag $TAG --batch 48 --max-new 3072)
    echo "[runner] === score $TAG $CELL $(date)"
    (cd $F4/scorepack && python score_f4_container.py --tag $TAG --cell $CELL --workers 16)
  done
done

# 3. oraculo + closed_book (guardas) para todas as épocas
EP=0
for CK in $CKPTS; do
  EP=$((EP+1))
  TAG="f4-r32-ep$EP"
  for CELL in oraculo closed_book; do
    echo "[runner] === gen $TAG $CELL $(date)"
    (cd $EVAL && python gen_f2.py --cell $CELL --model $MERGED \
        --adapter $CK --tag $TAG --batch 48 --max-new 3072)
    echo "[runner] === score $TAG $CELL $(date)"
    (cd $F4/scorepack && python score_f4_container.py --tag $TAG --cell $CELL --workers 16)
  done
done

echo "[runner] DONE $(date)"
