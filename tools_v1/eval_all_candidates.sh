#!/bin/bash
# eval_all_candidates.sh — avalia TODA a bateria do PASSO 4 a partir DESTE host (via túnel).
#
# Candidatos (GGUF na Spark, em ~/cemig-poc/models/):
#   - curva de saturação: tools_1_2b_step{750,1500,3000}_r32 + tools_1_2b_r32 (=4500) [Q4]
#   - varredura de rank:  tools_1_2b_r{16,32,64,128} [bf16 + Q4]
#   - baselines:          LFM2.5-1.2B-Instruct-QAD-Q4_0 (embarcado, --no-model-rewrite off),
#                         LFM2.5-1.2B-Instruct-base-Q4_0, lfm2.5-1.2b-sft_1_2b_r64-Q4_0 (sem tools)
#   - teto:               Qwen 27B (via juiz; args só, --no-model-rewrite)
#
# Cada candidato: sobe server na Spark, tunela, roda run_eval + score_e2e, derruba.
# O candidato FINAL escolhido roda 3x (media/desvio) — feito à parte com labels _run{1,2,3}.
#
# Uso: bash eval_all_candidates.sh "<lista de tags>"   (default = todos)
# Comentários PT-BR; código em inglês.
set -uo pipefail
MODELS_DIR='~/cemig-poc/models'
DEFAULT="tools_1_2b_step750_r32 tools_1_2b_step1500_r32 tools_1_2b_step3000_r32 \
tools_1_2b_r16 tools_1_2b_r32 tools_1_2b_r64 tools_1_2b_r128"
TAGS="${1:-$DEFAULT}"
TOPK="${TOPK:-2}"

for TAG in $TAGS; do
  GGUF="$MODELS_DIR/lfm2.5-1.2b-${TAG}-Q4_0.gguf"
  echo "########## $TAG (Q4) ##########"
  bash run_all_evals.sh "$GGUF" "$TAG" 8500 "$TOPK" || echo "FALHA $TAG"
done
echo "BATERIA COMPLETA"
