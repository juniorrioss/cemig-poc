#!/usr/bin/env bash
# run_task_vectors.sh — lança as 4 variantes de task vector no container
# cemig_mobile_walcy (H100, GPU 2 do nó = cuda:0 interno). Merge+delta roda
# em CPU/float32 (task_vector.py não usa .cuda()); barato, não disputa a GPU
# com o GRPO do braço C.
#
# Pré-requisitos (já feitos pelo worker H100):
#   - /workspace/cemig/code/task_vector.py (copiado da Spark, inalterado)
#   - adapters da Spark em /workspace/cemig/checkpoints/spark_adapters/
#     lfm3-f2-B-base-r{8,32}/final/
#   - LiquidAI/LFM3-1.7B e LFM3-1.7B-Base no HF_HOME do container
#
# Variantes (fórmulas exatas em TASK_VECTOR_README.md):
#   sum  : W = merge(base, LoRA) + (instruct - base)
#   prop : idem, com delta re-escalado por mean|W_base| / mean|W_base_treinado|
#          nos tensores treinados
set -euo pipefail
CODE=/workspace/cemig/code
CKPT=/workspace/cemig/checkpoints
for rank in r8 r32; do
  for variant in sum prop; do
    out="$CKPT/taskvec-${variant}-${rank}"
    log="$CKPT/taskvec-${variant}-${rank}.log"
    echo "[launch] taskvec-${variant}-${rank}"
    python "$CODE/task_vector.py" \
      --base LiquidAI/LFM3-1.7B-Base \
      --instruct LiquidAI/LFM3-1.7B \
      --adapter "$CKPT/spark_adapters/lfm3-f2-B-base-${rank}/final" \
      --variant "$variant" \
      --out "$out" > "$log" 2>&1
    echo "[done] $out"
  done
done
