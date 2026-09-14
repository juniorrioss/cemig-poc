#!/bin/bash
# sync_to_spark.sh — copia scripts do tools_2_6b p/ a DGX Spark (procedência, não-interativo).
# O dataset (tools_v1/data/train_*.jsonl) JÁ está na Spark (do tools_v1) — NÃO reenvia (o valor
# desta task é a comparabilidade: MESMOS 4.236 diálogos, mesmo split).
# Uso: bash sync_to_spark.sh
set -euo pipefail
SPARK=walcyrios@spark-b431
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$SPARK" 'mkdir -p ~/cemig-poc/tools_2_6b/data ~/cemig-poc/logs'
scp "$HERE"/verify_modules_2_6b.py "$HERE"/measure_truncation_2_6b.py \
    "$HERE"/train_2_6b_tools.py "$HERE"/run_train_tools_2_6b.sh \
    "$SPARK:cemig-poc/tools_2_6b/"
echo "sync OK -> $SPARK:~/cemig-poc/tools_2_6b/"
