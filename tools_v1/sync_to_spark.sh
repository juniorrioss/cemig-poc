#!/bin/bash
# sync_to_spark.sh — copia dados de treino + scripts p/ a DGX Spark (procedência, não-interativo).
# Uso: bash sync_to_spark.sh
set -euo pipefail
SPARK=walcyrios@spark-b431
DST='~/cemig-poc/tools_v1'
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$SPARK" 'mkdir -p ~/cemig-poc/tools_v1/data ~/cemig-poc/logs'
# datasets de treino (degraus + completo) e manifest
scp "$HERE"/data/train_step*.jsonl "$HERE"/data/train_full.jsonl \
    "$HERE"/data/prep_manifest.json "$HERE"/data/truncation.json \
    "$SPARK:cemig-poc/tools_v1/data/"
# scripts de treino (o motor train_v2 já está em ~/cemig-poc/slm_scripts_v2)
scp "$HERE"/train_1_2b_tools.py "$HERE"/run_train_tools.sh "$SPARK:cemig-poc/tools_v1/"
echo "sync OK -> $SPARK:$DST"
