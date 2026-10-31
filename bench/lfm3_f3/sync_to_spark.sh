#!/bin/bash
# sync_to_spark.sh — prepara ~/cemig-poc/lfm3_f3 na Spark para a grade de
# avaliação F3 (reusa o harness gen_f2.py da F2 INALTERADO, batch 48).
# A geração usa o venv da F2 (~/cemig-poc/lfm3_f2/.venv).
set -euo pipefail
SPARK=walcyrios@spark-b431
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/../.."

ssh $SPARK 'mkdir -p ~/cemig-poc/lfm3_f3/data'
# harness da F2 (gen) + common do F0 + parquet do benchmark (já estão na Spark
# em lfm3_f0/lfm3_f2; copiamos só o runner da grade F3)
scp "$HERE/eval/run_eval_grid_f3.sh" $SPARK:~/cemig-poc/lfm3_f3/
echo "[ok] Spark pronta (checkpoints chegam via scp da H100)"
