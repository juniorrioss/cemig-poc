#!/usr/bin/env bash
# run_dense_model.sh — pipeline de um modelo denso: build (GPU) -> dump -> eval (CPU/sklearn).
# Uso: ./run_dense_model.sh <slug> <hf_model> [dim_trunc] [prefix_kind] [q_prompt] [d_prompt] [--exp]
# Ex.: ./run_dense_model.sh e5small intfloat/multilingual-e5-small 0 e5
set -euo pipefail
cd "$(dirname "$0")"

SLUG="$1"; MODEL="$2"; DIM="${3:-0}"; PREFIX="${4:-}"; QP="${5:-}"; DP="${6:-}"; EXP="${7:-}"

TRAIN=../.venv-train/bin/python
CLF=../classifier/.venv/bin/python

BUILD_ARGS=(--model "$MODEL" --out "indices/dense_${SLUG}.db" --meta-out "results/dense_${SLUG}_meta.json")
DUMP_ARGS=(--dense-db "indices/dense_${SLUG}.db" --model "$MODEL" --out "results/dense_rank_${SLUG}.json")
[ "$DIM" != "0" ] && { BUILD_ARGS+=(--dim-trunc "$DIM"); DUMP_ARGS+=(--dim-trunc "$DIM"); }
[ -n "$PREFIX" ] && { BUILD_ARGS+=(--prefix-kind "$PREFIX"); DUMP_ARGS+=(--prefix-kind "$PREFIX"); }
[ -n "$QP" ] && { BUILD_ARGS+=(--query-prompt-name "$QP"); DUMP_ARGS+=(--query-prompt-name "$QP"); }
[ -n "$DP" ] && BUILD_ARGS+=(--doc-prompt-name "$DP")
[ "$EXP" = "--exp" ] && BUILD_ARGS+=(--with-expansion)
[ "$EXP" = "--exp-only" ] && BUILD_ARGS+=(--expansion-only)

echo "=== BUILD $SLUG ($MODEL) ==="
$TRAIN dense.py "${BUILD_ARGS[@]}" 2>&1 | grep -vE "Loading weights|FutureWarning|HTTP Request|InconsistentVersion|warnings.warn"
echo "=== DUMP $SLUG ==="
$TRAIN dump_dense.py "${DUMP_ARGS[@]}" 2>&1 | grep -vE "Loading weights|FutureWarning|HTTP Request|InconsistentVersion|warnings.warn"
echo "=== EVAL $SLUG ==="
$CLF eval_dense.py --dense-rank "results/dense_rank_${SLUG}.json" \
    --meta "results/dense_${SLUG}_meta.json" --label "$SLUG" \
    --json-out "results/dense_${SLUG}.json" 2>&1 | tail -12
