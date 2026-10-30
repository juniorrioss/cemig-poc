#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python
for tag in f2-A-r32-ep1 f2-A-r32-ep2 f2-A-r32-ep4 f2-A-r32-ep6 f2-A-r32-ep8 f2-B-base-r8-ep2 f2-B-base-r8-ep4 f2-B-base-r8-ep6 f2-B-base-r8-ep8; do
  [ -f data/gen/gen_${tag}_closed_book.jsonl ] || continue
  $PY eval/score_f2.py --tag $tag --cell closed_book --workers 16
done
echo JUDGE_BATCH2_DONE
