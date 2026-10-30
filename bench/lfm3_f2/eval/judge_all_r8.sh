#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python
for ep in 1 2 3 4 5 6 7 8 9 10 11 12; do
  $PY eval/score_f2.py --tag f2-A-r8-ep$ep --cell closed_book --workers 16
done
echo JUDGE_R8_DONE
