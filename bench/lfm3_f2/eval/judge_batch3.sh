#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python
for tag in f2-A-r32-ep12 f2-B-base-r8-ep12; do
  $PY eval/score_f2.py --tag $tag --cell closed_book --workers 16
done
echo DONE
