#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python
for tv in taskvec-sum-r8 taskvec-prop-r8 taskvec-sum-r32 taskvec-prop-r32; do
  $PY eval/score_f2.py --tag f2-B-$tv --cell closed_book --workers 16
done
$PY eval/score_f2.py --tag f2-C-k2v --cell closed_book --workers 16
for cell in oraculo ranking distrator; do
  $PY eval/score_f2.py --tag f2-B-taskvec-sum-r8 --cell $cell --workers 16
done
echo DONE
