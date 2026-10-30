#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../classifier/.venv/bin/python
# guardas A-r8 ep6/ep12
for ep in 6 12; do
  for cell in oraculo ranking distrator; do
    [ -f data/gen/gen_f2-A-r8-ep${ep}_${cell}.jsonl ] && $PY eval/score_f2.py --tag f2-A-r8-ep$ep --cell $cell --workers 16
  done
done
# closed-book B-base-r32 completados
for ep in 2 4 8 12; do
  $PY eval/score_f2.py --tag f2-B-base-r32-ep$ep --cell closed_book --workers 16
done
# probes
$PY probe/run_probe.py judge --tag F0-asis
$PY probe/run_probe.py judge --tag A-r8-ep6
$PY probe/run_probe.py judge --tag A-r32-ep2
echo JUDGE_FINAL_DONE
