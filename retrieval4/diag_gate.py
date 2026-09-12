#!/usr/bin/env python3
"""
diag_gate.py — Diagnóstico do gate de confiança por classificador (por que free piora).

Para um pkl de classificador, mede no holdout 151:
  - fração que dispara filtro-duro (prob top-1 >= thr);
  - acurácia da NR entre os filtrados-duro (se erra a NR, o filtro-duro MATA o recall);
  - top-1/top-2 do classificador restrito às 151 (não 171).
Ajuda a explicar por que um classificador "mais acurado" no holdout pode reduzir o recall.

Uso: ../classifier/.venv/bin/python diag_gate.py --clf models/v4_v1free.pkl --thr 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

from bm25 import NrClassifier  # noqa: E402
from eval_common import load_holdout  # noqa: E402
from nr_taxonomy import NONE_LABEL, normalize_nr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    clf = NrClassifier(args.clf)
    ho = load_holdout("qa_v2")
    topk, scores = clf.predict([it.question for it in ho], k=2)

    n = len(ho)
    hard = hard_ok = 0
    soft = soft_top1 = soft_top2 = 0
    t1 = t2 = 0
    for i, it in enumerate(ho):
        gold = normalize_nr(it.doc)
        preds = [p for p in topk[i] if p != NONE_LABEL]
        p1 = float(scores[i].max())
        if gold in topk[i][:1]:
            t1 += 1
        if gold in topk[i][:2]:
            t2 += 1
        if p1 >= args.thr and preds:
            hard += 1
            if preds[0] == gold:
                hard_ok += 1
        else:
            soft += 1
            if gold in topk[i][:1]:
                soft_top1 += 1
            if gold in topk[i][:2]:
                soft_top2 += 1

    print(f"clf={Path(args.clf).name} thr={args.thr}")
    print(f"  151 clf top1={100*t1/n:.1f}%  top2={100*t2/n:.1f}%")
    print(f"  filtro-duro: {hard}/{n} ({100*hard/n:.0f}%)  acerto-NR nos duros={hard_ok}/{hard} "
          f"({100*hard_ok/max(hard,1):.0f}%)  [errar aqui MATA o recall]")
    print(f"  boost-suave: {soft}/{n}  top1={100*soft_top1/max(soft,1):.0f}%  top2={100*soft_top2/max(soft,1):.0f}%")


if __name__ == "__main__":
    main()
