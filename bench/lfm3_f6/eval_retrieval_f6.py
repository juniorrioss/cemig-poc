#!/usr/bin/env python3
"""
eval_retrieval_f6.py — Passo 1 da F6: escolha do ponto de operação dos manuais no RRF.

Roda variantes da F5 (mesmo harness: eval_retrieval.run_variant -> POC 151 + bench 1536,
subconjunto de 400 com ouro em manual) e grava em bench/lfm3_f6/data/retrieval/.
Nomes de variante = gramática da F5: base[+man{w}][+margin{m}][+intent].

Uso: classifier/.venv/bin/python bench/lfm3_f6/eval_retrieval_f6.py --variants v4app,v4app+man1.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402,F401  (redireciona C.DATA)
import common_f5 as C  # noqa: E402
from eval_retrieval import brief, run_variant  # noqa: E402
from metrics import SpanStore  # noqa: E402
from retriever_f5 import F5Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    args = ap.parse_args()
    rtv = F5Retriever(with_manuals=True)
    store = SpanStore(rtv.man.span_store())
    rtv.qc.ensure([it["question"] for it in C.load_151()] + [r["question"] for r in C.load_bench()])
    for n in args.variants.split(","):
        print(brief(run_variant(rtv, n, store)), flush=True)


if __name__ == "__main__":
    main()
