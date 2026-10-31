"""
score_f3b.py — shim da régua honesta para o sweep F3b: reusa
bench/lfm3_f2/eval/score_f2.py (régua do F0, juiz 27B) com os diretórios
apontados para bench/lfm3_f3b/data (gen/ e scored/). Zero lógica nova.

Uso (local, juiz direto):
    classifier/.venv/bin/python bench/lfm3_f3b/eval/score_f3b.py \
        --tag f3b-recusa35-ep3 --cell ranking [--workers 16]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3b/eval
F3B = HERE.parent
REPO = F3B.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))

import score_f2  # noqa: E402

score_f2.F2 = F3B

if __name__ == "__main__":
    score_f2.main()
