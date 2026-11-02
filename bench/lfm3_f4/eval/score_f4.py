"""
score_f4.py — shim da régua honesta para as células ESTÁTICAS do F4: reusa
bench/lfm3_f2/eval/score_f2.py (régua do F0, juiz 27B) com os diretórios
apontados para bench/lfm3_f4/data. Zero lógica nova.

Uso (local, juiz direto):
    classifier/.venv/bin/python bench/lfm3_f4/eval/score_f4.py \
        --tag f4-r32-ep2 --cell ranking [--workers 16]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f4/eval
F4 = HERE.parent
REPO = F4.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))

import score_f2  # noqa: E402

score_f2.F2 = F4

if __name__ == "__main__":
    score_f2.main()
