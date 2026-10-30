"""
score_f3.py — shim da régua honesta para a F3: reusa bench/lfm3_f2/eval/
score_f2.py (régua do F0, juiz 27B) com os diretórios de dados apontados para
bench/lfm3_f3/data. Zero lógica nova (reuso obrigatório do brief).

Uso (local, juiz direto):
    classifier/.venv/bin/python bench/lfm3_f3/eval/score_f3.py \
        --tag f3-r32-ep2 --cell ranking [--workers 16]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3/eval
F3 = HERE.parent
REPO = F3.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))

import score_f2  # noqa: E402

# Redireciona os diretórios do score_f2 para a F3 (gen/ e scored/ locais).
score_f2.F2 = F3

if __name__ == "__main__":
    score_f2.main()
