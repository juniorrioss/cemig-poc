#!/usr/bin/env python3
"""
common_f6.py — F6: manuais nos índices (ordem do capitão) sobre a base da F5.

Reusa INTEGRALMENTE o código da F5 (bench/lfm3_f5: retriever_f5, manuals, compress,
metrics, score_e2e) — só redireciona as SAÍDAS para bench/lfm3_f6/data e os caches
NOVOS (janelas dos manuais) para bench/lfm3_f6/cache. Os caches da F5 (align.json,
manuals.db, manuals_dense.npz, units*.{json,npz}, qemb.npz) são lidos de
bench/lfm3_f5/cache (reprodutíveis com `make -C bench/lfm3_f5 align manuals units`).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
F5 = REPO / "bench" / "lfm3_f5"
for p in (str(F5), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common_f5 as C  # noqa: E402

DATA = HERE / "data"
CACHE = HERE / "cache"            # gitignored (reprodutível)
F5_CACHE = C.CACHE

# saídas de eval_retrieval/score da F5 passam a cair na F6
C.DATA = DATA

TOL = 0.005                       # tolerância da trava (0,5 p.p.)
