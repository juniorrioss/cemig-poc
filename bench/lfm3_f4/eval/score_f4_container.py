#!/usr/bin/env python3
"""
score_f4_container.py — julgamento das células ESTÁTICAS do F4 DENTRO do
container da H100 (ordem do capitão: zero dependência de rota externa; o juiz
27B está no mesmo nó, acessível pelo gateway docker 172.17.0.1:8005 via env
REGUA_JUDGE_URL).

É o score_f2 (régua honesta do F0) com os caminhos apontados para o
scorepack local (todos os módulos — common/ruler/judge_regua/common_f0 —
viajam juntos neste diretório):

  scorepack/
    score_f4_container.py (este)
    score_f2.py common.py ruler.py judge_regua.py common_f0.py
    gabarito_384.jsonl
    root/data/gen -> symlink p/ /workspace/cemig/eval/data (gens do gen_f2)
    root/data/scored/  (saída)

Uso (dentro do container):
    cd /workspace/cemig/f4/scorepack && \
    python score_f4_container.py --tag f4-r32-ep1 --cell ranking --workers 16
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# raiz de dados do score_f2: root/data/{gen,scored}
ROOT = HERE / "root"
(ROOT / "data" / "scored").mkdir(parents=True, exist_ok=True)
GEN_LINK = ROOT / "data" / "gen"
if not GEN_LINK.exists():
    GEN_LINK.symlink_to("/workspace/cemig/eval/data")

os.environ.setdefault("REGUA_JUDGE_URL", "http://172.17.0.1:8005/v1")

import score_f2  # noqa: E402

score_f2.F2 = ROOT
score_f2.GABARITO = HERE / "gabarito_384.jsonl"

if __name__ == "__main__":
    # parquet default do score_f2 aponta p/ o HF cache local — no container o
    # parquet canônico está em /workspace/cemig/eval/normas_train.parquet
    if "--parquet" not in sys.argv:
        sys.argv += ["--parquet", "/workspace/cemig/eval/normas_train.parquet"]
    score_f2.main()
