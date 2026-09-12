#!/usr/bin/env python3
"""
eval_expansion_v4.py — FRENTE B: mede o INCREMENTO isolado da expansão v4 (BM25 gated-exp).

Compara, no holdout 151, o índice da v3 (expansão v3) vs o índice v4 (expansão v3+v4),
usando o MESMO classificador baseline e o MESMO gate (réplica do app). Isola o efeito da
expansão mais rica antes de olhar a fusão densa. O holdout NUNCA calibra nada.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python eval_expansion_v4.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import Bm25Retriever, HybridBm25  # noqa: E402
from eval_common import app_fts_query, evaluate, footer, header, load_holdout  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-db", default=str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--v4-db", default=str(_HERE / "indices" / "index_hf_36nr_expv4.db"))
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--json-out", default=str(_HERE / "results" / "expansion_v4.json"))
    args = ap.parse_args()

    ho = load_holdout("qa_v2")

    def bm_raw(db):
        bm = Bm25Retriever(db, weights=EXP_W)
        return lambda it: bm.search(app_fts_query(it.question), 5)

    def gated(db):
        hy = HybridBm25(db, args.clf, weights=EXP_W)
        return lambda it: hy.rank(it.question, 5)

    header("FRENTE B — INCREMENTO DA EXPANSÃO v4 (holdout 151, BM25)")
    out = {}
    for label, db in [("v3 (perguntas+sinonimos)", args.v3_db),
                      ("v4 (+verbaliz+asr+sinon2)", args.v4_db)]:
        m_raw = evaluate(ho, bm_raw(db))
        m_gat = evaluate(ho, gated(db))
        print(m_raw.line(f"BM25 bruta [{label}]"))
        print(m_gat.line(f"BM25 gated [{label}]"))
        out[label] = {"raw": m_raw.as_dict(), "gated": m_gat.as_dict()}
    footer()
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {args.json_out}")


if __name__ == "__main__":
    main()
