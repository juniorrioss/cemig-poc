#!/usr/bin/env python3
"""
eval_baseline.py — Etapa 0: reproduz o baseline do main no harness v3.

Confirma paridade com os números conhecidos (AGENTS.md):
  - Baseline fala bruta (app path): R@2 ~14% nas 151;
  - Híbrido classificador-NR + BM25 gated: R@2 ~29.8% / R@5 ~41.1% nas 151.

Uso:
    python3 retrieval3/eval_baseline.py --db corpus/index_hf_36nr.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25 import Bm25Retriever, HybridBm25
from eval_common import evaluate, footer, header, load_holdout

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--model", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--subset", default="qa_v2", choices=["qa_v2", "all"])
    ap.add_argument("--json-out", default=str(Path(__file__).resolve().parent / "results" / "baseline.json"))
    args = ap.parse_args()

    items = load_holdout(args.subset)
    bm = Bm25Retriever(args.db)
    hy = HybridBm25(args.db, args.model)

    header(f"ETAPA 0 — BASELINE ({args.subset}, {len(items)} perguntas) — {Path(args.db).name}")
    m_base = evaluate(items, lambda it: bm.search_raw(it.question, 5))
    print(m_base.line("BM25 fala bruta (app path)"))
    m_hy = evaluate(items, lambda it: hy.rank(it.question, 5))
    print(m_hy.line("Híbrido classificador-NR + BM25 gated"))
    footer()

    out = {
        "subset": args.subset, "n": len(items), "db": Path(args.db).name,
        "baseline_raw": m_base.as_dict(),
        "hybrid_gated": m_hy.as_dict(),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
