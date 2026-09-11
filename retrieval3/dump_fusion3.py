#!/usr/bin/env python3
"""
dump_fusion3.py — Exporta candidatos da fusão FINAL 3-sinais (top-N) p/ reranking.

Fusão = RRF(s1 BM25-gated-exp, sT denso texto+exp, sE denso expansão-only), k=30.
Grava top-N com texto + expansão coloquial (p/ reranker). Roda no classifier/.venv.

Uso:
    ../classifier/.venv/bin/python dump_fusion3.py \
        --dense-text results/dense_rank_gemma768_exp.json \
        --dense-exp results/dense_rank_gemma768_exponly.json \
        --rrf-k 30 --topn 10 --out results/fusion3_cand.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25 import HybridBm25
from dev_utils import load_devset
from eval_common import load_holdout
from rrf import rrf_fuse

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--dense-text", required=True)
    ap.add_argument("--dense-exp", required=True)
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--rrf-k", type=float, default=30.0)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    exp = {}
    for line in Path(args.expansions).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        exp[r["id"]] = " ".join(r.get("perguntas", []) + r.get("sinonimos", []))

    dtext = json.loads(Path(args.dense_text).read_text(encoding="utf-8"))
    dexp = json.loads(Path(args.dense_exp).read_text(encoding="utf-8"))
    hy = HybridBm25(args.db, args.clf, weights=EXP_W)

    dev = load_devset()
    holdout = load_holdout("all")
    cache = {}
    for it in dev + holdout:
        fused = rrf_fuse([hy.rank(it.question, 60), dtext.get(it.id, []), dexp.get(it.id, [])],
                         [1.0, 1.0, 1.0], k=args.rrf_k, limit=args.topn)
        cache[it.id] = {"question": it.question,
                        "cands": [{"id": r["id"], "doc": r["doc"], "section": r["section"],
                                   "title": r["title"], "text": r["text"],
                                   "expansion": exp.get(r["id"], "")} for r in fused]}
    Path(args.out).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"dumped {len(cache)} queries (top-{args.topn}) -> {args.out}")


if __name__ == "__main__":
    main()
