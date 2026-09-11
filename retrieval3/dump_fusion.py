#!/usr/bin/env python3
"""
dump_fusion.py — Etapa 4 (prep): exporta candidatos da fusão (top-N) p/ reranking.

Roda no classifier/.venv (sklearn 1.9.1). Para dev e holdout, gera a fusão RRF
(BM25-gated-expandido + denso EmbeddingGemma) e grava os top-N candidatos com texto
completo, p/ o reranker (que roda no .venv-train com torch) reordenar.

Uso:
    ../classifier/.venv/bin/python dump_fusion.py \
        --dense-rank results/dense_rank_gemma_exp.json \
        --bm25-db indices/index_hf_36nr_exp.db --bm25-weights 1.5,3,2,1,1 \
        --rrf-k 10 --topn 10 --out results/fusion_cand.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-rank", required=True)
    ap.add_argument("--bm25-db", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--bm25-weights", default="1.5,3,2,1,1")
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--rrf-k", type=float, default=10.0)
    ap.add_argument("--w-bm", type=float, default=1.0)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"),
                    help="anexa a expansão coloquial ao texto do candidato (p/ reranker)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Carrega expansões (perguntas coloquiais + sinônimos) por chunk id.
    exp = {}
    exp_path = Path(args.expansions)
    if exp_path.exists():
        for line in exp_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp[r["id"]] = " ".join(r.get("perguntas", []) + r.get("sinonimos", []))

    drank = json.loads(Path(args.dense_rank).read_text(encoding="utf-8"))
    bmw = tuple(float(x) for x in args.bm25_weights.split(","))
    hy = HybridBm25(args.bm25_db, args.clf, weights=bmw)

    dev = load_devset()
    holdout = load_holdout("all")  # 151 + 20; a fusão não ajusta nada

    cache = {}
    for it in dev + holdout:
        fused = rrf_fuse([hy.rank(it.question, 60), drank.get(it.id, [])],
                         [args.w_bm, args.w_dense], k=args.rrf_k, limit=args.topn)
        cache[it.id] = {
            "question": it.question,
            "cands": [{"id": r["id"], "doc": r["doc"], "section": r["section"],
                       "title": r["title"], "text": r["text"],
                       "expansion": exp.get(r["id"], "")} for r in fused],
        }
    Path(args.out).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"dumped {len(cache)} queries (top-{args.topn}) -> {args.out}")


if __name__ == "__main__":
    main()
