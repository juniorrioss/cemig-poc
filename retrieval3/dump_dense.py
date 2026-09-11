#!/usr/bin/env python3
"""
dump_dense.py — Exporta rankings densos (top-K) do dev-set e holdout para cache JSON.

Roda no ambiente .venv-train (py3.10 + torch cu128 + sentence-transformers + sqlite-vec).
O cache permite avaliar a FUSÃO BM25+denso no ambiente do classificador (sklearn 1.9.1,
py3.12) sem precisar de torch lá. O holdout entra só como MEDIÇÃO (nada é ajustado nele).

Saída: {query_id: [{id,doc,section,title,text,score}, ...]} para dev e holdout.

Uso:
    ../.venv-train/bin/python dump_dense.py --dense-db indices/dense_minilm.db \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --out results/dense_rank_minilm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "classifier"))

from dense import DenseRetriever  # noqa: E402


def _load_qa(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        items.append({"id": r["id"], "question": r["question"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-db", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dim-trunc", type=int, default=0)
    ap.add_argument("--prefix-kind", default=None)
    ap.add_argument("--query-prompt-name", default=None)
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--kw-map", default="", help="JSON {query_id: keywords} p/ consultar denso "
                    "com keywords TF-IDF (sinal s4); gerado no classifier/.venv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dense = DenseRetriever(args.dense_db, args.model, dim_trunc=args.dim_trunc or None,
                          prefix_kind=args.prefix_kind, query_prompt_name=args.query_prompt_name)
    kw_map = json.loads(Path(args.kw_map).read_text(encoding="utf-8")) if args.kw_map else {}

    dev = _load_qa(_HERE / "data" / "dev_set.jsonl")
    holdout = _load_qa(_ROOT / "corpus" / "qa_pairs_v2.jsonl")
    smoke = _load_qa(_ROOT / "bench" / "data" / "smoke_qa_20.jsonl")

    cache = {}
    for it in dev + holdout + smoke:
        query = kw_map.get(it["id"]) or it["question"]
        res = dense.search(query, args.topk)
        cache[it["id"]] = [
            {"id": r["id"], "doc": r["doc"], "section": r["section"],
             "title": r["title"], "text": r["text"], "score": r["score"]}
            for r in res
        ]
    dense.close()
    Path(args.out).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"dumped {len(cache)} queries -> {args.out}")


if __name__ == "__main__":
    main()
