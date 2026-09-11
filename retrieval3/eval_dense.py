#!/usr/bin/env python3
"""
eval_dense.py — Etapa 3: avalia denso puro e híbrido BM25+denso (RRF), via cache.

Consome o cache de rankings densos (dump_dense.py) para não depender de torch neste
ambiente (classifier/.venv, sklearn 1.9.1). Mede no holdout 151:
  - denso puro;
  - BM25 híbrido gated (referência);
  - híbrido RRF(BM25-gated, denso) — k/pesos calibrados no DEV.

Uso (no classifier/.venv):
    ../classifier/.venv/bin/python eval_dense.py \
        --dense-rank results/dense_rank_minilm.json \
        --meta results/dense_minilm_meta.json --label MiniLM-L12
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bm25 import HybridBm25
from dev_utils import load_devset
from eval_common import evaluate, footer, header, load_holdout
from rrf import rrf_fuse

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-rank", required=True, help="cache JSON de dump_dense.py")
    ap.add_argument("--bm25-db", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--bm25-weights", default="", help="CSV de pesos BM25 (ex.: 1.5,3,2,1,1 p/ expandido)")
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--meta", default="")
    ap.add_argument("--label", default="dense")
    ap.add_argument("--json-out", default=str(_HERE / "results" / "dense.json"))
    args = ap.parse_args()

    dev = load_devset()
    holdout = load_holdout("qa_v2")
    drank = json.loads(Path(args.dense_rank).read_text(encoding="utf-8"))
    bmw = tuple(float(x) for x in args.bm25_weights.split(",")) if args.bm25_weights else None
    hy = HybridBm25(args.bm25_db, args.clf, weights=bmw)

    def dn(qid):
        return drank.get(qid, [])

    # --- Calibração RRF(bm25-gated, denso) no DEV ---
    print(f"\n### CALIBRAÇÃO RRF BM25+denso ({args.label}) no DEV ###")
    print(f"| {'k':>4} | {'w(bm,dense)':<12} | {'DEV R@2':>8} | {'DEV R@5':>8} |")
    print("|" + "-" * 6 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    dev_bm = {it.id: hy.rank(it.question, 60) for it in dev}
    ks = [10.0, 30.0, 60.0]
    wopts = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 0.5)]
    best = None
    dev_grid = []
    for k, w in itertools.product(ks, wopts):
        def rank_fn(it, k=k, w=w):
            return rrf_fuse([dev_bm[it.id], dn(it.id)], list(w), k=k, limit=5)
        m = evaluate(dev, rank_fn)
        dev_grid.append({"k": k, "w": w, "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
        print(f"| {k:>4.0f} | {str(w):<12} | {m.recall_at_2:>7.1f}% | {m.recall_at_5:>7.1f}% |")
        key = (m.recall_at_2, m.recall_at_5)
        if best is None or key > best[0]:
            best = (key, k, w)
    _, bk, bw = best
    print(f"\nMelhor no DEV: k={bk} w={bw} (R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    # --- GATE no holdout 151 ---
    header(f"ETAPA 3 — DENSO {args.label} — GATE holdout 151")
    m_dense = evaluate(holdout, lambda it: dn(it.id)[:5])
    print(m_dense.line(f"Denso puro ({args.label})"))
    m_hy = evaluate(holdout, lambda it: hy.rank(it.question, 5))
    print(m_hy.line("BM25 híbrido gated (ref)"))
    ho_bm = {it.id: hy.rank(it.question, 60) for it in holdout}
    m_rrf = evaluate(holdout, lambda it: rrf_fuse([ho_bm[it.id], dn(it.id)], list(bw), k=bk, limit=5))
    print(m_rrf.line(f"Híbrido BM25+denso RRF (k={bk:.0f}, w={bw})"))
    m_ref = evaluate(holdout, lambda it: rrf_fuse([ho_bm[it.id], dn(it.id)], [1.0, 1.0], k=60.0, limit=5))
    print(m_ref.line("Híbrido BM25+denso RRF (k=60, iguais)"))
    footer()

    cost = {}
    if args.meta and Path(args.meta).exists():
        cost = json.loads(Path(args.meta).read_text())

    out = {
        "label": args.label,
        "dev_grid": dev_grid, "best_dev": {"k": bk, "w": list(bw)},
        "gate_holdout_151": {
            "dense_pure": m_dense.as_dict(),
            "bm25_hybrid": m_hy.as_dict(),
            "rrf_best": m_rrf.as_dict(),
            "rrf_ref_k60": m_ref.as_dict(),
        },
        "mobile_cost": cost,
    }
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
