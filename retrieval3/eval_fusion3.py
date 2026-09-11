#!/usr/bin/env python3
"""
eval_fusion3.py — Etapa 2+3: fusão multi-sinal (RRF de N rankings) — busca do teto.

Combina sinais complementares por RRF, calibrando k/pesos no DEV e medindo no holdout:
  s1 = BM25 gated-expandido (classificador + boost, índice com campo expansion)
  s2 = BM25 fala bruta no índice expandido (resgata quando o classificador erra a NR)
  s3 = denso EmbeddingGemma-768 (índice text+expansão)
  s4 = denso EmbeddingGemma-768 consultado com keywords TF-IDF do classificador

Uso (classifier/.venv):
    ../classifier/.venv/bin/python eval_fusion3.py \
        --dense-rank results/dense_rank_gemma768_exp.json \
        --dense-rank-kw results/dense_rank_gemma768_exp_kw.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bm25 import Bm25Retriever, HybridBm25, NrClassifier, apply_soft_boost
from dev_utils import load_devset
from eval_common import app_fts_query, evaluate, footer, header, load_holdout
from nr_taxonomy import NONE_LABEL
from rrf import TfidfKeywordExtractor, rrf_fuse

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--dense-rank", required=True)
    ap.add_argument("--dense-rank-kw", default="")
    ap.add_argument("--json-out", default=str(_HERE / "results" / "fusion3.json"))
    args = ap.parse_args()

    dev = load_devset()
    holdout = load_holdout("qa_v2")
    drank = json.loads(Path(args.dense_rank).read_text(encoding="utf-8"))
    drank_kw = json.loads(Path(args.dense_rank_kw).read_text(encoding="utf-8")) if args.dense_rank_kw else {}

    hy = HybridBm25(args.db, args.clf, weights=EXP_W)
    bm = Bm25Retriever(args.db, weights=EXP_W)

    def signals(it):
        s1 = hy.rank(it.question, 60)                          # BM25 gated-exp
        s2 = bm.search(app_fts_query(it.question), 60)         # BM25 bruta-exp
        s3 = drank.get(it.id, [])                              # denso
        s4 = drank_kw.get(it.id, [])                           # denso c/ keywords
        return s1, s2, s3, s4

    dev_sig = {it.id: signals(it) for it in dev}
    ho_sig = {it.id: signals(it) for it in holdout}

    # Combinações de sinais a testar (índices em [s1,s2,s3,s4])
    combos = {
        "s1+s3": [0, 2],
        "s1+s2+s3": [0, 1, 2],
        "s1+s3+s4": [0, 2, 3],
        "s1+s2+s3+s4": [0, 1, 2, 3],
    }
    if not drank_kw:
        combos = {k: v for k, v in combos.items() if 3 not in v}

    ks = [10.0, 30.0, 60.0]
    print("\n### CALIBRAÇÃO fusão multi-sinal no DEV ###")
    print(f"| {'combo':<14} | {'k':>4} | {'pesos':<16} | {'DEV R@2':>8} | {'DEV R@5':>8} |")
    print("|" + "-" * 16 + "|" + "-" * 6 + "|" + "-" * 18 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    best = None
    grid = []
    for cname, idxs in combos.items():
        # pesos: cada sinal 1.0, e uma variante com dense reforçado
        weight_variants = [tuple(1.0 for _ in idxs)]
        if 2 in idxs:
            wv = [1.0] * len(idxs)
            wv[idxs.index(2)] = 2.0
            weight_variants.append(tuple(wv))
        for k, wv in itertools.product(ks, weight_variants):
            def rank_fn(it, idxs=idxs, k=k, wv=wv):
                sig = dev_sig[it.id]
                return rrf_fuse([sig[i] for i in idxs], list(wv), k=k, limit=5)
            m = evaluate(dev, rank_fn)
            grid.append({"combo": cname, "k": k, "w": wv, "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
            print(f"| {cname:<14} | {k:>4.0f} | {str(wv):<16} | {m.recall_at_2:>7.1f}% | {m.recall_at_5:>7.1f}% |")
            key = (m.recall_at_2, m.recall_at_5)
            if best is None or key > best[0]:
                best = (key, cname, idxs, k, wv)

    _, bc, bidx, bk, bwv = best
    print(f"\nMelhor no DEV: {bc} k={bk} pesos={bwv} (R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    # --- GATE holdout ---
    header("ETAPA 2+3 — FUSÃO MULTI-SINAL — GATE holdout 151")
    m_hy = evaluate(holdout, lambda it: hy.rank(it.question, 5))
    print(m_hy.line("BM25 gated-exp (ref Etapa 1)"))

    def best_fn(it):
        sig = ho_sig[it.id]
        return rrf_fuse([sig[i] for i in bidx], list(bwv), k=bk, limit=5)
    m_best = evaluate(holdout, best_fn)
    print(m_best.line(f"Fusão {bc} (dev-cal k={bk:.0f})"))

    # Reporta também todas as combos com k=60 pesos iguais (referência)
    ref_rows = {}
    for cname, idxs in combos.items():
        def rf(it, idxs=idxs):
            sig = ho_sig[it.id]
            return rrf_fuse([sig[i] for i in idxs], [1.0] * len(idxs), k=60.0, limit=5)
        m = evaluate(holdout, rf)
        ref_rows[cname] = m.as_dict()
        print(m.line(f"Fusão {cname} (k=60 iguais)"))
    footer()

    out = {"dev_grid": grid, "best_dev": {"combo": bc, "k": bk, "w": list(bwv)},
           "gate_holdout_151": {"bm25_gated_exp": m_hy.as_dict(),
                                "best_dev_cal": m_best.as_dict(),
                                "ref_k60": ref_rows}}
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
