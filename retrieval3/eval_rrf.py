#!/usr/bin/env python3
"""
eval_rrf.py — Etapa 2: calibra RRF no dev-set e mede o GATE no holdout (151).

Protocolo honesto:
  - CALIBRAÇÃO (k do RRF, pesos das consultas, uso da q3) só no dev-set sintético;
  - GATE reportado no holdout 151 (medição, não ajuste).

Uso:
    python3 retrieval3/eval_rrf.py --db corpus/index_hf_36nr.db
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bm25 import Bm25Retriever, HybridBm25
from dev_utils import load_devset
from eval_common import evaluate, footer, header, load_holdout
from rrf import RrfMultiQuery

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--model", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--json-out", default=str(_HERE / "results" / "rrf.json"))
    args = ap.parse_args()

    dev = load_devset()
    holdout = load_holdout("qa_v2")

    # --- Calibração no DEV: grid de k e pesos ---
    print("\n### CALIBRAÇÃO RRF no dev-set (nunca holdout) ###")
    print(f"| {'k':>4} | {'w(raw,boost,kw)':<16} | {'q3':>3} | {'DEV R@2':>8} | {'DEV R@5':>8} |")
    print("|" + "-" * 6 + "|" + "-" * 18 + "|" + "-" * 5 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    ks = [10.0, 30.0, 60.0]
    # anchor = r1 (gated forte); complementares r2 (fala bruta), r3 (keywords)
    weight_opts = [(1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (3.0, 1.0, 0.5), (3.0, 0.5, 0.5)]
    best = None
    grid = []
    for k, w, use_q3 in itertools.product(ks, weight_opts, [True, False]):
        r = RrfMultiQuery(args.db, args.model, rrf_k=k, weights=w, use_q3=use_q3)
        m = evaluate(dev, lambda it: r.rank(it.question, 5))
        r.bm.close()
        grid.append({"k": k, "weights": w, "use_q3": use_q3,
                     "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
        print(f"| {k:>4.0f} | {str(w):<16} | {str(use_q3):>3} | {m.recall_at_2:>7.1f}% | {m.recall_at_5:>7.1f}% |")
        key = (m.recall_at_2, m.recall_at_5)
        if best is None or key > best[0]:
            best = (key, k, w, use_q3)

    _, bk, bw, bq3 = best
    print(f"\nMelhor no DEV: k={bk} weights={bw} q3={bq3} "
          f"(DEV R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    # --- GATE no holdout 151 (medição honesta) ---
    header("ETAPA 2 — RRF MULTI-QUERY — GATE holdout 151")
    bm = Bm25Retriever(args.db)
    m_base = evaluate(holdout, lambda it: bm.search_raw(it.question, 5))
    print(m_base.line("BM25 fala bruta"))
    hy = HybridBm25(args.db, args.model)
    m_hy = evaluate(holdout, lambda it: hy.rank(it.question, 5))
    print(m_hy.line("Híbrido gated (Etapa 0)"))
    rrf_best = RrfMultiQuery(args.db, args.model, rrf_k=bk, weights=bw, use_q3=bq3)
    m_rrf = evaluate(holdout, lambda it: rrf_best.rank(it.question, 5))
    print(m_rrf.line(f"RRF multi-query (k={bk:.0f}, q3={bq3})"))
    # Também reporta config canônica k=60 pesos iguais (referência)
    rrf_ref = RrfMultiQuery(args.db, args.model, rrf_k=60.0, weights=(1.0, 1.0, 1.0), use_q3=True)
    m_ref = evaluate(holdout, lambda it: rrf_ref.rank(it.question, 5))
    print(m_ref.line("RRF ref (k=60, pesos iguais)"))
    footer()

    out = {
        "dev_grid": grid,
        "best_dev": {"k": bk, "weights": list(bw), "use_q3": bq3,
                     "dev_r2": best[0][0], "dev_r5": best[0][1]},
        "gate_holdout_151": {
            "baseline_raw": m_base.as_dict(),
            "hybrid_gated": m_hy.as_dict(),
            "rrf_best": m_rrf.as_dict(),
            "rrf_ref_k60": m_ref.as_dict(),
        },
    }
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
