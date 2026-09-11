#!/usr/bin/env python3
"""
eval_expansion.py — Etapa 1: calibra o peso do campo `expansion` e mede o GATE.

Calibra o peso do 5º campo (expansion) no dev-set sintético e reporta o gate no
holdout 151. Testa baseline + híbrido gated, ambos sobre o índice expandido, e
varre pesos do campo expansion (0.5 / 1 / 2 do texto, além de valores extras).

Uso:
    python3 retrieval3/eval_expansion.py --db retrieval3/indices/index_hf_36nr_exp.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25 import Bm25Retriever, HybridBm25
from dev_utils import load_devset
from eval_common import evaluate, footer, header, load_holdout

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent

# Pesos base dos 4 campos originais (produção) + expansion calibrável no 5º
BASE4 = (1.5, 3.0, 2.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--model", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--json-out", default=str(_HERE / "results" / "expansion.json"))
    args = ap.parse_args()

    dev = load_devset()
    holdout = load_holdout("qa_v2")

    # Pesos do campo expansion a testar (0 = como se não existisse)
    exp_weights = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    print("\n### CALIBRAÇÃO peso do campo expansion no DEV (baseline bruta) ###")
    print(f"| {'w_exp':>6} | {'DEV R@2':>8} | {'DEV R@5':>8} |")
    print("|" + "-" * 8 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    dev_grid = []
    best_w = 0.0
    best_key = (-1.0, -1.0)
    for we in exp_weights:
        w = (*BASE4, we)
        bm = Bm25Retriever(args.db, weights=w)
        m = evaluate(dev, lambda it: bm.search_raw(it.question, 5))
        bm.close()
        dev_grid.append({"w_exp": we, "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
        print(f"| {we:>6.1f} | {m.recall_at_2:>7.1f}% | {m.recall_at_5:>7.1f}% |")
        key = (m.recall_at_2, m.recall_at_5)
        if key > best_key:
            best_key = key
            best_w = we
    print(f"\nMelhor w_exp no DEV: {best_w} (DEV R@2={best_key[0]:.1f}% R@5={best_key[1]:.1f}%)")

    # --- GATE no holdout 151 ---
    header("ETAPA 1 — EXPANSÃO DE DOCUMENTO — GATE holdout 151")
    results = {}
    for label, we in [("sem expansão (w=0)", 0.0), (f"expansão w={best_w}", best_w), ("expansão w=2.0", 2.0)]:
        w = (*BASE4, we)
        bm = Bm25Retriever(args.db, weights=w)
        m_raw = evaluate(holdout, lambda it: bm.search_raw(it.question, 5))
        print(m_raw.line(f"BM25 bruta [{label}]"))
        bm.close()
        hy = HybridBm25(args.db, args.model, weights=w)
        m_hy = evaluate(holdout, lambda it: hy.rank(it.question, 5))
        print(m_hy.line(f"Híbrido gated [{label}]"))
        hy.bm.close()
        results[f"w{we}"] = {"raw": m_raw.as_dict(), "hybrid": m_hy.as_dict()}
    footer()

    out = {"dev_grid": dev_grid, "best_w_dev": best_w, "gate_holdout_151": results}
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
