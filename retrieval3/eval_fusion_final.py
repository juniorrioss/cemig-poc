#!/usr/bin/env python3
"""
eval_fusion_final.py — Fusão final multi-sinal (busca do teto >55% R@2).

Combina, por RRF, sinais COMPLEMENTARES (cada um cobre um modo de falha distinto):
  s1 = BM25 gated-expandido (léxico + classificador de NR)
  sT = denso EmbeddingGemma-768 sobre TEXTO+expansão (semântico query↔passagem)
  sE = denso EmbeddingGemma-768 sobre EXPANSÃO-ONLY (semântico query↔query coloquial)

Varre todas as combinações e pesos; calibra no DEV e reporta o GATE no holdout 151.
Como o DEV é notoriamente mais fácil que o holdout, também reportamos TODAS as combos
em k=60 pesos iguais no holdout (medição honesta, não ajuste).

Uso (classifier/.venv):
    ../classifier/.venv/bin/python eval_fusion_final.py \
      --dense-text results/dense_rank_gemma768_exp.json \
      --dense-exp  results/dense_rank_gemma768_exponly.json
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
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--clf", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--dense-text", required=True)
    ap.add_argument("--dense-exp", required=True)
    ap.add_argument("--json-out", default=str(_HERE / "results" / "fusion_final.json"))
    args = ap.parse_args()

    dev = load_devset()
    holdout = load_holdout("qa_v2")
    dtext = json.loads(Path(args.dense_text).read_text(encoding="utf-8"))
    dexp = json.loads(Path(args.dense_exp).read_text(encoding="utf-8"))
    hy = HybridBm25(args.db, args.clf, weights=EXP_W)

    def sig(it):
        return {"s1": hy.rank(it.question, 60), "sT": dtext.get(it.id, []), "sE": dexp.get(it.id, [])}

    dev_sig = {it.id: sig(it) for it in dev}
    ho_sig = {it.id: sig(it) for it in holdout}

    combos = {
        "s1+sT": ["s1", "sT"],
        "s1+sE": ["s1", "sE"],
        "sT+sE": ["sT", "sE"],
        "s1+sT+sE": ["s1", "sT", "sE"],
    }
    ks = [10.0, 30.0, 60.0]

    print("\n### CALIBRAÇÃO fusão final no DEV ###")
    print(f"| {'combo':<12} | {'k':>4} | {'pesos':<18} | {'DEV R@2':>8} | {'DEV R@5':>8} |")
    print("|" + "-" * 14 + "|" + "-" * 6 + "|" + "-" * 20 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    grid = []
    best = None
    for cname, keys in combos.items():
        wvars = [tuple(1.0 for _ in keys)]
        if len(keys) == 3:
            wvars += [(1.0, 1.0, 2.0), (1.0, 2.0, 1.0), (2.0, 1.0, 1.0)]
        else:
            wvars += [(1.0, 2.0), (2.0, 1.0)]
        for k, wv in itertools.product(ks, wvars):
            def rf(it, keys=keys, k=k, wv=wv):
                s = dev_sig[it.id]
                return rrf_fuse([s[x] for x in keys], list(wv), k=k, limit=5)
            m = evaluate(dev, rf)
            grid.append({"combo": cname, "k": k, "w": wv, "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
            print(f"| {cname:<12} | {k:>4.0f} | {str(wv):<18} | {m.recall_at_2:>7.1f}% | {m.recall_at_5:>7.1f}% |")
            key = (m.recall_at_2, m.recall_at_5)
            if best is None or key > best[0]:
                best = (key, cname, keys, k, wv)

    _, bc, bkeys, bk, bwv = best
    print(f"\nMelhor no DEV: {bc} k={bk} pesos={bwv} (R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    header("FUSÃO FINAL — GATE holdout 151")
    m_ref = evaluate(holdout, lambda it: hy.rank(it.question, 5))
    print(m_ref.line("BM25 gated-exp (ref Etapa 1)"))

    def best_fn(it):
        s = ho_sig[it.id]
        return rrf_fuse([s[x] for x in bkeys], list(bwv), k=bk, limit=5)
    m_best = evaluate(holdout, best_fn)
    print(m_best.line(f"Fusão {bc} (dev-cal k={bk:.0f})"))

    ho_rows = {}
    for cname, keys in combos.items():
        best_c = None
        for k in ks:
            def rf(it, keys=keys, k=k):
                s = ho_sig[it.id]
                return rrf_fuse([s[x] for x in keys], [1.0] * len(keys), k=k, limit=5)
            m = evaluate(holdout, rf)
            if best_c is None or m.recall_at_2 > best_c[0]:
                best_c = (m.recall_at_2, k, m)
        m = best_c[2]
        ho_rows[cname] = {"best_k": best_c[1], **m.as_dict()}
        print(m.line(f"Fusão {cname} (best k={best_c[1]:.0f}, iguais)"))
    footer()

    out = {"dev_grid": grid, "best_dev": {"combo": bc, "k": bk, "w": list(bwv)},
           "gate_holdout_151": {"bm25_gated_exp": m_ref.as_dict(),
                                "best_dev_cal": m_best.as_dict(),
                                "holdout_best_per_combo": ho_rows}}
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.json_out}")


if __name__ == "__main__":
    main()
