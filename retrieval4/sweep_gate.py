#!/usr/bin/env python3
"""
sweep_gate.py — FRENTE A (calibração): threshold de confiança, top-k do boost e k do RRF.

O brief pergunta: um top-3 com boost mais suave bate o top-2 atual? E a calibração de
confiança (os gates dependem dela)? Testamos variando SOBRE O CLASSIFICADOR BASELINE
(sem novos dados), calibrando no DEV sintético (nunca no holdout) e reportando o GATE
decisivo nas 151. A régua é R@2/R@5 da fusão 3-sinais.

Parâmetros varridos:
  - conf_thr : limiar do filtro-duro (0.4/0.5/0.6/0.7)
  - boost_k  : nº de NRs que recebem boost suave (2 ou 3)
  - factor   : intensidade do boost suave (5x/3x)
  - rrf_k    : k da fusão (10/30/60)

Uso (classifier/.venv): ../classifier/.venv/bin/python sweep_gate.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import Bm25Retriever, NrClassifier, apply_soft_boost  # noqa: E402
from eval_common import app_fts_query, evaluate, load_holdout  # noqa: E402
from dev_utils import load_devset  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from nr_taxonomy import NONE_LABEL  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
DB = str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db")
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
DTEXT = _ROOT / "retrieval3" / "results" / "bin_text_rank.json"
DEXP = _ROOT / "retrieval3" / "results" / "bin_exponly_rank.json"


class GatedS1:
    """s1 (BM25 gated-exp) com gate parametrizável: thr, top-k do boost, factor."""

    def __init__(self, db, clf_path, conf_thr, boost_k, factor):
        self.bm = Bm25Retriever(db, weights=EXP_W)
        self.clf = NrClassifier(clf_path)
        self.conf_thr = conf_thr
        self.boost_k = boost_k
        self.factor = factor

    def rank(self, question, limit=60):
        topk, scores = self.clf.predict([question], k=self.boost_k)
        preds = [p for p in topk[0] if p != NONE_LABEL]
        p1 = float(scores[0].max())
        q = app_fts_query(question)
        if p1 >= self.conf_thr and preds:
            res = self.bm.search(q, limit, doc_filter=preds[:1])
            if len(res) < limit:
                extra = self.bm.search(q, limit)
                seen = {r["id"] for r in res}
                res += [r for r in extra if r["id"] not in seen]
            return res[:limit]
        pool = self.bm.search(q, limit)
        return apply_soft_boost(pool, preds[:self.boost_k], self.factor)[:limit]


def main() -> None:
    dtext = json.loads(DTEXT.read_text(encoding="utf-8"))
    dexp = json.loads(DEXP.read_text(encoding="utf-8"))
    dev = load_devset()
    ho = load_holdout("qa_v2")

    thrs = [0.4, 0.5, 0.6, 0.7]
    boost_ks = [2, 3]
    factors = [5.0, 3.0]
    rrf_ks = [10.0, 30.0, 60.0]

    # Calibra no DEV: escolhe a config de maior R@2 (desempate R@5) na fusão.
    print("### CALIBRAÇÃO no DEV (fusão 3-sinais) ###")
    grid = []
    best = None
    for thr, bk, fac, rk in itertools.product(thrs, boost_ks, factors, rrf_ks):
        s1 = GatedS1(DB, CLF, thr, bk, fac)

        def fusion(it, s1=s1, rk=rk):
            return rrf_fuse([s1.rank(it.question, 60), dtext.get(it.id, []), dexp.get(it.id, [])],
                            [1.0, 1.0, 1.0], k=rk, limit=5)
        m = evaluate(dev, fusion)
        grid.append({"thr": thr, "boost_k": bk, "factor": fac, "rrf_k": rk,
                     "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
        key = (m.recall_at_2, m.recall_at_5)
        if best is None or key > best[0]:
            best = (key, thr, bk, fac, rk)
    _, bt, bbk, bf, brk = best
    print(f"Melhor no DEV: thr={bt} boost_k={bbk} factor={bf} rrf_k={brk} "
          f"(R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    # GATE no holdout 151: config dev-vencedora + a config v3 de referência.
    print("\n### GATE holdout 151 (DECISIVA) ###")
    rows = {}
    configs = {
        "v3_ref (thr0.5 k2 f5 rrf30)": (0.5, 2, 5.0, 30.0),
        f"dev_best (thr{bt} k{bbk} f{bf} rrf{brk:.0f})": (bt, bbk, bf, brk),
    }
    # inclui também alguns pontos informativos (top-3 boost e thr alto)
    configs["top3 (thr0.5 k3 f5 rrf30)"] = (0.5, 3, 5.0, 30.0)
    configs["thr0.6 k2 f5 rrf30"] = (0.6, 2, 5.0, 30.0)
    configs["thr0.7 k2 f5 rrf30"] = (0.7, 2, 5.0, 30.0)
    for name, (thr, bk, fac, rk) in configs.items():
        s1 = GatedS1(DB, CLF, thr, bk, fac)

        def fusion(it, s1=s1, rk=rk):
            return rrf_fuse([s1.rank(it.question, 60), dtext.get(it.id, []), dexp.get(it.id, [])],
                            [1.0, 1.0, 1.0], k=rk, limit=5)
        m = evaluate(ho, fusion)
        rows[name] = m.as_dict()
        print(f"  {name:<34} R@2={m.recall_at_2:>5.1f}%  R@5={m.recall_at_5:>5.1f}%  MRR={m.mrr:.4f}")

    Path(_HERE / "results" / "sweep_gate.json").write_text(json.dumps(
        {"dev_grid": grid, "dev_best": {"thr": bt, "boost_k": bbk, "factor": bf, "rrf_k": brk},
         "holdout_151": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em results/sweep_gate.json")


if __name__ == "__main__":
    main()
