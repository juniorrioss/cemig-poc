#!/usr/bin/env python3
"""
sweep_frenteb.py — FRENTE B: isola cada componente da expansão v4 na fusão (régua decisiva).

Mede R@2/R@5 nas 151 combinando:
  BM25   : índice v3 (index_hf_36nr_exp.db) ou v4 (index_hf_36nr_expv4.db)
  dense  : caches v3 (bin_*) ou v4 (densev4_*), independentes por sinal (text/exp)
Reusa o classificador baseline e o gate v3. RRF k=30 pesos iguais (config vencedora v3).
O objetivo é achar QUAL parte da v4 ajuda e qual atrapalha (ex.: ASR no denso).

Uso (classifier/.venv): ../classifier/.venv/bin/python sweep_frenteb.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import HybridBm25  # noqa: E402
from eval_common import evaluate, load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
V3_DB = str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db")
V4_DB = str(_HERE / "indices" / "index_hf_36nr_expv4.db")

DENSE = {
    "v3_text": _ROOT / "retrieval3" / "results" / "bin_text_rank.json",
    "v3_exp": _ROOT / "retrieval3" / "results" / "bin_exponly_rank.json",
    "v4_text": _HERE / "results" / "densev4_text_rank.json",
    "v4_exp": _HERE / "results" / "densev4_exp_rank.json",
}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> None:
    ho = load_holdout("qa_v2")
    dense = {k: load(v) for k, v in DENSE.items()}
    hy_v3 = HybridBm25(V3_DB, CLF, weights=EXP_W)
    hy_v4 = HybridBm25(V4_DB, CLF, weights=EXP_W)

    # cache s1 por índice (o gate independe do denso)
    s1_v3 = {it.id: hy_v3.rank(it.question, 60) for it in ho}
    s1_v4 = {it.id: hy_v4.rank(it.question, 60) for it in ho}

    configs = {
        "BASE v3 (bm v3 + dense v3)": ("v3", "v3_text", "v3_exp"),
        "bm v4 + dense v3": ("v4", "v3_text", "v3_exp"),
        "bm v3 + dense v4": ("v3", "v4_text", "v4_exp"),
        "bm v4 + dense v4": ("v4", "v4_text", "v4_exp"),
        "bm v4 + dense v4text + v3exp": ("v4", "v4_text", "v3_exp"),
        "bm v4 + dense v3text + v4exp": ("v4", "v3_text", "v4_exp"),
    }

    print("=" * 84)
    print("  FRENTE B — componentes da expansão v4 na FUSÃO (holdout 151, RRF k=30)")
    print("=" * 84)
    print(f"| {'config':<34} | {'R@1':>5} | {'R@2':>5} | {'R@5':>5} | {'MRR':>6} |")
    print("|" + "-" * 36 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 8 + "|")
    out: Dict[str, Dict] = {}
    for name, (bm, dt, de) in configs.items():
        s1c = s1_v3 if bm == "v3" else s1_v4
        dtc, dec = dense[dt], dense[de]

        def fusion(it, s1c=s1c, dtc=dtc, dec=dec):
            return rrf_fuse([s1c[it.id], dtc.get(it.id, []), dec.get(it.id, [])],
                            [1.0, 1.0, 1.0], k=30.0, limit=5)
        m = evaluate(ho, fusion)
        out[name] = m.as_dict()
        print(f"| {name:<34} | {m.recall_at_1:>4.1f}% | {m.recall_at_2:>4.1f}% | "
              f"{m.recall_at_5:>4.1f}% | {m.mrr:>6.4f} |")
    print("=" * 84)
    Path(_HERE / "results" / "sweep_frenteb.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Salvo em results/sweep_frenteb.json")


if __name__ == "__main__":
    main()
