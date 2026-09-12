#!/usr/bin/env python3
"""
calibrate_v4.py — Calibra RRF (k + pesos) do combo v4 vencedor no DEV; GATE no holdout.

Combo v4 (do sweep_frenteb): BM25 gated índice v4 + dense v3-text + dense v4-exp.
Calibra k∈{10,30,60} e pesos por sinal SÓ no dev-set sintético (nunca no holdout);
reporta o gate decisivo nas 151 (limpo) e sob ASR. Honesto: decisão ancorada no gate.

Precisa dos caches densos do DEV. Como o dev tem ids próprios, reencodamos os rankings
densos do dev com o mesmo servidor GGUF (via dense_v4 --split dev), se ausentes.

Uso (classifier/.venv, servidor GGUF em :8399):
  ../classifier/.venv/bin/python calibrate_v4.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import HybridBm25  # noqa: E402
from eval_common import evaluate, load_holdout  # noqa: E402
from dev_utils import load_devset  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from dense_v4 import DOC_PROMPT, QUERY_PROMPT, embed_batch, load_chunks, load_exp  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
V4_DB = str(_HERE / "indices" / "index_hf_36nr_expv4.db")
URL = "http://127.0.0.1:8399"


def dev_dense_ranks() -> Dict[str, Dict]:
    """Reencoda (ou carrega cache) os rankings densos do DEV: v3-text e v4-exp."""
    cache_t = _HERE / "results" / "densev4_dev_text_rank.json"
    cache_e = _HERE / "results" / "densev4_dev_exp_rank.json"
    if cache_t.exists() and cache_e.exists():
        return {"text": json.loads(cache_t.read_text()), "exp": json.loads(cache_e.read_text())}

    e3 = load_exp(_ROOT / "retrieval3" / "data" / "expansions.jsonl", "v3")
    e4 = load_exp(_HERE / "data" / "expansions_v4.jsonl", "v4")
    rows = load_chunks(str(_ROOT / "corpus" / "index_hf_36nr.db"))
    # v3-text usa a expansão v3; v4-exp usa a expansão v4-only
    text_v3 = []
    exp_v4 = []
    for r in rows:
        ev3 = e3.get(r["id"], "")
        ev4 = e4.get(r["id"], "")
        text_v3.append(r["text"] + " " + ev3 if ev3 else r["text"])
        exp_v4.append(ev4 or r["text"])

    def enc(texts):
        parts = []
        for i in range(0, len(texts), 32):
            parts.append(embed_batch(URL, [DOC_PROMPT + t for t in texts[i:i + 32]]))
        return np.vstack(parts)

    dt = enc(text_v3)
    de = enc(exp_v4)
    dev = load_devset()
    q = embed_batch(URL, [QUERY_PROMPT + it.question for it in dev])
    ids = np.array([r["id"] for r in rows])

    def ranks(qemb, demb):
        sims_all = qemb @ demb.T
        out = {}
        for qi, it in enumerate(dev):
            top = np.argsort(-sims_all[qi])[:60]
            out[it.id] = [{"id": int(ids[j]), "doc": rows[j]["doc"], "section": rows[j]["section"],
                           "title": rows[j]["title"], "text": rows[j]["text"],
                           "score": float(sims_all[qi][j])} for j in top]
        return out
    rt = ranks(q, dt)
    re_ = ranks(q, de)
    cache_t.write_text(json.dumps(rt, ensure_ascii=False))
    cache_e.write_text(json.dumps(re_, ensure_ascii=False))
    return {"text": rt, "exp": re_}


def main() -> None:
    dev = load_devset()
    ho = load_holdout("qa_v2")
    dev_dense = dev_dense_ranks()
    ho_dt = json.loads((_ROOT / "retrieval3" / "results" / "bin_text_rank.json").read_text())
    ho_de = json.loads((_HERE / "results" / "densev4_exp_rank.json").read_text())

    hy = HybridBm25(V4_DB, CLF, weights=EXP_W)
    dev_s1 = {it.id: hy.rank(it.question, 60) for it in dev}
    ho_s1 = {it.id: hy.rank(it.question, 60) for it in ho}

    ks = [10.0, 30.0, 60.0]
    wvars = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 1, 2), (1, 2, 2)]

    best = None
    grid = []
    for k, wv in itertools.product(ks, wvars):
        def rf(it, k=k, wv=wv):
            return rrf_fuse([dev_s1[it.id], dev_dense["text"].get(it.id, []),
                             dev_dense["exp"].get(it.id, [])], list(wv), k=k, limit=5)
        m = evaluate(dev, rf)
        grid.append({"k": k, "w": wv, "dev_r2": m.recall_at_2, "dev_r5": m.recall_at_5})
        key = (m.recall_at_2, m.recall_at_5)
        if best is None or key > best[0]:
            best = (key, k, wv)
    _, bk, bwv = best
    print(f"Melhor no DEV: k={bk} pesos={bwv} (R@2={best[0][0]:.1f}% R@5={best[0][1]:.1f}%)")

    print("\n### GATE holdout 151 (limpo) ###")
    rows = {}
    for name, (k, wv) in {"dev_best": (bk, bwv), "v3-ref (k30 iguais)": (30.0, (1, 1, 1))}.items():
        def rf(it, k=k, wv=wv):
            return rrf_fuse([ho_s1[it.id], ho_dt.get(it.id, []), ho_de.get(it.id, [])],
                            list(wv), k=k, limit=5)
        m = evaluate(ho, rf)
        rows[name] = m.as_dict()
        print(f"  {name:<22} k={k:.0f} w={wv} -> R@2={m.recall_at_2:.1f}% R@5={m.recall_at_5:.1f}% MRR={m.mrr:.4f}")

    Path(_HERE / "results" / "calibrate_v4.json").write_text(json.dumps(
        {"dev_grid": grid, "dev_best": {"k": bk, "w": list(bwv)}, "holdout_151": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSalvo em results/calibrate_v4.json")


if __name__ == "__main__":
    main()
