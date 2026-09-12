#!/usr/bin/env python3
"""
verify_bins.py — Paridade: recall reproduzido a partir dos .bin DVEC1 (como o app fará).

Lê os binários DVEC1 exportados, reencoda as QUERIES do holdout com o encoder GGUF (:8399),
faz o KNN brute-force exato (produto interno, igual ao DenseRetriever.kt) e funde com o
BM25 gated (índice v4) via RRF k=10 pesos (2,1,2) — a config embarcada. Confirma que o
recall bate o medido no pipeline (calibrate_v4 / eval_recall_v4).

Uso (classifier/.venv, servidor GGUF em :8399):
  ../classifier/.venv/bin/python verify_bins.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

import sqlite3  # noqa: E402

from bm25 import HybridBm25  # noqa: E402
from eval_common import evaluate, load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from dense_v4 import QUERY_PROMPT, embed_batch  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
V4_DB = str(_HERE / "indices" / "index_hf_36nr_expv4.db")
URL = "http://127.0.0.1:8399"


def load_dvec(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    assert data[:5] == b"DVEC1", "magic inválido"
    dim, count = struct.unpack("<ii", data[5:13])
    ids = np.empty(count, dtype=np.int32)
    mat = np.empty((count, dim), dtype=np.float32)
    off = 13
    rec = 4 + dim * 4
    for i in range(count):
        ids[i] = struct.unpack("<i", data[off:off + 4])[0]
        mat[i] = np.frombuffer(data[off + 4:off + rec], dtype=np.float32)
        off += rec
    return ids, mat


def knn_rank(qv: np.ndarray, ids: np.ndarray, mat: np.ndarray, pool: int = 60) -> List[Dict]:
    sims = mat @ qv
    top = np.argsort(-sims)[:pool]
    return [{"id": int(ids[j]), "score": float(sims[j])} for j in top]


def main() -> None:
    ids_t, mat_t = load_dvec(_HERE / "export" / "dense_text.bin")
    ids_e, mat_e = load_dvec(_HERE / "export" / "dense_exponly.bin")
    ho = load_holdout("qa_v2")
    qemb = embed_batch(URL, [QUERY_PROMPT + it.question for it in ho])

    hy = HybridBm25(V4_DB, CLF, weights=EXP_W)
    # metadata p/ hidratar ids densos (o app usa Fts5Retriever.fetchByIds)
    con = sqlite3.connect(V4_DB)
    con.row_factory = sqlite3.Row
    meta = {r["id"]: dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks").fetchall()}
    con.close()

    def hydrate(ranks):
        out = []
        for r in ranks:
            m = meta.get(r["id"])
            if m:
                out.append({**m, "score": r["score"]})
        return out

    def fusion(it, qi):
        s1 = hy.rank(it.question, 60)
        sT = hydrate(knn_rank(qemb[qi], ids_t, mat_t))
        sE = hydrate(knn_rank(qemb[qi], ids_e, mat_e))
        return rrf_fuse([s1, sT, sE], [2.0, 1.0, 2.0], k=10.0, limit=5)

    idx = {it.id: i for i, it in enumerate(ho)}
    m = evaluate(ho, lambda it: fusion(it, idx[it.id]))
    print("=" * 70)
    print("  PARIDADE dos .bin DVEC1 (config embarcada: RRF k=10 w=2,1,2)")
    print("=" * 70)
    print(m.line("Fusão v4 (bins DVEC1)"))
    print("=" * 70)
    print("Esperado ~ R@2 51.0% / R@5 65.6% (calibrate_v4). Diferença deve ser ~0.")


if __name__ == "__main__":
    main()
