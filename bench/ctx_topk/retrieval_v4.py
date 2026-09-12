#!/usr/bin/env python3
"""
retrieval_v4.py — Espelho do pipeline de retrieval EMBARCADO (v4) para a bancada
JANELA×TRECHOS. Reproduz fielmente o `HybridRetriever.searchV3` do app com a config v4:

  s1 = BM25 gated-EXPANDIDO   (índice FTS5 v4 = android/app/src/main/assets/index.db,
                               5 campos, pesos 1.5/3/2/1/1, classificador NR gated)
  sT = denso EmbeddingGemma-300M sobre TEXTO+expansão (query↔passagem)   [DVEC1 v4]
  sE = denso EmbeddingGemma-300M sobre EXPANSÃO-v4-only (query↔query)     [DVEC1 v4]
       → RRF(k=10, pesos BM25=2, texto=1, exp=2) → top-K final

PARIDADE (provada): usa os MESMOS `.bin` DVEC1 puxados do S24+ (data/dense_*.bin) e o
MESMO encoder GGUF on-device (llama-server :8399). Reproduz exatamente a régua embarcada
R@2 51.7% / R@5 65.6% / MRR 0.4944 nas 151 (verify_bins.py da retrieval4).

Diferença desta bancada vs. o app: aqui variamos top_k (2..5) e opcionalmente aplicamos
TRIM determinístico (trim.py) ao texto de cada chunk antes da síntese — para medir o
trade-off janela×trechos no juiz e no aparelho.

Ambiente: classifier/.venv (sklearn 1.9.1). Encoder denso via llama-server GGUF :8399.
Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

import sqlite3  # noqa: E402

from bm25 import HybridBm25  # noqa: E402
from rrf import rrf_fuse  # noqa: E402

from corpus.eval_retrieval import check_hit  # noqa: E402

# Config v4 embarcada (retrieval4/README.md + verify_bins.py):
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)   # pesos dos 5 campos FTS5 (com expansion)
RRF_K = 10.0                        # k do RRF v4 (calibrado no dev)
RRF_W = (2.0, 1.0, 2.0)             # pesos por sinal: BM25, texto, exp
POOL = 60                           # profundidade de cada sinal antes da fusão
QUERY_PROMPT = "task: search result | query: "

_V4_DB = _ROOT / "android" / "app" / "src" / "main" / "assets" / "index.db"   # índice FTS5 v4
_CLF = _ROOT / "classifier" / "models" / "classic_winner.pkl"
_BIN_TEXT = _HERE / "data" / "dense_text.bin"      # DVEC1 v4 (texto+exp) — puxado do S24+
_BIN_EXP = _HERE / "data" / "dense_exponly.bin"    # DVEC1 v4 (exp-only)  — puxado do S24+
_EMB_URL = "http://127.0.0.1:8399"


def embed_queries(url: str, texts: List[str], timeout: int = 300) -> np.ndarray:
    """Encoda queries com o encoder GGUF on-device (mean-pool + L2-norm nativo)."""
    out: List[np.ndarray] = []
    for i in range(0, len(texts), 64):
        batch = [QUERY_PROMPT + t for t in texts[i:i + 64]]
        r = requests.post(f"{url}/v1/embeddings", json={"input": batch, "model": "emb"}, timeout=timeout)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda x: x["index"])
        embs = np.array([row["embedding"] for row in rows], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out.append(embs / norms)
    return np.vstack(out)


def load_dvec(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê o formato DVEC1 (magic + dim + count + [id, vec]*)."""
    data = path.read_bytes()
    assert data[:5] == b"DVEC1", f"magic inválido em {path}"
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


class RetrieverV4:
    """Fusão RRF 3-sinais v4 (embarcada). Encoda todas as queries de uma vez (cache)."""

    def __init__(self, questions_by_id: Dict[str, str], emb_url: str = _EMB_URL):
        self.hy = HybridBm25(str(_V4_DB), str(_CLF), weights=EXP_W)
        self.clf_name = self.hy.clf.name
        self.ids_t, self.mat_t = load_dvec(_BIN_TEXT)
        self.ids_e, self.mat_e = load_dvec(_BIN_EXP)
        # Metadata p/ hidratar ids densos (o app usa Fts5Retriever.fetchByIds).
        con = sqlite3.connect(str(_V4_DB))
        con.row_factory = sqlite3.Row
        self.meta = {r["id"]: dict(r) for r in con.execute(
            "SELECT id, doc, section, title, text, page FROM chunks").fetchall()}
        con.close()
        # Encoda todas as queries de uma vez (1 encode por query serve os 2 índices densos).
        self._order = list(questions_by_id.keys())
        qemb = embed_queries(emb_url, [questions_by_id[q] for q in self._order])
        self.qemb = {qid: qemb[i] for i, qid in enumerate(self._order)}

    def _knn(self, qv: np.ndarray, ids: np.ndarray, mat: np.ndarray, pool: int = POOL) -> List[Dict[str, Any]]:
        sims = mat @ qv
        top = np.argsort(-sims)[:pool]
        return [{"id": int(ids[j]), "score": float(sims[j])} for j in top]

    def _hydrate(self, ranks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in ranks:
            m = self.meta.get(r["id"])
            if m:
                out.append({**m, "score": r["score"]})
        return out

    def search(self, item_id: str, raw_question: str, top_k: int = 2) -> Dict[str, Any]:
        """Estágio 1+2 v4 -> RRF k=10 w=(2,1,2) -> top-K. Igual ao searchV3 embarcado."""
        s1 = self.hy.rank(raw_question, POOL)
        qv = self.qemb[item_id]
        sT = self._hydrate(self._knn(qv, self.ids_t, self.mat_t))
        sE = self._hydrate(self._knn(qv, self.ids_e, self.mat_e))
        fused = rrf_fuse([s1, sT, sE], list(RRF_W), k=RRF_K, limit=top_k)

        # Diagnóstico do gate (telemetria/debug).
        topk, scores = self.hy.clf.predict([raw_question], k=2)
        from nr_taxonomy import NONE_LABEL
        preds = list(topk[0])
        p1 = float(scores[0].max())
        if p1 >= self.hy.conf_thr and preds and preds[0] != NONE_LABEL:
            mode = "hard"
        elif preds and preds[0] == NONE_LABEL:
            mode = "none"
        else:
            mode = "soft"
        decision = {"top1": preds[0] if preds else NONE_LABEL, "top1_prob": round(p1, 4),
                    "top2": preds[1] if len(preds) > 1 else NONE_LABEL, "mode": mode,
                    "fusion": "rrf3_v4", "rrf_k": RRF_K, "rrf_w": list(RRF_W)}
        return {"chunks": fused, "decision": decision}


def retrieval_hit(chunks: List[Dict[str, Any]], gold: Dict[str, Any]) -> bool:
    return any(check_hit(c, gold) for c in chunks)
