#!/usr/bin/env python3
"""
retriever_lazy.py — retriever REAL v4 da POC com encode de query SOB DEMANDA.

O RetrieverV4 do ctx_topk encoda todas as queries no __init__ (cache por id).
No F4 as consultas nascem durante a geração (2ª chamada do multihop, consultas
do modelo na avaliação), então este wrapper encoda na hora (llama-server GGUF
:8399, mesmo encoder on-device) e delega o resto (BM25 gated-exp + 2 densos +
RRF k=10 w=2,1,2) ao código provado. Zero lógica nova de fusão.

Uso: classifier/.venv/bin/python (precisa de sklearn p/ o classic_winner.pkl).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "ctx_topk"))

import retrieval_v4 as R4  # noqa: E402


class LazyRetrieverV4(R4.RetrieverV4):
    """RetrieverV4 com embed on-demand (thread-safe) e cache por texto."""

    def __init__(self, emb_url: str = R4._EMB_URL):
        # dict não-vazio: o embed_queries do ctx_topk não aceita lista vazia
        super().__init__({"__warmup__": "aquecimento do encoder"}, emb_url=emb_url)
        # a conexão SQLite do HybridBm25 nasce na main thread; os workers
        # (ThreadPool do parallel_map) precisam usá-la -> reabre com
        # check_same_thread=False (seguro: toda busca é serializada no lock)
        import sqlite3
        db_path = str(R4._V4_DB)
        con = sqlite3.connect(db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        self.hy.bm.con = con
        self._emb_url = emb_url
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        # o SQLite do HybridBm25 não é thread-safe -> serializa a BUSCA toda
        # (ela custa ~ms; o paralelismo útil está nas chamadas ao 27B)
        self._search_lock = threading.Lock()

    def _embed(self, text: str):
        with self._lock:
            if text in self._cache:
                return self._cache[text]
        vec = R4.embed_queries(self._emb_url, [text])[0]
        with self._lock:
            self._cache[text] = vec
        return vec

    def search_text(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Busca v4 completa para uma consulta arbitrária; retorna chunks hidratados."""
        vec = self._embed(query)
        qid = f"lazy::{hash(query)}"
        with self._search_lock:
            self.qemb[qid] = vec
            res = self.search(qid, query, top_k=top_k)
            del self.qemb[qid]
        return res["chunks"]
