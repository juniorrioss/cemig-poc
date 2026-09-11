#!/usr/bin/env python3
"""
bm25.py — Camada BM25 (FTS5) do Retrieval v3, reusável como biblioteca.

Espelha fielmente o caminho de busca do app (Fts5Retriever.kt / HybridRetriever.kt):
  - `app_fts_query` (stem-6 + prefixo* + OR) importado do harness comum;
  - pesos de campo de produção APP_W = (1.5, 3.0, 2.0, 1.0);
  - boost suave 5x nas top-2 NRs do classificador (gated é opcional).

Suporta índice com campo extra `expansion` (Etapa 1): pesos de 5 campos quando o
FTS5 tiver a coluna `expansion` (doc, section, title, text, expansion).

Procedência: task poc-retrieval-v3. Comentários PT-BR, código inglês.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))

from eval_common import APP_W, app_fts_query  # noqa: E402
from nr_taxonomy import NONE_LABEL  # noqa: E402

# Pesos com 5 campos (quando o índice tem 'expansion'); o 5º é calibrável.
APP_W5_DEFAULT = (1.5, 3.0, 2.0, 1.0, 1.0)


def open_db(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _fts_has_expansion(con: sqlite3.Connection) -> bool:
    """Detecta se o FTS5 foi criado com a coluna `expansion`."""
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks_fts'"
    ).fetchone()
    return bool(row) and "expansion" in (row[0] or "")


class Bm25Retriever:
    """Busca BM25 no índice FTS5 com pesos de produção (4 ou 5 campos)."""

    def __init__(self, db_path: str | Path, weights: Optional[Tuple[float, ...]] = None):
        self.con = open_db(db_path)
        self.has_exp = _fts_has_expansion(self.con)
        if weights is not None:
            self.w = weights
        else:
            self.w = APP_W5_DEFAULT if self.has_exp else APP_W

    def close(self) -> None:
        self.con.close()

    def _bm25_expr(self) -> str:
        return "bm25(chunks_fts, " + ", ".join(str(x) for x in self.w) + ")"

    def search(self, fts_q: str, limit: int, doc_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Busca ampla; devolve chunks com `score` (bm25 negativo, menor=melhor)."""
        if fts_q == '""':
            return []
        cols = "c.id, c.doc, c.section, c.title, c.page, c.text"
        cur = self.con.cursor()
        if doc_filter:
            ph = ",".join("?" for _ in doc_filter)
            sql = (f"SELECT {cols}, {self._bm25_expr()} AS score "
                   f"FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                   f"WHERE c.doc IN ({ph}) AND chunks_fts MATCH ? "
                   f"ORDER BY score ASC LIMIT ?;")
            params: Tuple = (*doc_filter, fts_q, limit)
        else:
            sql = (f"SELECT {cols}, {self._bm25_expr()} AS score "
                   f"FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                   f"WHERE chunks_fts MATCH ? ORDER BY score ASC LIMIT ?;")
            params = (fts_q, limit)
        try:
            cur.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in cur.fetchall()]

    def search_raw(self, question: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Baseline honesto do app: fala bruta -> app_fts_query -> BM25."""
        return self.search(app_fts_query(question), limit)


def apply_soft_boost(pool: List[Dict[str, Any]], boost_nrs: List[str], factor: float) -> List[Dict[str, Any]]:
    """Boost suave multiplicativo (bm25 negativo: *factor>1 torna melhor)."""
    bset = set(boost_nrs)
    for r in pool:
        r["adj"] = r["score"] * factor if r["doc"] in bset else r["score"]
    pool.sort(key=lambda x: x["adj"])
    return pool


class NrClassifier:
    """Wrapper do classificador vencedor (LogReg TF-IDF) p/ boost gated."""

    def __init__(self, model_path: str | Path):
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        self.char = blob["char"]
        self.word = blob["word"]
        self.clf = blob["clf"]
        self.name = blob.get("name", "?")
        self.classes = self.clf.classes_

    def predict(self, texts: List[str], k: int = 2) -> Tuple[List[List[str]], np.ndarray]:
        Xc = self.char.transform(texts)
        Xw = self.word.transform(texts)
        X = hstack([Xc, Xw]).tocsr()
        if hasattr(self.clf, "predict_proba"):
            scores = self.clf.predict_proba(X)
        else:
            scores = self.clf.decision_function(X)
        order = np.argsort(-scores, axis=1)[:, :k]
        topk = [[self.classes[j] for j in row] for row in order]
        return topk, scores


class HybridBm25:
    """Réplica do HybridRetriever.kt: classificador-NR + BM25 boost gated (main atual)."""

    def __init__(self, db_path: str | Path, model_path: str | Path,
                 factor: float = 5.0, conf_thr: float = 0.5, pool_limit: int = 60,
                 weights: Optional[Tuple[float, ...]] = None):
        self.bm = Bm25Retriever(db_path, weights=weights)
        self.clf = NrClassifier(model_path)
        self.factor = factor
        self.conf_thr = conf_thr
        self.pool_limit = pool_limit

    def rank(self, question: str, limit: int = 5) -> List[Dict[str, Any]]:
        topk, scores = self.clf.predict([question], k=2)
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
        pool = self.bm.search(q, self.pool_limit)
        return apply_soft_boost(pool, preds, self.factor)[:limit]
