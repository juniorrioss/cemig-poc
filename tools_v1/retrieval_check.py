#!/usr/bin/env python3
"""
retrieval_check.py — PASSO 2: valida que a CONSULTA gerada recupera o chunk-ouro no índice v4.

O produto central deste treino é a 'consulta' (fala crua -> termos de busca). Um exemplo só
entra no dataset se a consulta, executada contra o índice v4 (o MESMO que o app embarca),
recupera o chunk-ouro do diálogo no top-K. Assim o modelo só aprende argumentos que
FUNCIONAM na nossa busca (ordem do brief).

Usa BM25 sobre o índice v4 (retrieval4/indices/index_hf_36nr_expv4.db, 5 campos com
`expansion`). BM25 é o sinal que a consulta afeta DIRETAMENTE e é barato o suficiente para
filtrar dezenas de milhares de candidatos em paralelo (a fusão densa exige llama-server e é
reservada para a régua final nas 151, não para o filtro por-exemplo).

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_ROOT / "classifier"))

from bm25 import Bm25Retriever  # noqa: E402

V4_INDEX = _ROOT / "retrieval4" / "indices" / "index_hf_36nr_expv4.db"
APP_W5 = (1.5, 3.0, 2.0, 1.0, 1.0)


@lru_cache(maxsize=2)
def _retriever(db_path: str = str(V4_INDEX)) -> Bm25Retriever:
    return Bm25Retriever(db_path, weights=APP_W5)


def rank_ids(query: str, limit: int = 10, db_path: str = str(V4_INDEX)) -> List[int]:
    """Retorna os ids de chunk recuperados pela consulta (ordem BM25) no índice v4."""
    r = _retriever(db_path)
    res = r.search_raw(query, limit)
    return [row["id"] for row in res]


def recovers_gold(query: str, gold_id: int, topk: int = 5,
                  relevant_ids: Optional[List[int]] = None,
                  db_path: str = str(V4_INDEX)) -> bool:
    """True se a consulta recupera o chunk-ouro (ou um relevante) no top-K do índice v4."""
    ids = rank_ids(query, topk, db_path)
    if gold_id in ids:
        return True
    if relevant_ids:
        rel = set(relevant_ids)
        return any(i in rel for i in ids)
    return False


def gold_rank(query: str, gold_id: int, limit: int = 50,
              relevant_ids: Optional[List[int]] = None,
              db_path: str = str(V4_INDEX)) -> Optional[int]:
    """Posição (1-based) do chunk-ouro no ranking BM25 v4, ou None se fora do limite."""
    ids = rank_ids(query, limit, db_path)
    rel = set(relevant_ids or [])
    for i, cid in enumerate(ids, 1):
        if cid == gold_id or cid in rel:
            return i
    return None


if __name__ == "__main__":
    # smoke: consulta técnica vs fala crua no mesmo chunk-ouro conhecido.
    import sqlite3
    con = sqlite3.connect(str(V4_INDEX))
    row = con.execute("SELECT id, doc, section, text FROM chunks WHERE doc='nr-10' "
                      "AND text LIKE '%13,8%' LIMIT 1").fetchone()
    con.close()
    if row:
        cid = row[0]
        print(f"chunk-ouro id={cid} doc={row[1]} sec={row[2]}")
        for q in ["distancia seguranca zona risco 13,8 kV media tensao",
                  "distância segura pra trabalhar perto de 13800 volts",
                  "posso chegar perto do fio de alta"]:
            print(f"  rank={gold_rank(q, cid)!s:>5}  top5={recovers_gold(q, cid)}  q={q!r}")
    else:
        print("nenhum chunk 13,8 encontrado (índice v4 pode não ter a tabela reparada)")
