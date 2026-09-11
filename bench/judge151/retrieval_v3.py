#!/usr/bin/env python3
"""
retrieval_v3.py — Estágio 1+2 do RETRIEVAL v3 (fusão RRF 3-sinais) para o benchmark de
qualidade de RESPOSTA das 151 reais. Substitui o `HybridRetriever` do main (que usa só
BM25-gated no índice antigo) pelo pipeline vencedor da entrega v3 (retrieval3/README.md):

  s1 = BM25 gated-EXPANDIDO   (index_hf_36nr_exp.db, pesos 1.5/3/2/1/1, classificador NR)
  sT = denso EmbeddingGemma-300M sobre TEXTO+expansão   (query↔passagem)
  sE = denso EmbeddingGemma-300M sobre EXPANSÃO-ONLY     (query↔query coloquial)
       → RRF(k=30, pesos iguais) → top-2 final

PARIDADE HONESTA: os sinais densos (sT, sE) vêm do CACHE de rankings já produzido pelo
encode na 5070 (`retrieval3/results/dense_rank_gemma768_exp.json` e `..._exponly.json`),
keyed pelo id da pergunta. Isto é idêntico ao que o app fará on-device (1 encode da query
serve os dois índices densos); aqui só reusamos o encode já feito para não repicar a GPU.
O BM25-gated-expandido (s1) é computado ao vivo, exatamente como no app.

Mesma interface do `retrieval.HybridRetriever` (search/format_context/retrieval_hit) para
o `pipeline.py` trocar de retriever sem outras mudanças.

Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

# Reusa o híbrido gated-expandido e o RRF já validados na lib retrieval3.
from bm25 import HybridBm25  # noqa: E402  (retrieval3/bm25.py)
from rrf import rrf_fuse  # noqa: E402  (retrieval3/rrf.py)

from corpus.eval_retrieval import check_hit  # noqa: E402

# Config vencedora da entrega v3 (fusion_final.json / consolidation.json):
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)   # pesos dos 5 campos FTS5 (com expansion)
RRF_K = 30.0                        # k do RRF (fusão 3-sinais)
POOL = 60                           # profundidade de cada sinal antes da fusão
TOP_K = 2                           # top-2 final (topk2 lean ~820 tok)

_DEFAULT_EXP_DB = _ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db"
_DEFAULT_CLF = _ROOT / "classifier" / "models" / "classic_winner.pkl"
_DEFAULT_DENSE_TEXT = _ROOT / "retrieval3" / "results" / "dense_rank_gemma768_exp.json"
_DEFAULT_DENSE_EXP = _ROOT / "retrieval3" / "results" / "dense_rank_gemma768_exponly.json"


class RetrieverV3:
    """Fusão RRF 3-sinais (entrega v3). Interface compatível com retrieval.HybridRetriever."""

    def __init__(self, db_path: Optional[str] = None, model_path: Optional[str] = None,
                 dense_text_path: Optional[str] = None, dense_exp_path: Optional[str] = None):
        db = db_path or str(_DEFAULT_EXP_DB)
        clf = model_path or str(_DEFAULT_CLF)
        self.hy = HybridBm25(db, clf, weights=EXP_W)
        self.clf_name = self.hy.clf.name
        dt = dense_text_path or str(_DEFAULT_DENSE_TEXT)
        de = dense_exp_path or str(_DEFAULT_DENSE_EXP)
        self.dense_text: Dict[str, List[Dict[str, Any]]] = json.loads(Path(dt).read_text(encoding="utf-8"))
        self.dense_exp: Dict[str, List[Dict[str, Any]]] = json.loads(Path(de).read_text(encoding="utf-8"))

    def search(self, raw_question: str, top_k: int = TOP_K,
               item_id: Optional[str] = None) -> Dict[str, Any]:
        """Estágio 1 (BM25 gated-exp) + sinais densos cacheados -> RRF 3-sinais -> top-k.

        item_id é OBRIGATÓRIO para localizar os rankings densos cacheados (keyed por id).
        """
        s1 = self.hy.rank(raw_question, POOL)
        sT = self.dense_text.get(item_id, []) if item_id else []
        sE = self.dense_exp.get(item_id, []) if item_id else []

        # Diagnóstico do gate do estágio 1 (para telemetria/debug).
        topk, scores = self.hy.clf.predict([raw_question], k=2)
        preds = list(topk[0])
        p1 = float(scores[0].max())
        from nr_taxonomy import NONE_LABEL  # local p/ não poluir o topo
        if p1 >= self.hy.conf_thr and preds and preds[0] != NONE_LABEL:
            mode = "hard"
        elif preds and preds[0] == NONE_LABEL:
            mode = "none"
        else:
            mode = "soft"

        fused = rrf_fuse([s1, sT, sE], [1.0, 1.0, 1.0], k=RRF_K, limit=top_k)

        # Anexa os ranks por sinal a cada chunk fundido (para o modo debug/telemetria).
        def rank_of(rank_list: List[Dict[str, Any]], cid: int) -> int:
            for pos, r in enumerate(rank_list):
                if r["id"] == cid:
                    return pos + 1
            return -1

        chunks: List[Dict[str, Any]] = []
        for r in fused:
            c = dict(r)
            c["rank_bm25"] = rank_of(s1, r["id"])
            c["rank_dense_text"] = rank_of(sT, r["id"])
            c["rank_dense_exp"] = rank_of(sE, r["id"])
            chunks.append(c)

        decision = {
            "top1": preds[0] if preds else NONE_LABEL,
            "top1_prob": round(p1, 4),
            "top2": preds[1] if len(preds) > 1 else NONE_LABEL,
            "mode": mode,
            "fusion": "rrf3_s1sTsE",
            "rrf_k": RRF_K,
        }
        return {"chunks": chunks, "decision": decision, "fts_query": raw_question}


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Idêntico ao AskPipeline.kt (Turno 2). Reusa o do módulo retrieval do main."""
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def retrieval_hit(chunks: List[Dict[str, Any]], gold: Dict[str, Any]) -> bool:
    return any(check_hit(c, gold) for c in chunks)
