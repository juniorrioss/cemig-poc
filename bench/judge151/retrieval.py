#!/usr/bin/env python3
"""
retrieval.py — Estágio 1+2 híbrido idêntico ao app (HybridRetriever.kt), para o benchmark
de qualidade de RESPOSTA das 151 perguntas reais.

Reusa a política CONFIANÇA-GATED provada em classifier/hybrid.py e no app:
  - override: NR citada explicitamente na fala -> filtro DURO na NR;
  - top-1 == 'nenhuma'      -> BM25 amplo sem boost;
  - prob(top-1) >= 0.5      -> filtro DURO na top-1;
  - senão                   -> boost SUAVE 5x nas top-2.

A busca BM25 usa a FALA BRUTA (app_fts_query: stem-6 + prefixo* + OR, pesos 1.5/3/2/1),
coerente com os achados M3 (reescrita PIORA o retrieval intra-norma). Retorna top-2 chunks
(topk2 lean ~820 tok) idêntico ao AskPipeline de produção.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "classifier"))

from corpus.eval_fino import app_fts_query  # noqa: E402
from corpus.eval_retrieval import check_hit  # noqa: E402

# Reusa as funções de busca/boost já validadas do híbrido do classifier.
from classifier.hybrid import (  # noqa: E402
    APP_W,
    apply_soft_boost,
    hard_filter,
    load_classifier,
    predict_topk,
    search_pool,
)
from nr_taxonomy import NONE_LABEL  # noqa: E402

# Detecta menção explícita de norma na fala (idêntico a HybridRetriever.detectExplicitNr).
_EXPLICIT_NR = re.compile(r"\bnr[\s._-]*(\d{1,2})\b", re.IGNORECASE)

CONF_THRESHOLD = 0.5
BOOST_FACTOR = 5.0
TOP_NR = 2
TOP_K = 2


def detect_explicit_nr(text: str) -> Optional[str]:
    """Extrai a norma explícita citada na fala, normalizada em nr-XX; None se ausente."""
    m = _EXPLICIT_NR.search(text)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    return "nr-%02d" % n


class HybridRetriever:
    """Espelho Python do HybridRetriever.kt (paridade de classificador já provada)."""

    def __init__(self, db_path: str, model_path: str):
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self.char_vec, self.word_vec, self.clf, self.clf_name = load_classifier(Path(model_path))

    def classify(self, raw_question: str):
        """Retorna (top1, prob1, top2) da fala bruta."""
        topk, scores = predict_topk(self.char_vec, self.word_vec, self.clf, [raw_question], k=2)
        preds = topk[0]
        prob1 = float(scores[0].max())
        top1 = preds[0] if preds else NONE_LABEL
        top2 = preds[1] if len(preds) > 1 else NONE_LABEL
        return top1, prob1, top2

    def search(self, raw_question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Estágio 1 (gated) + estágio 2 (BM25 top-k). Retorna chunks + decisão de telemetria."""
        q = app_fts_query(raw_question)

        # Override: NR citada explicitamente -> filtro duro.
        explicit = detect_explicit_nr(raw_question)
        if explicit is not None:
            res = hard_filter(self.con, q, [explicit], top_k)
            decision = {"top1": explicit, "top1_prob": 1.0, "top2": explicit,
                        "boost_nrs": [explicit], "mode": "explicit"}
            return {"chunks": res[:top_k], "decision": decision, "fts_query": q}

        top1, prob1, top2 = self.classify(raw_question)

        if top1 == NONE_LABEL:
            res = search_pool(self.con, q, top_k)
            mode, boost = "none", []
        elif prob1 >= CONF_THRESHOLD:
            boost = [top1]
            res = hard_filter(self.con, q, boost, top_k)
            mode = "hard"
        else:
            boost = [t for t in (top1, top2) if t != NONE_LABEL]
            pool = search_pool(self.con, q, 60)
            res = apply_soft_boost(pool, boost, BOOST_FACTOR)[:top_k]
            mode = "soft"

        decision = {"top1": top1, "top1_prob": round(prob1, 4), "top2": top2,
                    "boost_nrs": boost, "mode": mode}
        return {"chunks": res[:top_k], "decision": decision, "fts_query": q}


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formata o contexto normativo idêntico ao AskPipeline.kt (Turno 2)."""
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def retrieval_hit(chunks: List[Dict[str, Any]], gold: Dict[str, Any]) -> bool:
    """True se algum chunk recuperado casa o gabarito ouro (check_hit do app)."""
    return any(check_hit(c, gold) for c in chunks)
