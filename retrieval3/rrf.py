#!/usr/bin/env python3
"""
rrf.py — Etapa 2: fusão Reciprocal Rank Fusion (RRF) multi-query (custo ~0).

Funde os rankings de 2-3 consultas BARATAS (todas já disponíveis, custo desprezível):
  q1 = fala bruta                       (app_fts_query da pergunta crua)
  q2 = fala + boost do classificador    (top-2 NR, boost suave 5x — híbrido atual)
  q3 = keywords TF-IDF top do próprio classificador (já calculadas no forward)

RRF: score(d) = Σ_q  weight_q / (k + rank_q(d)).  k calibrado no dev-set sintético
(nunca no holdout). Cada consulta contribui com um ranking; documentos que aparecem
bem posicionados em várias consultas sobem.

Procedência: task poc-retrieval-v3. PT-BR nos comentários, inglês no código.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))

from bm25 import Bm25Retriever, NrClassifier, apply_soft_boost  # noqa: E402
from eval_common import APP_STOPWORDS, app_fts_query  # noqa: E402
from nr_taxonomy import NONE_LABEL  # noqa: E402


def rrf_fuse(rankings: List[List[Dict[str, Any]]], weights: List[float],
             k: float = 60.0, limit: int = 5) -> List[Dict[str, Any]]:
    """Funde múltiplos rankings por RRF. Cada ranking é uma lista ordenada de chunks."""
    scores: Dict[int, float] = {}
    ref: Dict[int, Dict[str, Any]] = {}
    for rank_list, w in zip(rankings, weights):
        for pos, r in enumerate(rank_list):
            cid = r["id"]
            scores[cid] = scores.get(cid, 0.0) + w / (k + pos + 1)
            if cid not in ref:
                ref[cid] = r
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for cid, sc in ordered[:limit]:
        r = dict(ref[cid])
        r["rrf_score"] = sc
        out.append(r)
    return out


class TfidfKeywordExtractor:
    """Extrai as top palavras-chave da consulta segundo o vetorizador WORD do classificador.

    Reusa o TF-IDF já treinado (custo zero adicional além do forward do classificador):
    projeta a fala no espaço de n-gramas de palavra (1-2) e retorna os termos de maior
    peso TF-IDF (só unigramas reais, sem stopwords) — bom material p/ a 3ª consulta.
    """

    def __init__(self, clf: NrClassifier, top_n: int = 6):
        self.word_vec = clf.word
        self.top_n = top_n
        self.feature_names = np.array(self.word_vec.get_feature_names_out())

    def extract(self, text: str) -> str:
        X = self.word_vec.transform([text])
        if X.nnz == 0:
            return ""
        coo = X.tocoo()
        pairs = sorted(zip(coo.col, coo.data), key=lambda cd: -cd[1])
        terms: List[str] = []
        for col, _ in pairs:
            feat = self.feature_names[col]
            for tok in feat.split():
                if tok in APP_STOPWORDS or len(tok) <= 2:
                    continue
                if tok not in terms:
                    terms.append(tok)
            if len(terms) >= self.top_n:
                break
        return " ".join(terms[:self.top_n])


class RrfMultiQuery:
    """Pipeline Etapa 2: funde por RRF a saída GATED forte + consultas complementares.

    Insight do gate (holdout): o híbrido gated (filtro-duro confiante + boost suave) é
    MUITO mais forte que soft-fusion pura. Então usamos a própria ranking gated como
    consulta-âncora do RRF e só ADICIONAMOS consultas complementares (fala bruta e
    keywords TF-IDF), que resgatam chunks quando o classificador erra a NR.
    """

    def __init__(self, db_path: str | Path, model_path: str | Path,
                 factor: float = 5.0, pool_limit: int = 60, rrf_k: float = 60.0,
                 conf_thr: float = 0.5,
                 weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 use_q3: bool = True, bm_weights: Optional[Tuple[float, ...]] = None):
        self.bm = Bm25Retriever(db_path, weights=bm_weights)
        self.clf = NrClassifier(model_path)
        self.kw = TfidfKeywordExtractor(self.clf)
        self.factor = factor
        self.pool_limit = pool_limit
        self.rrf_k = rrf_k
        self.conf_thr = conf_thr
        self.weights = weights
        self.use_q3 = use_q3

    def _gated_ranking(self, question: str, preds: List[str], p1: float) -> List[Dict[str, Any]]:
        """Réplica do HybridBm25 (gated): filtro-duro se confiante, senão boost suave."""
        q = app_fts_query(question)
        if p1 >= self.conf_thr and preds:
            res = self.bm.search(q, self.pool_limit, doc_filter=preds[:1])
            if len(res) < self.pool_limit:
                extra = self.bm.search(q, self.pool_limit)
                seen = {r["id"] for r in res}
                res += [r for r in extra if r["id"] not in seen]
            return res
        pool = self.bm.search(q, self.pool_limit)
        return apply_soft_boost(pool, preds, self.factor)

    def rank(self, question: str, limit: int = 5) -> List[Dict[str, Any]]:
        topk, scores = self.clf.predict([question], k=2)
        preds = [p for p in topk[0] if p != NONE_LABEL]
        p1 = float(scores[0].max())

        # q1 (âncora): ranking gated forte (filtro-duro confiante + boost suave)
        r1 = self._gated_ranking(question, preds, p1)
        # q2 (complementar): fala bruta sem boost — resgata quando o classificador erra a NR
        r2 = self.bm.search(app_fts_query(question), self.pool_limit)

        rankings = [r1, r2]
        weights = [self.weights[0], self.weights[1]]

        # q3 (complementar): keywords TF-IDF top + boost suave nas top-2 NRs
        if self.use_q3:
            kw = self.kw.extract(question)
            if kw:
                pool3 = self.bm.search(app_fts_query(kw), self.pool_limit)
                r3 = apply_soft_boost(pool3, preds, self.factor)
                rankings.append(r3)
                weights.append(self.weights[2])

        return rrf_fuse(rankings, weights, k=self.rrf_k, limit=limit)
