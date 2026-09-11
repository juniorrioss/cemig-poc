#!/usr/bin/env python3
"""
eval_common.py — Harness de avaliação compartilhado do Retrieval v3.

Fonte única de verdade para:
  - carga do holdout INTOCÁVEL (151 reais de qa_pairs_v2 + 20 smoke);
  - `check_hit` idêntico ao do app (reusa corpus.eval_retrieval.check_hit);
  - `app_fts_query` idêntico ao Fts5Retriever.kt (reusa corpus.eval_fino);
  - cálculo de R@1/2/5 + MRR sobre uma função de ranking arbitrária.

REGRA DE OURO (ordem do capitão): o holdout NUNCA é usado para ajustar nada.
Qualquer calibração de peso/k usa o dev-set sintético próprio (retrieval3/data/dev_set.jsonl).

Procedência: gerado na task poc-retrieval-v3 (branch fm/poc-retrieval-v3).
Comentários em PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))          # para importar corpus.*
sys.path.insert(0, str(_ROOT / "classifier"))  # para importar nr_taxonomy/data_utils

from corpus.eval_retrieval import check_hit  # noqa: E402
from corpus.eval_fino import app_fts_query, APP_STOPWORDS  # noqa: E402

QA_V2_PATH = _ROOT / "corpus" / "qa_pairs_v2.jsonl"
SMOKE_PATH = _ROOT / "bench" / "data" / "smoke_qa_20.jsonl"

# Pesos BM25 de produção (Fts5Retriever.kt): doc, section, title, text
APP_W = (1.5, 3.0, 2.0, 1.0)


@dataclass
class QAItem:
    """Um item de holdout com o par-ouro completo p/ check_hit."""

    id: str
    question: str
    doc: str
    section: str
    chunk_id: int
    relevant_chunk_ids: List[int]
    query_terms: str
    source: str  # "qa_v2" | "smoke"

    def gold_pair(self) -> Dict[str, Any]:
        """Estrutura esperada por check_hit."""
        return {
            "doc": self.doc,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "relevant_chunk_ids": self.relevant_chunk_ids,
        }


def load_holdout(subset: str = "qa_v2") -> List[QAItem]:
    """Carrega o holdout fixo. subset: 'qa_v2' (151, métrica oficial) | 'all' (171)."""
    items: List[QAItem] = []
    for line in QA_V2_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append(QAItem(
            id=r["id"],
            question=r["question"],
            doc=str(r.get("doc", "") or "").lower(),
            section=str(r.get("section", "") or ""),
            chunk_id=r.get("chunk_id", -1),
            relevant_chunk_ids=r.get("relevant_chunk_ids", []),
            query_terms=r.get("query_terms", ""),
            source="qa_v2",
        ))
    if subset == "all":
        for line in SMOKE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(QAItem(
                id=r["id"],
                question=r["question"],
                doc=str(r.get("doc", "") or "").lower(),
                section=str(r.get("section", "") or ""),
                chunk_id=r.get("chunk_id", -1),
                relevant_chunk_ids=r.get("relevant_chunk_ids", []),
                query_terms=r.get("query_terms", ""),
                source="smoke",
            ))
    return items


@dataclass
class Metrics:
    """Recall@1/2/5 + MRR (%). n = perguntas avaliadas."""

    n: int
    recall_at_1: float
    recall_at_2: float
    recall_at_5: float
    mrr: float
    per_doc: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "recall_at_1": round(self.recall_at_1, 2),
            "recall_at_2": round(self.recall_at_2, 2),
            "recall_at_5": round(self.recall_at_5, 2),
            "mrr": round(self.mrr, 4),
            "per_doc": self.per_doc,
        }

    def line(self, label: str) -> str:
        return (f"| {label:<42} | {self.recall_at_1:>5.1f}% | {self.recall_at_2:>5.1f}% | "
                f"{self.recall_at_5:>5.1f}% | {self.mrr:>7.4f} |")


# Função de ranking: recebe QAItem, devolve lista de chunks (dicts com id/doc/section/text) em ordem.
RankFn = Callable[[QAItem], List[Dict[str, Any]]]


def evaluate(items: List[QAItem], rank_fn: RankFn, per_doc: bool = False) -> Metrics:
    """Avalia uma função de ranking sobre o holdout. Denominador = todas as perguntas."""
    n = len(items)
    h1 = h2 = h5 = 0
    mrr = 0.0
    dd: Dict[str, Dict[str, float]] = {}
    for it in items:
        res = rank_fn(it)
        gold = it.gold_pair()
        rank: Optional[int] = None
        for idx, r in enumerate(res[:5]):
            if check_hit(r, gold):
                rank = idx + 1
                break
        if per_doc:
            d = dd.setdefault(it.doc, {"n": 0, "h2": 0, "h5": 0})
            d["n"] += 1
        if rank is not None:
            if rank <= 1:
                h1 += 1
            if rank <= 2:
                h2 += 1
                if per_doc:
                    dd[it.doc]["h2"] += 1
            if rank <= 5:
                h5 += 1
                if per_doc:
                    dd[it.doc]["h5"] += 1
            mrr += 1.0 / rank
    per = {}
    if per_doc:
        for d, v in sorted(dd.items()):
            per[d] = {
                "n": int(v["n"]),
                "recall_at_2": round(100 * v["h2"] / v["n"], 1),
                "recall_at_5": round(100 * v["h5"] / v["n"], 1),
            }
    return Metrics(
        n=n,
        recall_at_1=100 * h1 / n,
        recall_at_2=100 * h2 / n,
        recall_at_5=100 * h5 / n,
        mrr=mrr / n,
        per_doc=per,
    )


def header(title: str) -> None:
    print("\n" + "=" * 84)
    print(f"  {title}")
    print("=" * 84)
    print(f"| {'Configuração':<42} | {'R@1':>6} | {'R@2':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 44 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")


def footer() -> None:
    print("=" * 84)


if __name__ == "__main__":
    ho = load_holdout("all")
    from collections import Counter
    print(f"holdout all: {len(ho)}  por fonte: {dict(Counter(h.source for h in ho))}")
    print(f"holdout qa_v2: {len(load_holdout('qa_v2'))}")
