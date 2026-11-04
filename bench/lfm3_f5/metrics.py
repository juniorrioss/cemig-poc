#!/usr/bin/env python3
"""
metrics.py — as duas réguas de retrieval da F5.

BENCH (benchmark_mobile, definição do mobile_eval — benchmark_mobile_test.py:174-214,
332-384, validada no smoke reproduzindo BM25/Naive publicados):
  chunk relevante  <=> chunk.file == row.source_file E span do chunk sobrepõe a evidência
  gain(chunk)      = chars de evidência cobertos / chars de evidência (união das citações)
  hit@k            = algum dos k primeiros é relevante
  evrec@k          = chars cobertos pela união dos spans do top-k (arquivo-ouro) / total
  nDCG@5           = DCG(gains top5) / DCG(5 melhores gains de TODOS os chunks do arquivo-ouro)
  MRR@5            = 1/rank do primeiro relevante
  content_hit@k    = (não oficial) texto recuperado contém >=70% dos tokens de alguma evidência

POC 151: check_hit oficial (corpus/eval_retrieval.py: doc+seção ou chunk_id) -> R@k, MRR.

Conjuntos não-ordenados (frente 2, vizinhos): hit_set / evrec_set sobre o conjunto.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import common_f5 as C  # noqa: E402
from corpus.eval_retrieval import check_hit  # noqa: E402


# ----------------------------------------------------------------------------- spans
class SpanStore:
    """file + spans de char de cada chunk (normas via align.json; manuais via manuals.py)."""

    def __init__(self, extra: Optional[Dict[int, Dict[str, Any]]] = None):
        a = json.loads((C.CACHE / "align.json").read_text(encoding="utf-8"))
        self.d: Dict[int, Dict[str, Any]] = {int(k): v for k, v in a.items()}
        if extra:
            self.d.update(extra)
        self.by_file: Dict[str, List[int]] = {}
        for cid, v in self.d.items():
            self.by_file.setdefault(v["file"], []).append(cid)

    def file(self, cid: int) -> Optional[str]:
        v = self.d.get(cid)
        return v["file"] if v else None

    def spans(self, cid: int) -> List[List[int]]:
        v = self.d.get(cid)
        return v["spans"] if v else []


def covered(spans: Iterable[Sequence[int]], ev: Sequence[Sequence[int]]) -> int:
    cov = set()
    for es, ee in ev:
        for s, e in spans:
            lo, hi = max(s, es), min(e, ee)
            if lo < hi:
                cov.update(range(lo, hi))
    return len(cov)


def ev_chars(ev) -> int:
    return covered([(0, max(e for _, e in ev))], ev)


def dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


class BenchScorer:
    def __init__(self, store: SpanStore):
        self.st = store
        self._ideal: Dict[str, float] = {}

    def gain(self, cid: int, row) -> float:
        if self.st.file(cid) != row["source_file"]:
            return 0.0
        return covered(self.st.spans(cid), row["ev"]) / max(1, ev_chars(row["ev"]))

    def ideal(self, row) -> float:
        rid = row["id"]
        if rid not in self._ideal:
            gs = sorted((self.gain(c, row) for c in self.st.by_file.get(row["source_file"], [])),
                        reverse=True)[:5]
            self._ideal[rid] = dcg(gs)
        return self._ideal[rid]

    def evrec(self, ids: Iterable[int], row) -> float:
        sp = [s for c in ids if self.st.file(c) == row["source_file"] for s in self.st.spans(c)]
        return covered(sp, row["ev"]) / max(1, ev_chars(row["ev"]))

    def score(self, ranked: List[int], row, extra_sets: Optional[Dict[str, List[int]]] = None
              ) -> Dict[str, float]:
        g = [self.gain(c, row) for c in ranked[:5]]
        first = next((i for i, x in enumerate(g) if x > 0), None)
        idl = self.ideal(row)
        out = {
            "hit1": float(first is not None and first < 1),
            "hit2": float(first is not None and first < 2),
            "hit5": float(first is not None),
            "evrec2": self.evrec(ranked[:2], row),
            "evrec5": self.evrec(ranked[:5], row),
            "ndcg5": dcg(g) / idl if idl > 0 else 0.0,
            "mrr5": 1.0 / (first + 1) if first is not None else 0.0,
        }
        for name, ids in (extra_sets or {}).items():
            out[f"hit_{name}"] = float(any(self.gain(c, row) > 0 for c in ids))
            out[f"evrec_{name}"] = self.evrec(ids, row)
        return out


# ----------------------------------------------------------------------------- content_hit
def _toks(s: str) -> List[str]:
    return [t[0] for t in C.tokens_with_offsets(s)]


def content_hit(texts: List[str], row, thr: float = 0.70) -> float:
    """1 se ALGUM trecho (individualmente) contém >=thr dos tokens de alguma evidência."""
    for t in texts:
        have = set(_toks(t))
        for e in row["evidence"]:
            et = _toks(e["text"])
            if et and sum(1 for x in et if x in have) / len(et) >= thr:
                return 1.0
    return 0.0


# ----------------------------------------------------------------------------- POC 151
def score_151(ranked_chunks: List[Dict[str, Any]], item: Dict[str, Any],
              extra_sets: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, float]:
    rank = next((i + 1 for i, c in enumerate(ranked_chunks[:5]) if check_hit(c, item)), None)
    out = {
        "r1": float(rank is not None and rank <= 1),
        "r2": float(rank is not None and rank <= 2),
        "r5": float(rank is not None),
        "mrr": 1.0 / rank if rank else 0.0,
    }
    for name, chs in (extra_sets or {}).items():
        out[f"hit_{name}"] = float(any(check_hit(c, item) for c in chs))
    return out


def mean_dicts(ds: List[Dict[str, float]]) -> Dict[str, float]:
    if not ds:
        return {}
    keys = ds[0].keys()
    return {k: round(sum(d[k] for d in ds) / len(ds), 4) for k in keys}
