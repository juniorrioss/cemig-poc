#!/usr/bin/env python3
"""
retriever_f5.py — retriever v4 PARAMETRIZÁVEL para as 4 frentes da F5.

Base = bench/ctx_topk/retrieval_v4.py (mesmos índices/bins/encoder/RRF k=10 w=2,1,2),
com o gate do estágio 1 reimplementado FIEL ao HybridRetriever.kt::searchV3:

  explicit (NR citada na fala)  -> filtro duro na NR citada (+ preenchimento amplo)
  top1 == 'nenhuma'             -> sem boost
  p(top1) >= conf_thr           -> filtro duro na top1 (+ preenchimento amplo)
  senão                         -> boost suave `factor` nas top-2 (sem 'nenhuma')

DIVERGÊNCIA achada no espelho antigo (HybridBm25.rank, retrieval3/bm25.py): quando
top1=='nenhuma' ele remove 'nenhuma' e aplica filtro duro/boost na TOP-2; o app NÃO
aplica boost nenhum. `gate='mirror'` reproduz o espelho antigo (âncora F3/F4).

Frente 2: `neighbors=1` expande o top-K com os vizinhos id±1 da MESMA norma.
Frente 4: `manual_w>0` adiciona 2 sinais de MANUAL (BM25 em tabela FTS separada +
denso próprio, ver manuals.py) ao RRF com peso próprio — nunca no BM25 das normas.

Query embeddings: cache em disco (cache/qemb.npz) por texto.
Interpretador: classifier/.venv. Encoder: llama-server GGUF :8399.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, REPO / "classifier", REPO / "retrieval3", REPO / "bench" / "ctx_topk", HERE):
    sys.path.insert(0, str(p))

import common_f5 as C  # noqa: E402
import retrieval_v4 as R4  # noqa: E402
from bm25 import Bm25Retriever, NrClassifier  # noqa: E402
from eval_common import app_fts_query  # noqa: E402
from nr_taxonomy import NONE_LABEL  # noqa: E402
from rrf import rrf_fuse  # noqa: E402


@dataclass(frozen=True)
class Cfg:
    """Configuração de uma variante de retrieval."""
    gate: str = "app"             # app | mirror | noclf
    explicit: bool = True         # override de NR explícita (app)
    conf_thr: float = 0.5         # limiar do filtro duro
    factor: float = 5.0           # boost suave nas top-2
    neighbors: int = 0            # Frente 2: vizinhos ±n de cada top-K
    manual_w: float = 0.0         # Frente 4: peso dos sinais de manual (0 = desligado)
    manual_gate: str = "always"   # always | intent (regex) | margin (denso manual >= denso norma + m)
    manual_margin: float = 0.0
    rrf_k: float = R4.RRF_K
    rrf_w: tuple = R4.RRF_W

    def with_(self, **kw) -> "Cfg":
        return replace(self, **kw)


MANUAL_ID_OFFSET = 1_000_000      # ids de chunk de manual = offset + idx (não colidem)


class QueryEmbCache:
    """Cache persistente de embeddings de query (texto -> vetor L2-normalizado)."""

    def __init__(self, path: Path = C.CACHE / "qemb.npz", url: str = C.EMB_URL):
        self.path, self.url = path, url
        self.d: Dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            for k, v in zip(z["keys"], z["vecs"]):
                self.d[str(k)] = v
        self.dirty = False

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def ensure(self, texts: List[str]) -> None:
        miss = list(dict.fromkeys(t for t in texts if self.key(t) not in self.d))
        for i in range(0, len(miss), 32):
            batch = miss[i:i + 32]
            vecs = R4.embed_queries(self.url, batch)
            for t, v in zip(batch, vecs):
                self.d[self.key(t)] = v
            self.dirty = True
        if self.dirty:
            self.save()

    def get(self, text: str) -> np.ndarray:
        k = self.key(text)
        if k not in self.d:
            self.ensure([text])
        return self.d[k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.d.keys()))
        vecs = np.stack(list(self.d.values())) if self.d else np.zeros((0, 768), np.float32)
        np.savez(self.path, keys=keys, vecs=vecs)
        self.dirty = False


class F5Retriever:
    def __init__(self, with_manuals: bool = False):
        self.bm = Bm25Retriever(str(R4._V4_DB), weights=R4.EXP_W)
        self.clf = NrClassifier(str(R4._CLF))
        self.ids_t, self.mat_t = R4.load_dvec(R4._BIN_TEXT)
        self.ids_e, self.mat_e = R4.load_dvec(R4._BIN_EXP)
        self.meta = C.load_chunks()
        self.qc = QueryEmbCache()
        self._clf_cache: Dict[str, tuple] = {}
        self.man = None
        if with_manuals:
            from manuals import ManualIndex
            self.man = ManualIndex.load()

    # ------------------------------------------------------------------ estágio 1
    def classify(self, q: str) -> tuple:
        if q not in self._clf_cache:
            topk, scores = self.clf.predict([q], k=2)
            self._clf_cache[q] = (list(topk[0]), float(scores[0].max()))
        return self._clf_cache[q]

    def _hard(self, fq: str, nr: str, pool: int) -> List[Dict[str, Any]]:
        res = self.bm.search(fq, pool, doc_filter=[nr])
        if len(res) < pool:
            seen = {r["id"] for r in res}
            res += [r for r in self.bm.search(fq, pool) if r["id"] not in seen]
        return res[:pool]

    def _soft(self, fq: str, nrs: List[str], factor: float, pool: int) -> List[Dict[str, Any]]:
        res = self.bm.search(fq, pool)
        bset = set(nrs)
        for r in res:
            r["adj"] = r["score"] * factor if r["doc"] in bset else r["score"]
        return sorted(res, key=lambda r: r["adj"])

    def bm25_signal(self, q: str, cfg: Cfg, pool: int = R4.POOL) -> tuple:
        """Retorna (ranking BM25 do estágio 1+2, modo do gate)."""
        fq = app_fts_query(q)
        if cfg.explicit:
            ex = R4.detect_explicit_nr(q)
            if ex is not None:
                return self._hard(fq, ex, pool), "explicit"
        if cfg.gate == "noclf":
            return self.bm.search(fq, pool), "noclf"
        preds, p1 = self.classify(q)
        if cfg.gate == "mirror":
            # espelho antigo (retrieval3/bm25.py HybridBm25.rank) — remove 'nenhuma'
            nn = [p for p in preds if p != NONE_LABEL]
            if p1 >= cfg.conf_thr and nn:
                return self._hard(fq, nn[0], pool), "hard"
            return self._soft(fq, nn, cfg.factor, pool), "soft"
        top1 = preds[0] if preds else NONE_LABEL
        if top1 == NONE_LABEL:
            return self.bm.search(fq, pool), "none"
        if p1 >= cfg.conf_thr:
            return self._hard(fq, top1, pool), "hard"
        boost = [p for p in preds[:2] if p != NONE_LABEL]
        return self._soft(fq, boost, cfg.factor, pool), "soft"

    # ------------------------------------------------------------------ densos
    def _knn(self, qv, ids, mat, pool=R4.POOL) -> List[Dict[str, Any]]:
        sims = mat @ qv
        top = np.argsort(-sims)[:pool]
        return [{"id": int(ids[j]), "score": float(sims[j])} for j in top]

    def manual_wanted(self, q: str, cfg: Cfg, qv=None, sT=None) -> bool:
        if cfg.manual_w <= 0 or self.man is None:
            return False
        if cfg.manual_gate == "always":
            return True
        if cfg.manual_gate == "margin":
            # gate sem regex: o melhor trecho de MANUAL é mais parecido com a fala que o
            # melhor trecho de NORMA (denso texto) por pelo menos `manual_margin`.
            top_m = float((self.man.mat @ qv).max())
            return top_m >= sT[0]["score"] + cfg.manual_margin
        return bool(C.MANUAL_INTENT_RE.search(q))

    # ------------------------------------------------------------------ busca
    def search(self, q: str, cfg: Cfg, top_k: int = 2, base_k: int = 2) -> Dict[str, Any]:
        """top_k = profundidade do ranking devolvido; base_k = trechos do app (topK=2),
        base da expansão de vizinhos (Frente 2)."""
        s1, mode = self.bm25_signal(q, cfg)
        qv = self.qc.get(q)
        sT = self._knn(qv, self.ids_t, self.mat_t)
        sE = self._knn(qv, self.ids_e, self.mat_e)
        signals = [s1, sT, sE]
        weights = list(cfg.rrf_w)
        if self.manual_wanted(q, cfg, qv, sT):
            mb, md = self.man.signals(q, qv)
            signals += [mb, md]
            weights += [cfg.manual_w, cfg.manual_w]
        fused = rrf_fuse(signals, weights, k=cfg.rrf_k, limit=max(top_k, 5))
        ids = [r["id"] for r in fused]
        top = ids[:top_k]
        out = {"ids": ids, "top": top, "mode": mode}
        if cfg.neighbors:
            out["expanded"] = self.expand_neighbors(ids[:base_k], cfg.neighbors)
        return out

    def expand_neighbors(self, ids: List[int], n: int) -> List[int]:
        """top-K + vizinhos ±n da MESMA norma/manual, em ordem de documento por âncora."""
        out: List[int] = []
        for cid in ids:
            doc = self.doc_of(cid)
            for d in range(-n, n + 1):
                nb = cid + d
                if nb not in out and self.doc_of(nb) == doc:
                    out.append(nb)
        return out

    def doc_of(self, cid: int) -> Optional[str]:
        if cid >= MANUAL_ID_OFFSET:
            return self.man.doc_of(cid) if self.man else None
        m = self.meta.get(cid)
        return m["doc"] if m else None

    def chunk(self, cid: int) -> Dict[str, Any]:
        if cid >= MANUAL_ID_OFFSET:
            return self.man.chunk(cid)
        return self.meta[cid]
