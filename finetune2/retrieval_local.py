#!/usr/bin/env python3
"""
retrieval_local.py — retriever v3 (fusão RRF 3-sinais) do finetune2, robusto a artefatos
gitignored ausentes. Idêntico em semântica ao bench/judge151/retrieval_v3.py, mas:

  - usa os caches densos COMMITADOS `retrieval3/results/bin_{text,exponly}_rank.json` (formato
    on-device DVEC1, R@2 51% — a paridade real do app) quando os caches gemma768 gitignored não
    existem. Estes cobrem o holdout 151 (keyed por qa-id);
  - fallback honesto para BM25-gated-exp puro quando NÃO há sinal denso para o item_id (ex.:
    perguntas do dev/dpo_questions que não estão nos caches) — o app faz o mesmo fallback quando
    o encoder denso está indisponível.

Motivação: o worktree irmão que hospedava os venvs/índices/caches densos gemma768 foi apagado
no meio da tarefa; reconstruí classic_winner.pkl (train_classic) e index_hf_36nr_exp.db
(build_expanded_index) a partir de artefatos commitados, e aqui reuso os caches densos commitados.

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

from bm25 import HybridBm25  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from corpus.eval_retrieval import check_hit  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
RRF_K = 30.0
POOL = 60
TOP_K = 2

_EXP_DB = _ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db"
_CLF = _ROOT / "classifier" / "models" / "classic_winner.pkl"
# Caches densos: preferir gemma768 (se existir); senão os bin_* commitados (DVEC1 on-device).
_DENSE_TEXT_CANDS = [
    _ROOT / "retrieval3" / "results" / "dense_rank_gemma768_exp.json",
    _ROOT / "retrieval3" / "results" / "bin_text_rank.json",
]
_DENSE_EXP_CANDS = [
    _ROOT / "retrieval3" / "results" / "dense_rank_gemma768_exponly.json",
    _ROOT / "retrieval3" / "results" / "bin_exponly_rank.json",
]


def _first_existing(cands: List[Path]) -> Optional[Path]:
    for c in cands:
        if c.exists():
            return c
    return None


class RetrieverLocal:
    """Fusão RRF 3-sinais com caches densos commitados + fallback BM25 honesto."""

    def __init__(self) -> None:
        self.hy = HybridBm25(str(_EXP_DB), str(_CLF), weights=EXP_W)
        self.clf_name = self.hy.clf.name
        dt = _first_existing(_DENSE_TEXT_CANDS)
        de = _first_existing(_DENSE_EXP_CANDS)
        self.dense_text: Dict[str, List[Dict[str, Any]]] = (
            json.loads(dt.read_text(encoding="utf-8")) if dt else {})
        self.dense_exp: Dict[str, List[Dict[str, Any]]] = (
            json.loads(de.read_text(encoding="utf-8")) if de else {})
        self.dense_source = {"text": dt.name if dt else None, "exp": de.name if de else None}

    def search(self, raw_question: str, top_k: int = TOP_K,
               item_id: Optional[str] = None) -> Dict[str, Any]:
        s1 = self.hy.rank(raw_question, POOL)
        sT = self.dense_text.get(item_id, []) if item_id else []
        sE = self.dense_exp.get(item_id, []) if item_id else []

        topk, scores = self.hy.clf.predict([raw_question], k=2)
        preds = list(topk[0])
        p1 = float(scores[0].max())
        from nr_taxonomy import NONE_LABEL
        if p1 >= self.hy.conf_thr and preds and preds[0] != NONE_LABEL:
            mode = "hard"
        elif preds and preds[0] == NONE_LABEL:
            mode = "none"
        else:
            mode = "soft"

        # Se não há sinal denso p/ este item (fora dos caches commitados), fallback = BM25-gated-exp.
        signals = [s for s in (s1, sT, sE) if s]
        fused = rrf_fuse(signals, [1.0] * len(signals), k=RRF_K, limit=top_k)

        chunks = [dict(r) for r in fused]
        decision = {
            "top1": preds[0] if preds else NONE_LABEL,
            "top1_prob": round(p1, 4),
            "top2": preds[1] if len(preds) > 1 else NONE_LABEL,
            "mode": mode,
            "fusion": "rrf3" if (sT or sE) else "bm25_only_fallback",
            "dense_available": bool(sT or sE),
        }
        return {"chunks": chunks, "decision": decision, "fts_query": raw_question}


def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def retrieval_hit(chunks: List[Dict[str, Any]], gold: Dict[str, Any]) -> bool:
    return any(check_hit(c, gold) for c in chunks)


if __name__ == "__main__":
    r = RetrieverLocal()
    print("dense source:", r.dense_source)
    out = r.search("posso trabalhar so com luva de borracha na subestacao?", top_k=2, item_id="qa-001")
    print("mode:", out["decision"]["mode"], "fusion:", out["decision"]["fusion"])
    print("chunks:", [(c["doc"], c["section"]) for c in out["chunks"]])
