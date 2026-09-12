#!/usr/bin/env python3
"""
retrieval_teto.py — Recupera os chunks da v4 EMBARCADA para as 151 e cacheia em disco.

Reproduz EXATAMENTE o retrieval que roda no aparelho (config embarcada da v4):
  s1 = BM25 gated-expandido índice v4  (indices/index_hf_36nr_expv4.db, classificador NR)
  sT = denso EmbeddingGemma-300M texto+expansão v3  (cache bin_text_rank.json)
  sE = denso EmbeddingGemma-300M expansão-only v4    (cache densev4_exp_rank.json)
  fusão = RRF(k=10, pesos 2,1,2) -> top-K   (config dev-calibrada embarcada)

Paridade confirmada: R@2 51.0% / R@5 65.6% (== retrieval4/verify_bins.py).

Também expõe o TRECHO-OURO (oráculo) por pergunta, para medir o teto de GERAÇÃO
separado do teto de RETRIEVAL (ordem do capitão: reportar oráculo).

Uso (classifier/.venv, servidor GGUF embedding em :8399 só p/ (re)gerar cache exp v4):
  ../classifier/.venv/bin/python retrieval_teto.py --build   # gera data/chunks_v4.json
Comentários PT-BR; código em inglês. Procedência: task poc-prompt-teto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_ROOT / "retrieval4"))
sys.path.insert(0, str(_ROOT / "bench" / "regua"))

from bm25 import HybridBm25  # noqa: E402
from eval_common import load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from corpus.eval_retrieval import check_hit  # noqa: E402
from common import GoldResolver  # noqa: E402 (bench/regua/common.py)

# Config embarcada da v4.
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
RRF_K = 10.0
RRF_W = [2, 1, 2]
POOL = 60

V4_DB = _ROOT / "retrieval4" / "indices" / "index_hf_36nr_expv4.db"
CLF = _ROOT / "classifier" / "models" / "classic_winner.pkl"
DENSE_TEXT = _ROOT / "retrieval3" / "results" / "bin_text_rank.json"
DENSE_EXP_V4 = _ROOT / "retrieval4" / "results" / "densev4_exp_rank.json"
QA = _ROOT / "corpus" / "qa_pairs_v2.jsonl"

CHUNKS_CACHE = _HERE / "data" / "chunks_v4.json"


def _gold(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc": str(qa.get("doc", "") or "").lower(),
        "section": str(qa.get("section", "") or ""),
        "chunk_id": qa.get("chunk_id", -1),
        "relevant_chunk_ids": qa.get("relevant_chunk_ids", []),
    }


def build_cache(top_k: int = 2) -> Dict[str, Any]:
    """Roda a fusão v4 nas 151 e grava chunks recuperados + trecho-ouro por pergunta."""
    ho = load_holdout("qa_v2")
    qa_full = {json.loads(l)["id"]: json.loads(l)
               for l in QA.read_text(encoding="utf-8").splitlines() if l.strip()}
    dt = json.loads(DENSE_TEXT.read_text(encoding="utf-8"))
    de = json.loads(DENSE_EXP_V4.read_text(encoding="utf-8"))
    hy = HybridBm25(str(V4_DB), str(CLF), weights=EXP_W)
    resolver = GoldResolver()

    out: Dict[str, Any] = {}
    n_hit = 0
    for it in ho:
        qa = qa_full[it.id]
        gold = _gold(qa)
        s1 = hy.rank(it.question, POOL)
        fused = rrf_fuse([s1, dt.get(it.id, []), de.get(it.id, [])], RRF_W, k=RRF_K, limit=top_k)
        hit = any(check_hit(c, gold) for c in fused)
        n_hit += int(hit)
        # Trecho-ouro (oráculo) — mesmo resolvedor da régua.
        gold_chunks = resolver.resolve(gold["doc"], gold["section"])
        out[it.id] = {
            "item_id": it.id,
            "question": it.question,
            "doc": gold["doc"],
            "section": gold["section"],
            "golden_answer": qa.get("golden_answer", ""),
            "retrieval_hit": hit,
            "chunks": [{"id": c["id"], "doc": c["doc"], "section": c["section"],
                        "title": c["title"], "text": c["text"]} for c in fused],
            "gold_chunks": [{"id": c["id"], "doc": c["doc"], "section": c["section"],
                             "title": c["title"], "text": c["text"]} for c in gold_chunks],
        }
    meta = {
        "task": "poc-prompt-teto",
        "retriever": "v4_embarcado (BM25v4-gated + dense-text-v3 + dense-exp-v4, RRF k=10 w=2,1,2)",
        "top_k": top_k,
        "n": len(out),
        "recall_at_k_pct": round(100 * n_hit / max(1, len(out)), 1),
    }
    CHUNKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_CACHE.write_text(json.dumps({"metadata": meta, "items": out},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def load_cache() -> Dict[str, Any]:
    return json.loads(CHUNKS_CACHE.read_text(encoding="utf-8"))


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Idêntico ao AskPipeline.kt (Turno 2)."""
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--top-k", type=int, default=2)
    args = ap.parse_args()
    if args.build:
        meta = build_cache(args.top_k)
        print(json.dumps(meta, ensure_ascii=False, indent=1))
    else:
        print("Use --build para gerar o cache. Cache atual:", CHUNKS_CACHE.exists())


if __name__ == "__main__":
    main()
