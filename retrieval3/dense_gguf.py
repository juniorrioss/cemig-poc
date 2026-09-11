#!/usr/bin/env python3
"""
dense_gguf.py — Reencoda os índices densos com o MESMO encoder GGUF que o app usará
on-device (EmbeddingGemma-300M QAT-Q4_0 via llama.cpp embedding mode), para medir a
PARIDADE de recall contra o índice de referência (SentenceTransformer FP32 com as 2
Dense heads 768→3072→768).

Motivo (crítico): o GGUF do llama.cpp NÃO inclui as Dense heads do wrapper
SentenceTransformer — só o transformer + mean pooling + norm (dim 768). Portanto os
vetores do GGUF são um espaço DIFERENTE do índice de referência; para o app funcionar,
os índices de documento têm de ser reencodados com o MESMO GGUF (query e documento no
mesmo espaço). Aqui produzimos esses índices e os rankings honestos.

Usa o llama-server em modo embedding (--embedding --pooling mean) via HTTP, aplicando os
prompts oficiais do EmbeddingGemma:
  query    -> "task: search result | query: {text}"
  document -> "title: none | text: {text}"

Saída: rankings por pergunta do holdout (compatível com eval_dense.py/eval_fusion_final.py).

Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "classifier"))

from eval_common import load_holdout  # noqa: E402

QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "


def embed_batch(url: str, texts: List[str], timeout: int = 120) -> np.ndarray:
    """Chama o llama-server (embeddings). Retorna matriz L2-normalizada (n, dim)."""
    payload = {"input": texts, "model": "emb"}
    r = requests.post(f"{url}/v1/embeddings", json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    # A ordem do retorno segue index; ordena por 'index' p/ garantir.
    rows = sorted(d["data"], key=lambda x: x["index"])
    embs = np.array([row["embedding"] for row in rows], dtype=np.float32)
    # Renormaliza por garantia (o servidor já normaliza).
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def load_chunks(db_path: str, expansions: Optional[Dict[int, str]], expansion_only: bool):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks ORDER BY id").fetchall()]
    con.close()
    texts = []
    for r in rows:
        ex = expansions.get(r["id"], "") if expansions else ""
        if expansion_only:
            texts.append(ex or r["text"])
        elif ex:
            texts.append(r["text"] + " " + ex)
        else:
            texts.append(r["text"])
    return rows, texts


def load_expansions(path: str) -> Dict[int, str]:
    exp: Dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        exp[r["id"]] = " ".join(r.get("perguntas", []) + r.get("sinonimos", []))
    return exp


def encode_docs(url: str, texts: List[str], batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = [DOC_PROMPT + t for t in texts[i:i + batch]]
        out.append(embed_batch(url, chunk))
        if (i // batch) % 5 == 0:
            print(f"  docs {i + len(chunk)}/{len(texts)}", file=sys.stderr)
    return np.vstack(out)


def build_and_rank(url: str, src_db: str, expansions_path: str, expansion_only: bool,
                   with_expansion: bool, out_rank: str, batch: int, pool: int) -> Dict:
    exp = None
    if with_expansion or expansion_only:
        exp = load_expansions(expansions_path)
    rows, texts = load_chunks(src_db, exp, expansion_only)

    t0 = time.time()
    doc_emb = encode_docs(url, texts, batch)  # (N, 768) normalizado
    enc_docs_s = time.time() - t0

    holdout = load_holdout("qa_v2")
    q_texts = [QUERY_PROMPT + it.question for it in holdout]
    t0 = time.time()
    q_emb = embed_batch(url, q_texts)  # (Q, 768)
    enc_q_s = time.time() - t0

    # Ranking por cosseno (vetores normalizados -> produto interno).
    ids = np.array([r["id"] for r in rows])
    ranks: Dict[str, List[Dict]] = {}
    sims_all = q_emb @ doc_emb.T  # (Q, N)
    for qi, it in enumerate(holdout):
        sims = sims_all[qi]
        top = np.argsort(-sims)[:pool]
        ranks[it.id] = [
            {"id": int(ids[j]), "doc": rows[j]["doc"], "section": rows[j]["section"],
             "title": rows[j]["title"], "text": rows[j]["text"], "score": float(sims[j])}
            for j in top
        ]
    Path(out_rank).write_text(json.dumps(ranks, ensure_ascii=False), encoding="utf-8")
    return {
        "n_chunks": len(rows), "n_queries": len(holdout), "dim": int(doc_emb.shape[1]),
        "encode_docs_s": round(enc_docs_s, 1), "encode_queries_s": round(enc_q_s, 2),
        "encode_query_ms_gpu": round(enc_q_s / max(1, len(holdout)) * 1000, 1),
        "out_rank": out_rank,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8399")
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--with-expansion", action="store_true")
    ap.add_argument("--expansion-only", action="store_true")
    ap.add_argument("--out-rank", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()
    meta = build_and_rank(args.url, args.src, args.expansions, args.expansion_only,
                          args.with_expansion, args.out_rank, args.batch, args.pool)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
