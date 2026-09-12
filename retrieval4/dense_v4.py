#!/usr/bin/env python3
"""
dense_v4.py — FRENTE B: reencoda os índices densos com a expansão v4 (encoder GGUF do app).

Usa o MESMO encoder que roda on-device (EmbeddingGemma-300M QAT-Q4_0 via llama-server em
modo embedding), reencodando os documentos com a expansão v4 mais rica. Produz dois caches
de ranking do holdout 151 (compatíveis com eval_recall_v4/rrf):
  - text : DOC = texto normativo + expansão(v3+v4)          (query↔passagem)
  - exp  : DOC = expansão(v3+v4) only                        (query↔query coloquial)

Prompts oficiais do EmbeddingGemma:
  query    -> "task: search result | query: {text}"
  document -> "title: none | text: {text}"

Higiene: não-interativo, timeout, batch, procedência. NÃO usa a GPU além do servidor.

Uso (classifier/.venv), com o llama-server GGUF embedding já no ar (porta 8399):
  ../classifier/.venv/bin/python dense_v4.py --which text --out results/densev4_text_rank.json
  ../classifier/.venv/bin/python dense_v4.py --which exp  --out results/densev4_exp_rank.json

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
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
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

from eval_common import load_holdout  # noqa: E402

QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "


def embed_batch(url: str, texts: List[str], timeout: int = 180) -> np.ndarray:
    r = requests.post(f"{url}/v1/embeddings", json={"input": texts, "model": "emb"}, timeout=timeout)
    r.raise_for_status()
    rows = sorted(r.json()["data"], key=lambda x: x["index"])
    embs = np.array([row["embedding"] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def load_exp(path: Path, kind: str) -> Dict[int, str]:
    """kind='v3' -> perguntas+sinonimos; kind='v4' -> verbalizacoes+asr+sinonimos2."""
    exp: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        if kind == "v3":
            parts = list(r.get("perguntas", [])) + list(r.get("sinonimos", []))
        else:
            parts = (list(r.get("verbalizacoes", [])) + list(r.get("asr", []))
                     + list(r.get("sinonimos2", [])))
        exp[r["id"]] = " ".join(parts)
    return exp


def load_chunks(db_path: str) -> List[Dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks ORDER BY id").fetchall()]
    con.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8399")
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--v3", default=str(_ROOT / "retrieval3" / "data" / "expansions.jsonl"))
    ap.add_argument("--v4", default=str(_HERE / "data" / "expansions_v4.jsonl"))
    ap.add_argument("--which", choices=["text", "exp"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    e3 = load_exp(Path(args.v3), "v3")
    e4 = load_exp(Path(args.v4), "v4")
    rows = load_chunks(args.src)

    texts: List[str] = []
    for r in rows:
        ex = (e3.get(r["id"], "") + " " + e4.get(r["id"], "")).strip()
        if args.which == "exp":
            texts.append(ex or r["text"])
        else:
            texts.append(r["text"] + " " + ex if ex else r["text"])

    t0 = time.time()
    doc_emb_parts = []
    for i in range(0, len(texts), args.batch):
        batch = [DOC_PROMPT + t for t in texts[i:i + args.batch]]
        doc_emb_parts.append(embed_batch(args.url, batch))
        if (i // args.batch) % 10 == 0:
            print(f"  docs {i + len(batch)}/{len(texts)}", file=sys.stderr, flush=True)
    doc_emb = np.vstack(doc_emb_parts)
    enc_docs_s = time.time() - t0

    ho = load_holdout("qa_v2")
    q_emb = embed_batch(args.url, [QUERY_PROMPT + it.question for it in ho])

    ids = np.array([r["id"] for r in rows])
    sims_all = q_emb @ doc_emb.T
    ranks: Dict[str, List[Dict]] = {}
    for qi, it in enumerate(ho):
        sims = sims_all[qi]
        top = np.argsort(-sims)[:args.pool]
        ranks[it.id] = [
            {"id": int(ids[j]), "doc": rows[j]["doc"], "section": rows[j]["section"],
             "title": rows[j]["title"], "text": rows[j]["text"], "score": float(sims[j])}
            for j in top]
    Path(args.out).write_text(json.dumps(ranks, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"which": args.which, "n_chunks": len(rows), "dim": int(doc_emb.shape[1]),
                      "encode_docs_s": round(enc_docs_s, 1), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
