#!/usr/bin/env python3
"""
build_dense_gguf_index.py — Constrói os índices sqlite-vec ON-DEVICE a partir do encoder
GGUF (EmbeddingGemma-300M QAT-Q4_0 via llama.cpp embedding mode). São os índices que vão
nos assets do app: os vetores dos DOCUMENTOS têm de estar no MESMO espaço que a query
codificada on-device (por isso reencodamos com o GGUF, não com o SentenceTransformer FP32).

Gera dois índices vec0(float[768]) espelhando a entrega v3:
  - dense_text_gguf.db     : texto normativo + expansão coloquial   (query↔passagem)
  - dense_exponly_gguf.db  : só a expansão coloquial                (query↔query)

Formato idêntico ao dense.py (tabela meta + virtual table vec) para reuso no Android
(mesmo motor sqlite-vec arm64). Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

DOC_PROMPT = "title: none | text: "


def _pack_f32(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())


def embed_docs(url: str, texts: List[str], batch: int, timeout: int = 120) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        payload = {"input": [DOC_PROMPT + t for t in texts[i:i + batch]], "model": "emb"}
        r = requests.post(f"{url}/v1/embeddings", json=payload, timeout=timeout)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda x: x["index"])
        out.append(np.array([row["embedding"] for row in rows], dtype=np.float32))
        if (i // batch) % 20 == 0:
            print(f"  {i + len(rows)}/{len(texts)}", file=sys.stderr)
    emb = np.vstack(out)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


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


def load_chunks(db_path: str, exp: Optional[Dict[int, str]], expansion_only: bool):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks ORDER BY id").fetchall()]
    con.close()
    texts = []
    for r in rows:
        ex = exp.get(r["id"], "") if exp else ""
        if expansion_only:
            texts.append(ex or r["text"])
        elif ex:
            texts.append(r["text"] + " " + ex)
        else:
            texts.append(r["text"])
    return rows, texts


def build(url: str, src_db: str, expansions: str, expansion_only: bool,
          out_db: str, batch: int) -> Dict:
    import sqlite_vec
    exp = load_expansions(expansions)
    rows, texts = load_chunks(src_db, exp, expansion_only)
    t0 = time.time()
    emb = embed_docs(url, texts, batch)
    enc_s = time.time() - t0
    dim = emb.shape[1]

    out = Path(out_db)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(str(out))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY, doc TEXT, section TEXT, title TEXT, text TEXT)")
    con.executemany("INSERT INTO meta VALUES (?,?,?,?,?)",
                    [(c["id"], c["doc"], c["section"], c["title"], c["text"]) for c in rows])
    con.execute(f"CREATE VIRTUAL TABLE vec USING vec0(embedding float[{dim}])")
    con.executemany("INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                    [(c["id"], _pack_f32(v)) for c, v in zip(rows, emb)])
    con.commit()
    con.close()
    return {"out_db": out_db, "n_chunks": len(rows), "dim": int(dim),
            "index_mb": round(out.stat().st_size / 1e6, 2), "encode_docs_s": round(enc_s, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8399")
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--expansion-only", action="store_true")
    ap.add_argument("--out-db", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    meta = build(args.url, args.src, args.expansions, args.expansion_only, args.out_db, args.batch)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
