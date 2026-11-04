#!/usr/bin/env python3
"""
manuals.py — Frente 4: os 9 MANUAIS como "aba separada" no RRF.

Lição de 2026 (corpus v2 / docai): despejar manuais no BM25 das normas DILUI (R@5 −23 p.p.).
Aqui eles NUNCA entram no índice das normas: ficam numa base FTS5 PRÓPRIA
(cache/manuals.db, mesmo tokenizer unicode61 remove_diacritics 2, mesma query do app)
+ um índice denso PRÓPRIO (EmbeddingGemma GGUF, prompt de documento oficial), e entram
no RRF como sinais extras com peso próprio (retriever_f5.Cfg.manual_w).

Chunking: parágrafos do texto do manual (nrs.parquet/manual) agrupados até ~1500 chars
(média dos chunks de norma = 1488), offsets NATIVOS (sem realinhamento); parágrafos de
ruído de diagramação ("*.indd", sumário com pontilhado) são descartados. Prefixo de
exibição "NR-XX · Manual — " (citável, como o das normas).

Uso (build): classifier/.venv/bin/python bench/lfm3_f5/manuals.py   (llama-server :8399)
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, REPO / "classifier", REPO / "retrieval3", HERE):
    sys.path.insert(0, str(p))

import common_f5 as C  # noqa: E402
from eval_common import app_fts_query  # noqa: E402

DB = C.CACHE / "manuals.db"
VEC = C.CACHE / "manuals_dense.npz"
TARGET = 1500
MAXC = 2200
OFFSET = 1_000_000
POOL = 60
_NOISE = re.compile(r"(\.indd\b|\.{6,}|^\s*\d+\s*$)")


def chunk_manual(text: str) -> List[Dict[str, Any]]:
    paras = [(m.start(), m.end()) for m in re.finditer(r"[^\n](?:.|\n(?!\n))*", text)]
    out, cur = [], []
    for s, e in paras:
        p = text[s:e].strip()
        if not p or _NOISE.search(p) or len(p) < 3:
            continue
        cur.append((s, e))
        if sum(b - a for a, b in cur) >= TARGET:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    res = []
    for grp in out:
        # quebra grupos longos demais (parágrafo gigante) em janelas de MAXC
        spans = []
        for s, e in grp:
            while e - s > MAXC:
                spans.append((s, s + MAXC))
                s += MAXC
            spans.append((s, e))
        res.append(spans)
    return res


def build() -> None:
    S = C.load_sources()
    C.CACHE.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(str(DB))
    con.executescript("""
    CREATE TABLE mchunks(id INTEGER PRIMARY KEY, doc TEXT, section TEXT, title TEXT,
                         page INTEGER, text TEXT, file TEXT, spans TEXT);
    CREATE VIRTUAL TABLE mchunks_fts USING fts5(doc, section, title, text,
        content='mchunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
    """)
    cid = OFFSET
    rows = []
    for doc, mf in sorted(S["manual_file_of_doc"].items()):
        text = S["by_file"][mf]
        title = re.sub(r"^#\s*", "", text.split("\n", 1)[0]).strip()[:120]
        nr = "NR-%02d" % int(doc.split("-")[1])
        for spans in chunk_manual(text):
            body = " ".join(re.sub(r"\s+", " ", text[s:e]).strip() for s, e in spans)
            disp = f"{nr} · Manual {title} — {body}"
            rows.append((cid, doc, "manual", title, 0, disp, mf,
                         ";".join(f"{s},{e}" for s, e in spans)))
            cid += 1
    con.executemany("INSERT INTO mchunks VALUES (?,?,?,?,?,?,?,?)", rows)
    con.execute("INSERT INTO mchunks_fts(rowid, doc, section, title, text) "
                "SELECT id, doc, section, title, text FROM mchunks")
    con.commit()
    con.close()
    print(f"[manuals] {len(rows)} chunks de {len(S['manual_file_of_doc'])} manuais -> {DB}")
    # denso próprio (mesmo GGUF do app; prompt oficial de documento)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    vecs = []
    texts = [C.DOC_PROMPT + r[5] for r in rows]
    for i in range(0, len(texts), 16):
        r = requests.post(f"{C.EMB_URL}/v1/embeddings",
                          json={"input": texts[i:i + 16], "model": "emb"}, timeout=600)
        r.raise_for_status()
        d = sorted(r.json()["data"], key=lambda x: x["index"])
        e = np.array([x["embedding"] for x in d], dtype=np.float32)
        vecs.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9))
    np.savez(VEC, ids=ids, vecs=np.vstack(vecs))
    print(f"[manuals] denso {len(ids)}x{vecs[0].shape[1]} -> {VEC}")


class ManualIndex:
    def __init__(self):
        self.con = sqlite3.connect(str(DB), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.meta = {r["id"]: dict(r) for r in self.con.execute("SELECT * FROM mchunks")}
        z = np.load(VEC)
        self.ids, self.mat = z["ids"], z["vecs"]

    @classmethod
    def load(cls) -> "ManualIndex":
        if not DB.exists() or not VEC.exists():
            build()
        return cls()

    def bm25(self, q: str, pool: int = POOL) -> List[Dict[str, Any]]:
        fq = app_fts_query(q)
        if fq == '""':
            return []
        try:
            cur = self.con.execute(
                "SELECT m.id, bm25(mchunks_fts, 1.5, 3.0, 2.0, 1.0) AS score FROM mchunks_fts "
                "JOIN mchunks m ON m.id = mchunks_fts.rowid WHERE mchunks_fts MATCH ? "
                "ORDER BY score LIMIT ?", (fq, pool))
        except sqlite3.OperationalError:
            return []
        return [{"id": r["id"], "score": r["score"]} for r in cur]

    def dense(self, qv: np.ndarray, pool: int = POOL) -> List[Dict[str, Any]]:
        sims = self.mat @ qv
        top = np.argsort(-sims)[:pool]
        return [{"id": int(self.ids[j]), "score": float(sims[j])} for j in top]

    def signals(self, q: str, qv: np.ndarray):
        return self.bm25(q), self.dense(qv)

    def doc_of(self, cid: int):
        m = self.meta.get(cid)
        return m["doc"] if m else None

    def chunk(self, cid: int) -> Dict[str, Any]:
        return self.meta[cid]

    def span_store(self) -> Dict[int, Dict[str, Any]]:
        return {cid: {"file": m["file"],
                      "spans": [[int(a), int(b)] for a, b in
                                (x.split(",") for x in m["spans"].split(";"))]}
                for cid, m in self.meta.items()}


if __name__ == "__main__":
    build()
