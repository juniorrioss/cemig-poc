#!/usr/bin/env python3
"""
build_expanded_index.py — Etapa 1: constrói índice FTS5 com campo `expansion`.

Copia os 2202 chunks do índice de produção e adiciona uma 5ª coluna FTS5
`expansion` com o material coloquial gerado offline (perguntas de campo + sinônimos
leigos), permitindo casar a fala do operário com o chunk normativo certo.

O peso do campo `expansion` no bm25() é calibrável (Fts5Retriever suporta N pesos).
Mantemos o mesmo tokenizador de produção: unicode61 remove_diacritics 2.

Uso:
    python3 retrieval3/build_expanded_index.py \
        --src corpus/index_hf_36nr.db \
        --expansions retrieval3/data/expansions.jsonl \
        --out retrieval3/indices/index_hf_36nr_exp.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc TEXT NOT NULL,
    section TEXT NOT NULL,
    title TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    expansion TEXT NOT NULL DEFAULT ''
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    doc, section, title, text, expansion,
    content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, doc, section, title, text, expansion)
    VALUES (new.id, new.doc, new.section, new.title, new.text, new.expansion);
END;
"""


def load_expansions(path: Path) -> Dict[int, str]:
    """Concatena perguntas + sinônimos por chunk id num único blob de texto."""
    exp: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        parts: List[str] = []
        parts += r.get("perguntas", [])
        parts += r.get("sinonimos", [])
        exp[r["id"]] = " ".join(parts)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--expansions", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--out", default=str(_HERE / "indices" / "index_hf_36nr_exp.db"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    exp = load_expansions(Path(args.expansions))
    src = sqlite3.connect(args.src)
    src.row_factory = sqlite3.Row
    rows = [dict(r) for r in src.execute(
        "SELECT id, doc, section, title, page, text FROM chunks ORDER BY id").fetchall()]
    src.close()

    covered = sum(1 for r in rows if r["id"] in exp)
    print(f"chunks={len(rows)} com expansão={covered} ({100*covered/len(rows):.1f}%)")

    dst = sqlite3.connect(str(out_path))
    dst.executescript(SCHEMA)
    dst.executemany(
        "INSERT INTO chunks (id, doc, section, title, page, text, expansion) "
        "VALUES (:id, :doc, :section, :title, :page, :text, :expansion)",
        [{**r, "expansion": exp.get(r["id"], "")} for r in rows],
    )
    dst.commit()
    dst.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    dst.commit()
    dst.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"Índice gravado: {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
