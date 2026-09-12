#!/usr/bin/env python3
"""
build_index_v4.py — FRENTE B: índice FTS5 com expansão v3 + expansão v4 (mais rica).

Constrói dois índices para medir o INCREMENTO isolado da expansão v4:
  --mode v3only : só a expansão v3 (perguntas + sinônimos)          [= índice da v3]
  --mode v4     : expansão v3 + v4 (verbalizacoes + asr + sinonimos2) no MESMO campo
                  `expansion` (o app já lê 5 campos; peso do campo não muda).

Mesmo schema/tokenizador de produção (unicode61 remove_diacritics 2), mesmos 5 campos,
para o Fts5Retriever/HybridRetriever do app funcionar sem mudança de código.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python build_index_v4.py --mode v4 \
     --out indices/index_hf_36nr_expv4.db

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
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


def load_v3(path: Path) -> Dict[int, List[str]]:
    """Perguntas + sinônimos da expansão v3."""
    exp: Dict[int, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        parts = list(r.get("perguntas", [])) + list(r.get("sinonimos", []))
        exp[r["id"]] = parts
    return exp


def load_v4(path: Path) -> Dict[int, List[str]]:
    """Verbalizacoes + asr + sinonimos2 da expansão v4."""
    exp: Dict[int, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        parts = (list(r.get("verbalizacoes", [])) + list(r.get("asr", []))
                 + list(r.get("sinonimos2", [])))
        exp[r["id"]] = parts
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--v3", default=str(_ROOT / "retrieval3" / "data" / "expansions.jsonl"))
    ap.add_argument("--v4", default=str(_HERE / "data" / "expansions_v4.jsonl"))
    ap.add_argument("--mode", choices=["v3only", "v4"], default="v4")
    ap.add_argument("--out", default=str(_HERE / "indices" / "index_hf_36nr_expv4.db"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    v3 = load_v3(Path(args.v3))
    v4 = load_v4(Path(args.v4)) if args.mode == "v4" else {}

    src = sqlite3.connect(args.src)
    src.row_factory = sqlite3.Row
    rows = [dict(r) for r in src.execute(
        "SELECT id, doc, section, title, page, text FROM chunks ORDER BY id").fetchall()]
    src.close()

    def expansion_for(cid: int) -> str:
        parts = list(v3.get(cid, []))
        if args.mode == "v4":
            parts += list(v4.get(cid, []))
        return " ".join(parts)

    covered_v3 = sum(1 for r in rows if r["id"] in v3)
    covered_v4 = sum(1 for r in rows if r["id"] in v4)
    print(f"mode={args.mode} chunks={len(rows)} v3-cov={covered_v3} v4-cov={covered_v4}")

    dst = sqlite3.connect(str(out_path))
    dst.executescript(SCHEMA)
    dst.executemany(
        "INSERT INTO chunks (id, doc, section, title, page, text, expansion) "
        "VALUES (:id, :doc, :section, :title, :page, :text, :expansion)",
        [{**r, "expansion": expansion_for(r["id"])} for r in rows])
    dst.commit()
    dst.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    dst.commit()
    dst.close()
    print(f"Índice: {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
