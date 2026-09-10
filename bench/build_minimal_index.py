"""
build_minimal_index.py — Gera um banco de fallback SQLite FTS5 com NR-10 e NR-06.
Usado para garantir auto-suficiência do módulo bench/ caso corpus/index.db
da trilha T1 ainda não esteja disponível no ambiente.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pypdf


DEFAULT_PDF_DIR = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs")
DEFAULT_OUT_DB = Path("bench/data/minimal_index.db")


def parse_nr_pdf(pdf_path: Path, doc_code: str) -> List[Dict[str, Any]]:
    """Lê o PDF e segmenta em chunks contextuais com cabeçalho de norma e seção."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    chunks = []
    current_section = ""
    current_title = ""
    current_text = []
    current_page = 1

    sec_pattern = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(.*)")

    for p_idx, page in enumerate(reader.pages):
        page_num = p_idx + 1
        text = page.extract_text() or ""
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            m = sec_pattern.match(line_str)
            if m:
                if current_text and len(" ".join(current_text)) > 150:
                    body = " ".join(current_text)
                    header = f"{doc_code.upper()} · {current_section} {current_title} — "
                    chunks.append({
                        "doc": doc_code,
                        "section": current_section or "geral",
                        "title": current_title or "Disposições",
                        "page": current_page,
                        "text": header + body,
                    })
                    current_text = []
                current_section = m.group(1)
                current_title = m.group(2)[:80]
                current_page = page_num
            else:
                current_text.append(line_str)
                if len(" ".join(current_text)) > 1200:
                    body = " ".join(current_text)
                    header = f"{doc_code.upper()} · {current_section} {current_title} — "
                    chunks.append({
                        "doc": doc_code,
                        "section": current_section or "geral",
                        "title": current_title or "Disposições",
                        "page": current_page,
                        "text": header + body,
                    })
                    current_text = []

    if current_text:
        body = " ".join(current_text)
        header = f"{doc_code.upper()} · {current_section} {current_title} — "
        chunks.append({
            "doc": doc_code,
            "section": current_section or "geral",
            "title": current_title or "Disposições",
            "page": current_page,
            "text": header + body,
        })
    return chunks


def build_minimal_db(pdf_dir: Path = DEFAULT_PDF_DIR, out_db: Path = DEFAULT_OUT_DB) -> Path:
    """Gera o banco SQLite FTS5 a partir dos PDFs das NRs 10 e 06."""
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()

    nr10_chunks = parse_nr_pdf(pdf_dir / "nr-10.pdf", "nr-10")
    nr06_chunks = parse_nr_pdf(pdf_dir / "nr-06.pdf", "nr-06")
    all_chunks = nr10_chunks + nr06_chunks

    con = sqlite3.connect(str(out_db))
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc TEXT NOT NULL,
            section TEXT NOT NULL,
            title TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            doc,
            section,
            title,
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
    """)

    cur.execute("""
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, doc, section, title, text)
            VALUES (new.id, new.doc, new.section, new.title, new.text);
        END;
    """)

    records = [
        (c["doc"], c["section"], c["title"], c["page"], c["text"])
        for c in all_chunks
    ]
    cur.executemany(
        "INSERT INTO chunks (doc, section, title, page, text) VALUES (?, ?, ?, ?, ?)",
        records,
    )
    con.commit()
    cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    con.commit()
    con.close()

    print(f"Banco gerado com sucesso: {out_db} ({len(all_chunks)} chunks, {out_db.stat().st_size} bytes)")
    return out_db


if __name__ == "__main__":
    build_minimal_db()
