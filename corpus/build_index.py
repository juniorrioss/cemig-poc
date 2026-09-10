"""
build_index.py — Criação do índice SQLite FTS5 (index.db) para retrieval BM25 embutido.

Gera:
1. Tabela 'chunks(id, doc, section, title, page, text)'
2. Tabela virtual 'chunks_fts' usando FTS5 com content='chunks' e tokenizer 'unicode61 remove_diacritics 2'
3. Triggers de sincronização automática
4. Consulta de referência com ORDER BY bm25(...)
"""

import argparse
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path("corpus/index.db")
DEFAULT_CHUNKS_PATH = Path("corpus/data/chunks.json")

# Stopwords mínimas em português para higienização opcional de consultas
PORTUGUESE_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos",
    "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas",
    "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", "foram",
    "ha", "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo",
    "meu", "meus", "minha", "minhas", "muito", "na", "nas", "não", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
    "se", "seja", "sem", "só", "sua", "suas", "seu", "seus", "também", "te", "tem",
    "têm", "temos", "ter", "teu", "teus", "tua", "tuas", "um", "uma", "você", "vocês"
}


def sanitize_fts_token(token: str) -> Optional[str]:
    """Sanitiza um termo para sintaxe FTS5, escapando caracteres especiais."""
    # Remove pontuações estranhas das bordas
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', token.strip())
    if not cleaned:
        return None
    # Palavras com hífens ou pontos numéricos (ex: nr-10, 10.2.8) são encapsuladas em aspas duplas
    return f'"{cleaned}"'


def format_fts_query(raw_query: str, mode: str = "OR", filter_stopwords: bool = False) -> str:
    """Prepara a query FTS5 a partir de texto bruto ou lista de palavras-chave."""
    raw_tokens = re.findall(r'[\w\.-]+', raw_query.lower())
    valid_tokens = []

    for tok in raw_tokens:
        tok_clean = re.sub(r'^[^\w]+|[^\w]+$', '', tok)
        if not tok_clean:
            continue
        if filter_stopwords and tok_clean in PORTUGUESE_STOPWORDS and len(raw_tokens) > 2:
            continue
        valid_tokens.append(f'"{tok_clean}"')

    if not valid_tokens:
        return '""'

    joiner = f" {mode.upper()} "
    return joiner.join(valid_tokens)


def init_database(con: sqlite3.Connection) -> None:
    """Cria a tabela de conteúdo chunks e a tabela virtual FTS5 com triggers."""
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS chunks_fts;")
    cur.execute("DROP TABLE IF EXISTS chunks;")

    # Tabela canônica de chunks
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

    # Virtual table FTS5 com content='chunks' e tokenizer unicode61 sem acentos
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

    # Triggers de sincronização entre chunks e chunks_fts
    cur.execute("""
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, doc, section, title, text)
            VALUES (new.id, new.doc, new.section, new.title, new.text);
        END;
    """)

    cur.execute("""
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, doc, section, title, text)
            VALUES ('delete', old.id, old.doc, old.section, old.title, old.text);
        END;
    """)

    cur.execute("""
        CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, doc, section, title, text)
            VALUES ('delete', old.id, old.doc, old.section, old.title, old.text);
            INSERT INTO chunks_fts(rowid, doc, section, title, text)
            VALUES (new.id, new.doc, new.section, new.title, new.text);
        END;
    """)

    con.commit()


def populate_chunks(con: sqlite3.Connection, chunks_data: List[Dict[str, Any]]) -> int:
    """Insere a lista de chunks no banco de dados SQLite."""
    cur = con.cursor()
    records = []
    for c in chunks_data:
        records.append((
            c["doc"],
            c["section"],
            c["title"],
            int(c["page"]),
            c["text"],
        ))

    cur.executemany("""
        INSERT INTO chunks (doc, section, title, page, text)
        VALUES (?, ?, ?, ?, ?);
    """, records)

    con.commit()
    logger.info("Inseridos %d chunks na tabela chunks (e indexados no FTS5)", len(records))
    return len(records)


def build_index(chunks_path: str | Path = DEFAULT_CHUNKS_PATH, db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Carrega os chunks do JSON e gera o arquivo index.db SQLite FTS5."""
    chunks_path = Path(chunks_path)
    db_path = Path(db_path)

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Arquivo de chunks não encontrado: {chunks_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    con = sqlite3.connect(str(db_path))
    try:
        init_database(con)
        populate_chunks(con, chunks_data)
        # Otimiza o índice FTS5
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
        con.commit()
    finally:
        con.close()

    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    logger.info("Banco de dados SQLite FTS5 gerado: %s (%.2f MB)", db_path, db_size_mb)
    return db_path


def search_reference(
    db_path: str | Path,
    query: str,
    top_k: int = 5,
    mode: str = "OR",
    filter_stopwords: bool = True,
    weights: Tuple[float, float, float, float] = (1.5, 3.0, 2.0, 1.0),
) -> List[Dict[str, Any]]:
    """Executa a consulta de referência com BM25 no SQLite FTS5."""
    fts_query = format_fts_query(query, mode=mode, filter_stopwords=filter_stopwords)
    if fts_query == '""':
        return []

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    w_doc, w_sec, w_title, w_text = weights
    sql = f"""
        SELECT
            c.id,
            c.doc,
            c.section,
            c.title,
            c.page,
            c.text,
            bm25(chunks_fts, {w_doc}, {w_sec}, {w_title}, {w_text}) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC
        LIMIT ?;
    """

    try:
        cur.execute(sql, (fts_query, top_k))
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "doc": r["doc"],
                "section": r["section"],
                "title": r["title"],
                "page": r["page"],
                "text": r["text"],
                "bm25_score": r["score"],
            })
        return results
    except sqlite3.OperationalError as e:
        logger.warning("Erro FTS5 na consulta '%s' (formatada: '%s'): %s", query, fts_query, e)
        return []
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o banco index.db (SQLite FTS5) com tabela chunks e BM25.")
    parser.add_argument("--chunks-file", type=str, default=None, help="Arquivo JSON de chunks para indexação.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco index.db gerado.")
    parser.add_argument("--rebuild", action="store_true", help="Força a reconstrução do banco mesmo se já existir.")
    parser.add_argument("--query", type=str, default=None, help="Executa consulta de teste e imprime resultados.")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados a retornar.")
    parser.add_argument("--mode", type=str, default="OR", choices=["OR", "AND"], help="Modo booleano da consulta.")
    args = parser.parse_args()

    db_path = Path(args.db)
    chunks_path = Path(args.chunks_file) if args.chunks_file else DEFAULT_CHUNKS_PATH

    if not db_path.exists() or args.rebuild or (args.chunks_file and not args.query):
        build_index(chunks_path=chunks_path, db_path=db_path)

    if args.query:
        print(f"\n--- Buscando: '{args.query}' (modo {args.mode}) ---")
        results = search_reference(db_path, args.query, top_k=args.top_k, mode=args.mode)
        if not results:
            print("Nenhum resultado encontrado.")
        for idx, res in enumerate(results, 1):
            print(f"\n#{idx} [Score: {res['bm25_score']:.4f}] ID {res['id']} | {res['doc'].upper()} | Seção {res['section']} (Pág {res['page']})")
            print(f"Título: {res['title']}")
            print(f"Texto: {res['text'][:220]}...")


if __name__ == "__main__":
    main()
