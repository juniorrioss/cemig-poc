"""
06_replace_ceiling.py — Config REPLACE (não-aditiva) + teto do tokenizer.

Isola a variável de verdade:
  - REPLACE: troca os chunks corrompidos do Anexo II da NR-10 (glifos/colunas-fantasma)
    pela frase estruturada e verificada do Document AI. NÃO despeja tabelas genéricas no
    BM25 (evita a poluição -2.0 R@2 medida no aditivo-23, mesmo efeito dos manuais).
  - TETO DO TOKENIZER: o caminho do app (app_fts_query) DESCARTA '13,8' (a vírgula quebra
    o token) e 'kV' (len 2). Comparamos a busca com o tokenizer do app vs um tokenizer
    "numeric-preserving" (mantém '13,8' entre aspas) para mostrar o ganho REAL destravável.

Roda o caso do capitão (distância 13,8 kV) nas duas variantes e nos dois índices.

Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from corpus.eval_fino import app_fts_query, _strip_accents, APP_STOPWORDS
from corpus.eval_retrieval import check_hit

WEIGHTS = (1.5, 3.0, 2.0, 1.0)


def numeric_preserving_query(raw_query: str) -> str:
    """Tokenizer variante: preserva números pt-BR ('13,8', '0,38') e siglas curtas (kV) como termos exatos."""
    normalized = _strip_accents(raw_query).lower()
    # Captura números decimais pt-BR ANTES de quebrar em tokens.
    decimals = re.findall(r"\d+,\d+", raw_query)
    raw_tokens = re.findall(r"[\w.-]+", normalized)
    parts: List[str] = []
    for tok in raw_tokens:
        if tok in APP_STOPWORDS:
            continue
        if tok == "kv":
            parts.append('"kv"')
            continue
        if len(tok) <= 2:
            continue
        stem = tok[:6] if len(tok) > 6 else tok
        parts.append(f'"{stem}"*' if ("-" in stem or "." in stem) else f"{stem}*")
    for d in decimals:
        parts.append(f'"{d}"')
    return " OR ".join(dict.fromkeys(parts)) if parts else '""'


def build_replace_index(current_db: Path, zona_sentence_file: Path, out_db: Path) -> None:
    """Cria índice REPLACE: chunks atuais, trocando os do Anexo II da NR-10 pela frase DocAI."""
    con = sqlite3.connect(str(current_db)); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT doc,section,title,page,text FROM chunks ORDER BY id")]
    con.close()

    sentences = json.loads(zona_sentence_file.read_text(encoding="utf-8"))
    zona = [s["sentence"] for s in sentences if s.get("kind") == "zona_risco"]
    zona_text = ("NR-10 · Anexo II Zona de Risco e Zona Controlada (tabela de distâncias, "
                 "reconstruída via Document AI). " + " ".join(zona))

    new_rows = []
    replaced = False
    for r in rows:
        if r["doc"] == "nr-10" and r["section"].lower().startswith("anexo ii"):
            if not replaced:
                new_rows.append({"doc": "nr-10", "section": "Anexo Ii",
                                 "title": "Anexo II Zona de Risco e Zona Controlada (DocAI)",
                                 "page": 1, "text": zona_text})
                replaced = True
            # descarta os chunks corrompidos do Anexo II (274-276)
            continue
        new_rows.append(r)

    if out_db.exists():
        out_db.unlink()
    con = sqlite3.connect(str(out_db)); cur = con.cursor()
    cur.execute("""CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc TEXT,section TEXT,title TEXT,page INTEGER,text TEXT);""")
    cur.execute("""CREATE VIRTUAL TABLE chunks_fts USING fts5(doc,section,title,text,
        content='chunks',content_rowid='id',tokenize='unicode61 remove_diacritics 2');""")
    cur.execute("""CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid,doc,section,title,text)
        VALUES(new.id,new.doc,new.section,new.title,new.text); END;""")
    cur.executemany("INSERT INTO chunks(doc,section,title,page,text) VALUES(?,?,?,?,?)",
                    [(r["doc"], r["section"], r["title"], int(r["page"]), r["text"]) for r in new_rows])
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    con.commit(); con.close()


def search(db: Path, query_fts: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if query_fts == '""':
        return []
    con = sqlite3.connect(str(db)); con.row_factory = sqlite3.Row
    w = WEIGHTS
    sql = f"""SELECT c.id,c.doc,c.section,c.title,c.text,
        bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]}) s
        FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
        WHERE chunks_fts MATCH ? ORDER BY s ASC LIMIT ?;"""
    try:
        rows = [dict(r) for r in con.execute(sql, (query_fts, top_k))]
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return rows


def eval_full(db: Path, pairs: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    h1 = h2 = h5 = 0; mrr = 0.0
    for p in pairs:
        res = search(db, tokenizer(p["question"]), 5)
        rank = next((i + 1 for i, r in enumerate(res) if check_hit(r, p)), None)
        if rank:
            if rank <= 1: h1 += 1
            if rank <= 2: h2 += 1
            if rank <= 5: h5 += 1
            mrr += 1.0 / rank
    n = len(pairs)
    return {"n": n, "R@1": round(100*h1/n, 1), "R@2": round(100*h2/n, 1),
            "R@5": round(100*h5/n, 1), "MRR": round(mrr/n, 4)}


def captain_distance(db: Path) -> Dict[str, Any]:
    q = "qual a distancia segura para media tensao de 13,8 kV?"
    out = {}
    for tk_name, tk in [("app", app_fts_query), ("numeric_preserving", numeric_preserving_query)]:
        res = search(db, tk(q), 3)
        out[tk_name] = [{"doc": r["doc"], "section": r["section"],
                         "has_answer": ("13,8" in r["text"] and "0,38" in r["text"]),
                         "text": r["text"][:160]} for r in res]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-db", default="corpus/index_hf_36nr.db")
    ap.add_argument("--zona-file", default="docai/data_all/sentences_nr-10.json")
    ap.add_argument("--replace-db", default="docai/data/index_docai_replace.db")
    ap.add_argument("--qa", default="corpus/qa_pairs_v2.jsonl")
    ap.add_argument("--out", default="docai/data/replace_ceiling_report.json")
    args = ap.parse_args()

    build_replace_index(Path(args.current_db), Path(args.zona_file), Path(args.replace_db))
    pairs = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
    nr10 = [p for p in pairs if p["doc"] == "nr-10"]

    report = {
        "all_151": {
            "current_apptok": eval_full(Path(args.current_db), pairs, app_fts_query),
            "replace_apptok": eval_full(Path(args.replace_db), pairs, app_fts_query),
            "current_numtok": eval_full(Path(args.current_db), pairs, numeric_preserving_query),
            "replace_numtok": eval_full(Path(args.replace_db), pairs, numeric_preserving_query),
        },
        "nr10_only": {
            "current_apptok": eval_full(Path(args.current_db), nr10, app_fts_query),
            "replace_apptok": eval_full(Path(args.replace_db), nr10, app_fts_query),
            "current_numtok": eval_full(Path(args.current_db), nr10, numeric_preserving_query),
            "replace_numtok": eval_full(Path(args.replace_db), nr10, numeric_preserving_query),
        },
        "captain_distance_case": {
            "current": captain_distance(Path(args.current_db)),
            "replace": captain_distance(Path(args.replace_db)),
        },
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    a = report["all_151"]
    print("=== 151 GERAL (app-tok):   R@2 %.1f -> %.1f | R@5 %.1f -> %.1f ===" % (
        a["current_apptok"]["R@2"], a["replace_apptok"]["R@2"],
        a["current_apptok"]["R@5"], a["replace_apptok"]["R@5"]))
    print("=== 151 GERAL (num-tok):   R@2 %.1f -> %.1f | R@5 %.1f -> %.1f ===" % (
        a["current_numtok"]["R@2"], a["replace_numtok"]["R@2"],
        a["current_numtok"]["R@5"], a["replace_numtok"]["R@5"]))
    print("Relatório em", args.out)


if __name__ == "__main__":
    main()
