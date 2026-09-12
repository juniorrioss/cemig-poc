"""
04_reindex_compare.py — ETAPA 6: reindexação aditiva + comparação obrigatória nas MESMAS 151.

Constrói um índice novo (index_docai.db) = TODOS os chunks do índice canônico atual
(corpus/index_hf_36nr.db) + chunks NOVOS com as frases de tabela reconstruídas pelo
Document AI para as 5 NRs-piloto. Aditivo: as 13 NRs puramente textuais não são tocadas
(isola a variável, evita regressão).

Compara atual vs novo nas MESMAS 151 do holdout, com o CAMINHO DO APP (app_fts_query:
stem-6 + prefixo* + OR; pesos 1.5/3/2/1) e casamento POR CONTEÚDO (check_hit por doc+seção),
não por id — o holdout está amarrado a conteúdo, não a ids do índice.

Também roda os DOIS casos do capitão como consultas ad-hoc.

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from corpus.eval_fino import app_fts_query  # replica fiel do app
from corpus.eval_retrieval import check_hit   # casamento por doc+seção/conteúdo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS = (1.5, 3.0, 2.0, 1.0)


def load_current_chunks(db_path: Path) -> List[Dict[str, Any]]:
    """Lê todos os chunks do índice canônico atual (preserva ordem por id)."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT doc, section, title, page, text FROM chunks ORDER BY id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def build_docai_chunks(parsed_dir: Path, nrs: List[str]) -> List[Dict[str, Any]]:
    """
    Gera chunks NOVOS a partir das frases achatadas do DocAI.
    Agrupa frases por (nr, section-derivada) em blocos de ~280 palavras para não diluir o BM25.
    As frases zona_risco da NR-10 ficam num bloco próprio (section 'Anexo Ii' — casa o gold).
    """
    new_chunks: List[Dict[str, Any]] = []
    for nr in nrs:
        sf = parsed_dir / f"sentences_{nr}.json"
        if not sf.exists():
            continue
        sentences = json.loads(sf.read_text(encoding="utf-8"))
        # Separa zona_risco (nr-10) do resto.
        zona = [s["sentence"] for s in sentences if s.get("kind") == "zona_risco"]
        generic = [s["sentence"] for s in sentences if s.get("kind") != "zona_risco"]

        if zona:
            # Um único chunk com todas as frases de zona (curto e coeso).
            text = f"NR-{nr.split('-')[1]} · Anexo II Zona de Risco e Zona Controlada (tabela de distâncias). " + " ".join(zona)
            new_chunks.append({
                "doc": nr, "section": "Anexo Ii",
                "title": "Anexo II Zona de Risco e Zona Controlada (DocAI)",
                "page": 1, "text": text,
            })

        # Genéricas: agrupa em blocos de ~280 palavras.
        buf: List[str] = []
        wc = 0
        block = 0
        for sent in generic:
            buf.append(sent)
            wc += len(sent.split())
            if wc >= 280:
                new_chunks.append({
                    "doc": nr, "section": f"Tabela DocAI {block}",
                    "title": f"Tabelas reconstruídas (DocAI) — bloco {block}",
                    "page": 1, "text": f"NR-{nr.split('-')[1]} · Tabelas — " + " ".join(buf),
                })
                buf = []; wc = 0; block += 1
        if buf:
            new_chunks.append({
                "doc": nr, "section": f"Tabela DocAI {block}",
                "title": f"Tabelas reconstruídas (DocAI) — bloco {block}",
                "page": 1, "text": f"NR-{nr.split('-')[1]} · Tabelas — " + " ".join(buf),
            })
    return new_chunks


def build_index(chunks: List[Dict[str, Any]], db_path: Path) -> None:
    """Constrói um índice FTS5 idêntico em schema ao build_index.py do projeto."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("""CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc TEXT NOT NULL, section TEXT NOT NULL, title TEXT NOT NULL,
        page INTEGER NOT NULL, text TEXT NOT NULL);""")
    cur.execute("""CREATE VIRTUAL TABLE chunks_fts USING fts5(doc, section, title, text,
        content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2');""")
    for trig, body in [
        ("ai", "INSERT INTO chunks_fts(rowid,doc,section,title,text) VALUES(new.id,new.doc,new.section,new.title,new.text);"),
    ]:
        cur.execute(f"CREATE TRIGGER chunks_{trig} AFTER INSERT ON chunks BEGIN {body} END;")
    cur.executemany("INSERT INTO chunks(doc,section,title,page,text) VALUES(?,?,?,?,?)",
                    [(c["doc"], c["section"], c["title"], int(c["page"]), c["text"]) for c in chunks])
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    con.commit(); con.close()


def search(db_path: Path, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Busca BM25 pelo caminho do app (app_fts_query + pesos 1.5/3/2/1)."""
    fts_q = app_fts_query(query)
    if fts_q == '""':
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    w = WEIGHTS
    sql = f"""SELECT c.id,c.doc,c.section,c.title,c.page,c.text,
        bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]}) AS score
        FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
        WHERE chunks_fts MATCH ? ORDER BY score ASC LIMIT ?;"""
    try:
        rows = con.execute(sql, (fts_q, top_k)).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning("FTS erro '%s': %s", query, e)
        rows = []
    con.close()
    return [dict(r) for r in rows]


def eval_index(db_path: Path, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """R@1/R@2/R@5 e MRR pelo caminho do app, casando por conteúdo (check_hit)."""
    h1 = h2 = h5 = 0
    mrr = 0.0
    per_doc: Dict[str, List[int]] = {}
    misses: List[str] = []
    for p in pairs:
        res = search(db_path, p["question"], top_k=5)
        rank = None
        for i, r in enumerate(res[:5]):
            if check_hit(r, p):
                rank = i + 1
                break
        d = p["doc"]
        per_doc.setdefault(d, [0, 0])
        per_doc[d][1] += 1
        if rank:
            if rank <= 1: h1 += 1
            if rank <= 2: h2 += 1; per_doc[d][0] += 1
            if rank <= 5: h5 += 1
            mrr += 1.0 / rank
        else:
            misses.append(p["id"])
    n = len(pairs)
    return {
        "n": n,
        "R@1": round(100 * h1 / n, 1),
        "R@2": round(100 * h2 / n, 1),
        "R@5": round(100 * h5 / n, 1),
        "MRR": round(mrr / n, 4),
        "misses": misses,
        "per_doc_r2": {k: round(100 * v[0] / v[1], 1) for k, v in sorted(per_doc.items())},
    }


def captain_cases(db_path: Path) -> Dict[str, Any]:
    """Roda os dois casos do capitão como consultas ad-hoc; devolve o top-3."""
    cases = {
        "distancia_13.8kv": "qual a distancia segura para media tensao de 13,8 kV?",
        "poste_bambo": "o poste que preciso subir nao me parece firme",
    }
    out = {}
    for key, q in cases.items():
        res = search(db_path, q, top_k=3)
        out[key] = {
            "query": q,
            "top3": [{"doc": r["doc"], "section": r["section"],
                      "text": r["text"][:240]} for r in res],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reindex aditivo DocAI + comparação nas 151.")
    ap.add_argument("--current-db", default="corpus/index_hf_36nr.db")
    ap.add_argument("--parsed-dir", default="docai/data")
    ap.add_argument("--new-db", default="docai/data/index_docai.db")
    ap.add_argument("--qa", default="corpus/qa_pairs_v2.jsonl")
    ap.add_argument("--nrs", default="nr-10,nr-15,nr-28,nr-04,nr-32")
    ap.add_argument("--out", default="docai/data/compare_report.json")
    args = ap.parse_args()

    nrs = [n.strip() for n in args.nrs.split(",")]
    current = load_current_chunks(Path(args.current_db))
    docai_chunks = build_docai_chunks(Path(args.parsed_dir), nrs)
    logger.info("chunks atuais=%d, chunks DocAI novos=%d", len(current), len(docai_chunks))

    new_db = Path(args.new_db)
    build_index(current + docai_chunks, new_db)
    logger.info("Índice novo construído: %s (%d chunks)", new_db, len(current) + len(docai_chunks))

    pairs = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]

    m_cur = eval_index(Path(args.current_db), pairs)
    m_new = eval_index(new_db, pairs)

    # Sub-análise só nas 45 de NR-10 (onde há material-piloto no holdout).
    nr10 = [p for p in pairs if p["doc"] == "nr-10"]
    m_cur_10 = eval_index(Path(args.current_db), nr10)
    m_new_10 = eval_index(new_db, nr10)

    cases_cur = captain_cases(Path(args.current_db))
    cases_new = captain_cases(new_db)

    report = {
        "all_151": {"current": m_cur, "docai": m_new,
                    "delta_R@2": round(m_new["R@2"] - m_cur["R@2"], 1),
                    "delta_R@5": round(m_new["R@5"] - m_cur["R@5"], 1)},
        "nr10_only": {"current": m_cur_10, "docai": m_new_10,
                      "delta_R@2": round(m_new_10["R@2"] - m_cur_10["R@2"], 1),
                      "delta_R@5": round(m_new_10["R@5"] - m_cur_10["R@5"], 1)},
        "captain_cases": {"current": cases_cur, "docai": cases_new},
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=== 151 GERAL: R@2 %.1f -> %.1f (%+.1f), R@5 %.1f -> %.1f (%+.1f) ===",
                m_cur["R@2"], m_new["R@2"], report["all_151"]["delta_R@2"],
                m_cur["R@5"], m_new["R@5"], report["all_151"]["delta_R@5"])
    logger.info("=== NR-10 (45): R@2 %.1f -> %.1f (%+.1f), R@5 %.1f -> %.1f (%+.1f) ===",
                m_cur_10["R@2"], m_new_10["R@2"], report["nr10_only"]["delta_R@2"],
                m_cur_10["R@5"], m_new_10["R@5"], report["nr10_only"]["delta_R@5"])
    logger.info("Relatório salvo em %s", args.out)


if __name__ == "__main__":
    main()
