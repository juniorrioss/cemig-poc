"""
eval_v2.py — Avaliação comparativa completa do Corpus v2 (SQLite FTS5 / BM25).

Executa as 4 análises comparativas requeridas pelo Firstmate / POC CEMIG:
  (i)   index v1 (PDF) vs (a) index_hf_5nr.db (Markdown Limpo) nas 101 perguntas do v1;
  (ii)  (a) index_hf_5nr.db vs (b) index_hf_5nr_manual.db — avaliação de poluição pelo manual
        e medição exata do deslocamento do chunk-ouro da norma do top-3;
  (iii) (a) index_hf_5nr.db vs (c) index_hf_36nr.db nas 101 perguntas — medição de diluição
        do recall das 5 NRs críticas ao expandir para 36 normas;
  (iv)  (c) index_hf_36nr.db no conjunto consolidado v2 (101 v1 + novas perguntas ouro).

Métricas: Recall@1, Recall@3, Recall@5 e MRR (Mean Reciprocal Rank).
"""

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from corpus.build_index import format_fts_query, search_reference
from corpus.eval_retrieval import EvalMetrics, check_hit


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_V1 = Path("corpus/index.db")
DEFAULT_INDEX_HF_5NR = Path("corpus/index_hf_5nr.db")
DEFAULT_INDEX_HF_5NR_MANUAL = Path("corpus/index_hf_5nr_manual.db")
DEFAULT_INDEX_HF_36NR = Path("corpus/index_hf_36nr.db")

DEFAULT_QA_V1 = Path("corpus/qa_pairs.jsonl")
DEFAULT_QA_V2 = Path("corpus/qa_pairs_v2.jsonl")


def run_strategy_on_db(
    db_path: Path,
    qa_pairs: List[Dict[str, Any]],
    strategy: str,
) -> Tuple[EvalMetrics, Dict[str, EvalMetrics], List[Optional[int]]]:
    """Executa uma estratégia específica de retrieval em um banco SQLite e retorna métricas."""
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    total_m = EvalMetrics(total_queries=len(qa_pairs), hits_at_1=0, hits_at_3=0, hits_at_5=0, mrr_sum=0.0)
    by_doc_m: Dict[str, EvalMetrics] = defaultdict(
        lambda: EvalMetrics(total_queries=0, hits_at_1=0, hits_at_3=0, hits_at_5=0, mrr_sum=0.0)
    )
    query_ranks: List[Optional[int]] = []

    for p in qa_pairs:
        doc = p.get("doc", "outros")
        dm = by_doc_m[doc]
        dm.total_queries += 1

        if strategy == "raw":
            fts_q = format_fts_query(p["question"], mode="OR", filter_stopwords=True)
            cur.execute("""
                SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT 5;
            """, (fts_q,))
        elif strategy == "terms_or":
            fts_q = format_fts_query(p["query_terms"], mode="OR", filter_stopwords=False)
            cur.execute("""
                SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT 5;
            """, (fts_q,))
        elif strategy == "boosted":
            fts_q = format_fts_query(f"{p['doc']} {p['query_terms']}", mode="OR", filter_stopwords=False)
            cur.execute("""
                SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                       bm25(chunks_fts, 5.0, 5.0, 2.0, 1.0) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT 5;
            """, (fts_q,))
        elif strategy == "filtered":
            fts_q = format_fts_query(p["query_terms"], mode="OR", filter_stopwords=False)
            cur.execute("""
                SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                       bm25(chunks_fts, 1.0, 3.0, 2.0, 1.0) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE c.doc = ? AND chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT 5;
            """, (p["doc"], fts_q))
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")

        res = [dict(r) for r in cur.fetchall()]
        found_rank: Optional[int] = None

        for rank_idx, r in enumerate(res[:5]):
            if check_hit(r, p):
                found_rank = rank_idx + 1
                break

        query_ranks.append(found_rank)

        if found_rank is not None:
            if found_rank <= 1:
                total_m.hits_at_1 += 1
                dm.hits_at_1 += 1
            if found_rank <= 3:
                total_m.hits_at_3 += 1
                dm.hits_at_3 += 1
            if found_rank <= 5:
                total_m.hits_at_5 += 1
                dm.hits_at_5 += 1
            total_m.mrr_sum += 1.0 / found_rank
            dm.mrr_sum += 1.0 / found_rank

    con.close()
    return total_m, dict(by_doc_m), query_ranks


def measure_manual_displacement(
    db_norma: Path,
    db_manual: Path,
    qa_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mede quantas vezes um chunk do manual comentado desloca o chunk da norma do top-3."""
    con_a = sqlite3.connect(str(db_norma))
    con_a.row_factory = sqlite3.Row
    cur_a = con_a.cursor()

    con_b = sqlite3.connect(str(db_manual))
    con_b.row_factory = sqlite3.Row
    cur_b = con_b.cursor()

    displacements = 0
    displaced_list: List[Dict[str, Any]] = []

    for p in qa_pairs:
        fts_q = format_fts_query(p["query_terms"], mode="OR", filter_stopwords=False)

        # Busca no banco apenas com a norma (a)
        cur_a.execute("""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT 5;
        """, (fts_q,))
        res_a = [dict(r) for r in cur_a.fetchall()]

        # Busca no banco com norma + manual (b)
        cur_b.execute("""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT 5;
        """, (fts_q,))
        res_b = [dict(r) for r in cur_b.fetchall()]

        # Verifica se na norma pura estava no top 3
        norm_in_top3_a = any(check_hit(r, p) and r["doc"] == p["doc"] for r in res_a[:3])
        # Verifica se no banco misto a norma continua no top 3
        norm_in_top3_b = any(check_hit(r, p) and r["doc"] == p["doc"] for r in res_b[:3])
        # Verifica se algum manual entrou no top 3
        manual_in_top3_b = any(r["doc"].endswith("-manual") for r in res_b[:3])

        if norm_in_top3_a and not norm_in_top3_b and manual_in_top3_b:
            displacements += 1
            displaced_list.append({
                "id": p["id"],
                "doc": p["doc"],
                "section": p.get("section"),
                "question": p["question"],
                "top3_b_docs": [r["doc"] for r in res_b[:3]],
            })

    con_a.close()
    con_b.close()

    total_q = len(qa_pairs)
    displacement_rate = (displacements / total_q) * 100 if total_q else 0.0
    return {
        "total_queries": total_q,
        "displacements": displacements,
        "displacement_rate": displacement_rate,
        "sample_displaced": displaced_list[:10],
    }


def print_table_header(title: str, query_count: int) -> None:
    print("\n" + "=" * 88)
    print(f"  {title.upper()}")
    print(f"  Total de Casos de Teste Avaliados: {query_count}")
    print("=" * 88)
    print(f"| {'Índice Avaliado':<36} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'MRR':>8} |")
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")


def print_row(label: str, m: EvalMetrics) -> None:
    print(f"| {label:<36} | {m.recall_at_1:>8.1f}% | {m.recall_at_3:>8.1f}% | {m.recall_at_5:>8.1f}% | {m.mrr:>8.4f} |")


def run_all_evaluations(
    v1_db: Path = DEFAULT_INDEX_V1,
    hf_5nr_db: Path = DEFAULT_INDEX_HF_5NR,
    hf_5nr_manual_db: Path = DEFAULT_INDEX_HF_5NR_MANUAL,
    hf_36nr_db: Path = DEFAULT_INDEX_HF_36NR,
    qa_v1_path: Path = DEFAULT_QA_V1,
    qa_v2_path: Path = DEFAULT_QA_V2,
) -> Dict[str, Any]:
    """Executa a bateria de avaliações completa dos 4 cenários do brief."""
    with open(qa_v1_path, "r", encoding="utf-8") as f:
        qa_v1 = [json.loads(line) for line in f if line.strip()]

    with open(qa_v2_path, "r", encoding="utf-8") as f:
        qa_v2 = [json.loads(line) for line in f if line.strip()]

    eval_results: Dict[str, Any] = {}

    # =========================================================================
    # (i) index v1 (PDF) vs (a) index_hf_5nr (Markdown Limpo) — 101 perguntas v1
    # =========================================================================
    print_table_header("Comparação (i): Fonte Limpa Markdown vs Extração PDF (101 Perguntas v1)", len(qa_v1))

    # Avalia na estratégia principal recomendada: Tool-Calling Filtrado por Norma
    m_v1_filt, _, _ = run_strategy_on_db(v1_db, qa_v1, "filtered")
    m_a_filt, _, _ = run_strategy_on_db(hf_5nr_db, qa_v1, "filtered")
    print_row("PDF v1 (Filtrado por Norma)", m_v1_filt)
    print_row("HF 5NR (a) (Filtrado por Norma)", m_a_filt)
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")

    # Avalia na estratégia Tool-Calling Termos Soltos (busca aberta)
    m_v1_terms, _, _ = run_strategy_on_db(v1_db, qa_v1, "terms_or")
    m_a_terms, _, _ = run_strategy_on_db(hf_5nr_db, qa_v1, "terms_or")
    print_row("PDF v1 (Termos Soltos OR)", m_v1_terms)
    print_row("HF 5NR (a) (Termos Soltos OR)", m_a_terms)
    print("=" * 88)

    eval_results["comp_i"] = {
        "pdf_v1_filtered": m_v1_filt,
        "hf_5nr_filtered": m_a_filt,
        "pdf_v1_terms": m_v1_terms,
        "hf_5nr_terms": m_a_terms,
    }

    # =========================================================================
    # (ii) (a) HF 5NR vs (b) HF 5NR + Manual — O manual ajuda ou polui?
    # =========================================================================
    print_table_header("Comparação (ii): Efeito do Manual Comentado (a vs b) (101 Perguntas v1)", len(qa_v1))

    m_b_filt, _, _ = run_strategy_on_db(hf_5nr_manual_db, qa_v1, "filtered")
    m_b_terms, _, _ = run_strategy_on_db(hf_5nr_manual_db, qa_v1, "terms_or")
    m_a_boost, _, _ = run_strategy_on_db(hf_5nr_db, qa_v1, "boosted")
    m_b_boost, _, _ = run_strategy_on_db(hf_5nr_manual_db, qa_v1, "boosted")

    print_row("HF 5NR (a) Norma Pura (Termos Soltos)", m_a_terms)
    print_row("HF 5NR+Manual (b) (Termos Soltos)", m_b_terms)
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")
    print_row("HF 5NR (a) Norma Pura (Boost Doc)", m_a_boost)
    print_row("HF 5NR+Manual (b) (Boost Doc)", m_b_boost)
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")
    print_row("HF 5NR (a) Norma Pura (Filtrado Doc)", m_a_filt)
    print_row("HF 5NR+Manual (b) (Filtrado Doc)", m_b_filt)
    print("=" * 88)

    # Medição exata do deslocamento do chunk-ouro da norma do top-3
    disp_stats = measure_manual_displacement(hf_5nr_db, hf_5nr_manual_db, qa_v1)
    print(f"\n[Análise de Poluição] Deslocamentos do Top-3 Causados pelo Manual Comentado:")
    print(f"  Total de Perguntas: {disp_stats['total_queries']}")
    print(f"  Perguntas com Chunk da Norma expulso do Top-3 por Chunks do Manual: {disp_stats['displacements']} ({disp_stats['displacement_rate']:.1f}%)")
    print("  Exemplos de consultas afetadas:")
    for ex in disp_stats["sample_displaced"][:5]:
        print(f"    - [{ex['id']} / {ex['doc']}]: \"{ex['question'][:68]}...\"")
        print(f"      Top 3 retornado em (b): {ex['top3_b_docs']}")

    eval_results["comp_ii"] = {
        "hf_5nr_terms": m_a_terms,
        "hf_5nr_manual_terms": m_b_terms,
        "displacement": disp_stats,
    }

    # =========================================================================
    # (iii) (a) HF 5NR vs (c) HF 36NR nas 101 perguntas — Diluição por Expansão
    # =========================================================================
    print_table_header("Comparação (iii): Diluição por Expansão 5 NR -> 36 NR (101 Perguntas v1)", len(qa_v1))

    m_c_filt_101, _, _ = run_strategy_on_db(hf_36nr_db, qa_v1, "filtered")
    m_c_terms_101, _, _ = run_strategy_on_db(hf_36nr_db, qa_v1, "terms_or")
    m_c_boost_101, _, _ = run_strategy_on_db(hf_36nr_db, qa_v1, "boosted")

    print_row("HF 5NR (a) (Filtrado por Norma)", m_a_filt)
    print_row("HF 36NR (c) (Filtrado por Norma)", m_c_filt_101)
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")
    print_row("HF 5NR (a) (Boost de Doc)", m_a_boost)
    print_row("HF 36NR (c) (Boost de Doc)", m_c_boost_101)
    print("|" + "-" * 38 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")
    print_row("HF 5NR (a) (Termos Soltos OR)", m_a_terms)
    print_row("HF 36NR (c) (Termos Soltos OR)", m_c_terms_101)
    print("=" * 88)

    eval_results["comp_iii"] = {
        "hf_5nr_filtered": m_a_filt,
        "hf_36nr_filtered": m_c_filt_101,
        "hf_5nr_terms": m_a_terms,
        "hf_36nr_terms": m_c_terms_101,
    }

    # =========================================================================
    # (iv) (c) HF 36NR no Conjunto Completo v2 (151 perguntas)
    # =========================================================================
    print_table_header("Avaliação (iv): Desempenho do HF 36NR no Dataset Completo v2 (151 Perguntas)", len(qa_v2))

    m_c_raw_v2, by_doc_raw, _ = run_strategy_on_db(hf_36nr_db, qa_v2, "raw")
    m_c_terms_v2, by_doc_terms, _ = run_strategy_on_db(hf_36nr_db, qa_v2, "terms_or")
    m_c_boost_v2, by_doc_boost, _ = run_strategy_on_db(hf_36nr_db, qa_v2, "boosted")
    m_c_filt_v2, by_doc_filt, _ = run_strategy_on_db(hf_36nr_db, qa_v2, "filtered")

    print_row("1. Pergunta Bruta (Voz / Stopwords)", m_c_raw_v2)
    print_row("2. Tool-Calling SLM (Termos Soltos OR)", m_c_terms_v2)
    print_row("3. Tool-Calling SLM (Termos + Boost)", m_c_boost_v2)
    print_row("4. Tool-Calling SLM (Filtrado por Norma)", m_c_filt_v2)
    print("=" * 88)

    print("\n--- Detalhamento por Norma Regulamentadora no Índice 36 NR (Estratégia Filtrada) ---")
    print(f"| {'Norma':<8} | {'Qtd Pares':>10} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'MRR':>8} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")

    for doc_key in sorted(by_doc_filt.keys()):
        dm = by_doc_filt[doc_key]
        print(f"| {doc_key.upper():<8} | {dm.total_queries:>10} | {dm.recall_at_1:>8.1f}% | {dm.recall_at_3:>8.1f}% | {dm.recall_at_5:>8.1f}% | {dm.mrr:>8.4f} |")
    print("-" * 68 + "\n")

    eval_results["comp_iv"] = {
        "overall_filtered": m_c_filt_v2,
        "overall_terms": m_c_terms_v2,
        "by_doc": by_doc_filt,
    }

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação comparativa de retrieval BM25 para o corpus v2.")
    parser.add_argument("--v1-db", type=str, default=str(DEFAULT_INDEX_V1), help="Banco v1 (PDF).")
    parser.add_argument("--hf-5nr-db", type=str, default=str(DEFAULT_INDEX_HF_5NR), help="Banco HF 5 NRs.")
    parser.add_argument("--hf-5nr-manual-db", type=str, default=str(DEFAULT_INDEX_HF_5NR_MANUAL), help="Banco HF 5 NRs + manuais.")
    parser.add_argument("--hf-36nr-db", type=str, default=str(DEFAULT_INDEX_HF_36NR), help="Banco HF 36 NRs.")
    parser.add_argument("--qa-v1", type=str, default=str(DEFAULT_QA_V1), help="Dataset de teste v1 (101 perguntas).")
    parser.add_argument("--qa-v2", type=str, default=str(DEFAULT_QA_V2), help="Dataset consolidado v2 (151 perguntas).")
    args = parser.parse_args()

    results = run_all_evaluations(
        v1_db=Path(args.v1_db),
        hf_5nr_db=Path(args.hf_5nr_db),
        hf_5nr_manual_db=Path(args.hf_5nr_manual_db),
        hf_36nr_db=Path(args.hf_36nr_db),
        qa_v1_path=Path(args.qa_v1),
        qa_v2_path=Path(args.qa_v2),
    )


if __name__ == "__main__":
    main()
