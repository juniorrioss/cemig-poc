"""
eval_retrieval.py — Avaliação de retrieval BM25 (Recall@1, 3, 5 e MRR) no SQLite FTS5.

Compara o desempenho de recuperação contra o dataset sintético de perguntas de operários (JSONL):
1. Pergunta Bruta (Filtro de stopwords + OR) — busca direta da fala sem reescrita
2. Tool-Calling SLM (Termos Soltos OR) — busca com palavras-chave extraídas pelo modelo
3. Tool-Calling SLM (Termos Soltos + Boost de Norma) — termos + ponderação do documento
4. Tool-Calling SLM Filtrado por Norma (doc = 'nr-XX') — tool call com escopo de norma

Gera tabela comparativa, breakdown por NR e análise detalhada dos resultados.
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path("corpus/index.db")
DEFAULT_QA_PATH = Path("corpus/qa_pairs.jsonl")


@dataclass
class EvalMetrics:
    """Métricas consolidadas de recuperação de informação."""
    total_queries: int
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int
    mrr_sum: float

    @property
    def recall_at_1(self) -> float:
        return (self.hits_at_1 / self.total_queries) * 100 if self.total_queries else 0.0

    @property
    def recall_at_3(self) -> float:
        return (self.hits_at_3 / self.total_queries) * 100 if self.total_queries else 0.0

    @property
    def recall_at_5(self) -> float:
        return (self.hits_at_5 / self.total_queries) * 100 if self.total_queries else 0.0

    @property
    def mrr(self) -> float:
        return (self.mrr_sum / self.total_queries) if self.total_queries else 0.0


def check_hit(retrieved_chunk: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """Verifica se o chunk recuperado corresponde ao gabarito ouro."""
    # 1. Correspondência exata por chunk_id primário ou lista de chunks relevantes
    if retrieved_chunk["id"] == pair.get("chunk_id"):
        return True
    if pair.get("relevant_chunk_ids") and retrieved_chunk["id"] in pair["relevant_chunk_ids"]:
        return True

    # 2. Correspondência semântica e estrutural por documento e seção/item
    target_doc = pair.get("doc", "").lower()
    target_sec = pair.get("section", "").lower()
    ret_doc = retrieved_chunk["doc"].lower()
    ret_sec = retrieved_chunk["section"].lower()
    ret_text = retrieved_chunk["text"].lower()

    if target_doc == ret_doc:
        # Seção do gabarito contida na seção ou texto do chunk
        if target_sec in ret_sec or ret_sec in target_sec:
            return True
        if target_sec and target_sec in ret_text:
            return True

    return False


def run_evaluation_strategy(
    db_path: Path,
    qa_pairs: List[Dict[str, Any]],
    query_func: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
) -> Tuple[EvalMetrics, Dict[str, EvalMetrics]]:
    """Executa a avaliação para uma estratégia de consulta, calculando totais e por NR."""
    total_metrics = EvalMetrics(
        total_queries=len(qa_pairs),
        hits_at_1=0,
        hits_at_3=0,
        hits_at_5=0,
        mrr_sum=0.0,
    )
    by_doc_metrics: Dict[str, EvalMetrics] = defaultdict(
        lambda: EvalMetrics(total_queries=0, hits_at_1=0, hits_at_3=0, hits_at_5=0, mrr_sum=0.0)
    )

    for p in qa_pairs:
        doc = p.get("doc", "outros")
        doc_m = by_doc_metrics[doc]
        doc_m.total_queries += 1

        results = query_func(p)
        found_rank: Optional[int] = None

        for rank_idx, r in enumerate(results[:5]):
            if check_hit(r, p):
                found_rank = rank_idx + 1
                break

        if found_rank is not None:
            if found_rank <= 1:
                total_metrics.hits_at_1 += 1
                doc_m.hits_at_1 += 1
            if found_rank <= 3:
                total_metrics.hits_at_3 += 1
                doc_m.hits_at_3 += 1
            if found_rank <= 5:
                total_metrics.hits_at_5 += 1
                doc_m.hits_at_5 += 1

            total_metrics.mrr_sum += 1.0 / found_rank
            doc_m.mrr_sum += 1.0 / found_rank

    return total_metrics, dict(by_doc_metrics)


def evaluate_all(db_path: Path = DEFAULT_DB_PATH, qa_path: Path = DEFAULT_QA_PATH) -> Dict[str, Any]:
    """Executa a bateria completa de experimentos de retrieval BM25."""
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados index.db não encontrado em {db_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"Arquivo de perguntas e respostas ouro não encontrado em {qa_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # --- Definição das Estratégias ---

    # 1. Pergunta Bruta de Voz (Stopwords removidas, busca OR livre)
    def query_raw(p: Dict[str, Any]) -> List[Dict[str, Any]]:
        return search_reference(db_path, p["question"], top_k=5, mode="OR", filter_stopwords=True)

    # 2. Tool-Calling SLM: Termos Soltos (OR)
    def query_tool_or(p: Dict[str, Any]) -> List[Dict[str, Any]]:
        return search_reference(db_path, p["query_terms"], top_k=5, mode="OR", filter_stopwords=False)

    # 3. Tool-Calling SLM: Termos Soltos + Boost de Documento
    def query_tool_boosted(p: Dict[str, Any]) -> List[Dict[str, Any]]:
        fts_q = format_fts_query(f"{p['doc']} {p['query_terms']}", mode="OR", filter_stopwords=False)
        cur = con.cursor()
        cur.execute("""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts, 5.0, 5.0, 2.0, 1.0) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT 5;
        """, (fts_q,))
        return [dict(r) for r in cur.fetchall()]

    # 4. Tool-Calling SLM: Filtrado por Norma (quando o tool call especifica a norma)
    def query_tool_doc_filtered(p: Dict[str, Any]) -> List[Dict[str, Any]]:
        fts_q = format_fts_query(p["query_terms"], mode="OR", filter_stopwords=False)
        cur = con.cursor()
        cur.execute("""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts, 1.0, 3.0, 2.0, 1.0) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE c.doc = ? AND chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT 5;
        """, (p["doc"], fts_q))
        return [dict(r) for r in cur.fetchall()]

    strategies = {
        "1. Pergunta Bruta (Voz / Stopwords)": query_raw,
        "2. Tool-Calling SLM (Termos Soltos OR)": query_tool_or,
        "3. Tool-Calling SLM (Termos + Boost Doc)": query_tool_boosted,
        "4. Tool-Calling SLM (Filtrado por Norma)": query_tool_doc_filtered,
    }

    results_table: List[Dict[str, Any]] = []
    doc_breakdown: Dict[str, Dict[str, EvalMetrics]] = {}

    for strat_name, q_func in strategies.items():
        total_m, by_doc_m = run_evaluation_strategy(db_path, qa_pairs, q_func)
        results_table.append({
            "strategy": strat_name,
            "metrics": total_m,
        })
        doc_breakdown[strat_name] = by_doc_m

    con.close()

    # Formatação e Impressão no Terminal
    print("\n" + "=" * 84)
    print("      AVALIAÇÃO DE RETRIEVAL BM25 (SQLITE FTS5) — CEMIG POC ASSISTENTE OFFLINE")
    print(f"      Total de Pares P&R Avaliados: {len(qa_pairs)} em 5 NRs críticas")
    print("=" * 84)
    print(f"| {'Estratégia de Retrieval':<40} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'MRR':>8} |")
    print("|" + "-" * 42 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")

    for entry in results_table:
        m: EvalMetrics = entry["metrics"]
        print(f"| {entry['strategy']:<40} | {m.recall_at_1:>8.1f}% | {m.recall_at_3:>8.1f}% | {m.recall_at_5:>8.1f}% | {m.mrr:>8.4f} |")

    print("=" * 84)

    # Detalhamento por NR para a estratégia com filtro de norma (principal)
    main_strat = "4. Tool-Calling SLM (Filtrado por Norma)"
    print(f"\n--- Detalhamento por Norma Regulamentadora ({main_strat}) ---")
    print(f"| {'Norma':<8} | {'Qtd Pares':>10} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'MRR':>8} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|")

    for doc_key in sorted(doc_breakdown[main_strat].keys()):
        dm = doc_breakdown[main_strat][doc_key]
        print(f"| {doc_key.upper():<8} | {dm.total_queries:>10} | {dm.recall_at_1:>8.1f}% | {dm.recall_at_3:>8.1f}% | {dm.recall_at_5:>8.1f}% | {dm.mrr:>8.4f} |")
    print("-" * 68 + "\n")

    return {
        "overall": results_table,
        "by_doc": doc_breakdown,
        "qa_count": len(qa_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia retrieval BM25 (Recall@1/3/5 e MRR) no index.db.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco index.db.")
    parser.add_argument("--qa-file", type=str, default=str(DEFAULT_QA_PATH), help="Arquivo JSONL com perguntas de teste.")
    args = parser.parse_args()

    results = evaluate_all(db_path=Path(args.db), qa_path=Path(args.qa_file))

    # Verifica se atingiu a meta de 80% na estratégia recomendada de tool-calling
    main_metrics: EvalMetrics = results["overall"][-1]["metrics"]
    target_met = main_metrics.recall_at_5 >= 80.0
    if target_met:
        logger.info("Meta de recall atingida: Recall@5 = %.1f%% (>= 80.0%%)!", main_metrics.recall_at_5)
    else:
        logger.warning("Recall@5 de %.1f%% abaixo da meta recomendada (80.0%%).", main_metrics.recall_at_5)


if __name__ == "__main__":
    main()
