#!/usr/bin/env python3
"""
validate_dataset.py - Execution filter and dataset preparation for rewriter fine-tuning.

Filtra os pares gerados garantindo que a keyword-ouro recupera o chunk-fonte no top-2
do BM25 real (corpus/build_index.py::search_reference).
Gera split 80/10/10 POR NR para avaliar generalização a normas não vistas.
Exporta no formato ChatML padronizado com o prompt de sistema do harness.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Adiciona diretório corpus ao path para importar search_reference
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "corpus"))

try:
    from build_index import search_reference
except ImportError:
    raise ImportError("Não foi possível importar search_reference de corpus/build_index.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("validate_dataset")

DEFAULT_INPUT_PATH = Path("finetune/data/synthetic_raw.jsonl")
DEFAULT_OUTPUT_DIR = Path("finetune/data")
DEFAULT_DB_PATH = Path("corpus/index_hf_36nr.db")

# System prompt oficial do Turno 1 (idêntico ao de bench/harness.py)
REWRITE_SYSTEM_PROMPT = (
    "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n"
    "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n"
    "Diretrizes mandatórias:\n"
    "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n"
    "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n"
    "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n"
    "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."
)


def validate_pair_bm25(
    item: Dict[str, Any],
    db_path: Path,
    top_k: int = 2
) -> Tuple[bool, int, List[int]]:
    """
    Executa a busca BM25 real no SQLite FTS5 com as golden_keywords geradas.
    Retorna (is_hit, rank, top_ids). O critério de hit é rank <= top_k.
    """
    kw = item.get("golden_keywords", "").strip()
    if not kw:
        return False, -1, []

    results = search_reference(str(db_path), kw, top_k=top_k)
    target_chunk_id = item.get("chunk_id")
    target_doc = (item.get("doc") or "").lower().strip()
    target_sec = (item.get("section") or "").lower().strip()

    top_ids = [r["id"] for r in results]

    for rank, r in enumerate(results, 1):
        # 1. Match por ID exato de chunk
        if target_chunk_id is not None and r["id"] == target_chunk_id:
            return True, rank, top_ids

        # 2. Match estrutural por documento e seção/item idênticos
        ret_doc = (r.get("doc") or "").lower().strip()
        ret_sec = (r.get("section") or "").lower().strip()
        if target_doc and ret_doc and target_doc == ret_doc:
            if target_sec and (target_sec == ret_sec or target_sec in ret_sec or ret_sec in target_sec):
                return True, rank, top_ids

    return False, -1, top_ids


def format_to_chatml(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formata o par validado para o formato ChatML padronizado de instrução.
    """
    history = (item.get("history") or "").strip()
    question = (item.get("question") or "").strip()
    golden_kw = (item.get("golden_keywords") or "").strip()

    if item.get("is_multiturn") and history:
        user_content = f"Histórico:\n{history}\nDúvida do trabalhador: {question}\nTermos técnicos para busca:"
    else:
        user_content = f"Dúvida do trabalhador: {question}\nTermos técnicos para busca:"

    return {
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": golden_kw}
        ],
        "metadata": {
            "id": item.get("id"),
            "chunk_id": item.get("chunk_id"),
            "doc": item.get("doc"),
            "section": item.get("section"),
            "title": item.get("title"),
            "persona": item.get("persona"),
            "register": item.get("register"),
            "asr_error": item.get("asr_error"),
            "is_multiturn": item.get("is_multiturn")
        }
    }


def split_by_nr(
    items: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Divide o conjunto em treino, validação e teste agrupando por Norma Regulamentadora (NR).
    Isso assegura que o conjunto de teste contenha NRs não vistas durante o treino,
    avaliando a capacidade de generalização conceitual do reescritor.
    """
    # Mapeia itens por NR
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        doc = (it["metadata"]["doc"] or "unknown").lower()
        by_doc.setdefault(doc, []).append(it)

    total_items = len(items)
    target_val_count = int(total_items * val_ratio)
    target_test_count = int(total_items * test_ratio)

    # Identifica NRs que devem obrigatoriamente estar no treino (as mais críticas de campo da CEMIG)
    mandatory_train_nrs = {"nr-10", "nr-06", "nr-35", "nr-12", "nr-18"}

    candidate_holdout_nrs = [d for d in by_doc.keys() if d not in mandatory_train_nrs]
    random.Random(seed).shuffle(candidate_holdout_nrs)

    test_nrs: Set[str] = set()
    val_nrs: Set[str] = set()
    current_test_count = 0
    current_val_count = 0

    # Aloca NRs para teste
    for nr in candidate_holdout_nrs:
        cnt = len(by_doc[nr])
        if current_test_count + cnt <= target_test_count * 1.3 or not test_nrs:
            test_nrs.add(nr)
            current_test_count += cnt
            if current_test_count >= target_test_count:
                break

    # Aloca NRs para validação
    remaining = [nr for nr in candidate_holdout_nrs if nr not in test_nrs]
    for nr in remaining:
        cnt = len(by_doc[nr])
        if current_val_count + cnt <= target_val_count * 1.3 or not val_nrs:
            val_nrs.add(nr)
            current_val_count += cnt
            if current_val_count >= target_val_count:
                break

    train_nrs = set(by_doc.keys()) - test_nrs - val_nrs

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []

    for nr, doc_items in by_doc.items():
        if nr in test_nrs:
            test_items.extend(doc_items)
        elif nr in val_nrs:
            val_items.extend(doc_items)
        else:
            train_items.extend(doc_items)

    logger.info("Split por NR concluído:")
    logger.info("  Treino:     %d itens (%d NRs: %s)", len(train_items), len(train_nrs), sorted(list(train_nrs)))
    logger.info("  Validação:  %d itens (%d NRs: %s)", len(val_items), len(val_nrs), sorted(list(val_nrs)))
    logger.info("  Teste (NV): %d itens (%d NRs: %s)", len(test_items), len(test_nrs), sorted(list(test_nrs)))

    return train_items, val_items, test_items


def main():
    parser = argparse.ArgumentParser(description="Validação BM25 e preparação de datasets para fine-tuning do rewriter.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Arquivo JSONL com pares gerados")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Diretório de saída dos splits")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco SQLite FTS5")
    parser.add_argument("--top-k", type=int, default=2, help="Critério de recuperação BM25 (padrão top-2)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Proporção para validação (0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Proporção para teste não visto (0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade do split")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando e validando pares de %s via BM25 (top-%d)...", input_path, args.top_k)

    total_candidates = 0
    valid_items: List[Dict[str, Any]] = []
    rejected_count = 0
    rank1_count = 0
    rank2_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_candidates += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            is_hit, rank, top_ids = validate_pair_bm25(item, db_path, top_k=args.top_k)
            if is_hit:
                chatml_item = format_to_chatml(item)
                valid_items.append(chatml_item)
                if rank == 1:
                    rank1_count += 1
                elif rank == 2:
                    rank2_count += 1
            else:
                rejected_count += 1

    pass_rate = (len(valid_items) / max(total_candidates, 1)) * 100
    logger.info(
        "Filtro de Execução Concluído: %d/%d pares aprovados (%.1f%% pass rate). Rank 1: %d, Rank 2: %d, Rejeitados: %d",
        len(valid_items), total_candidates, pass_rate, rank1_count, rank2_count, rejected_count
    )

    if not valid_items:
        logger.error("Nenhum par válido aprovado no filtro BM25! Verifique os termos gerados.")
        sys.exit(1)

    # Realiza o split por NR para assegurar generalização
    train_set, val_set, test_set = split_by_nr(
        valid_items,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Grava os arquivos de split
    splits = {
        "train.jsonl": train_set,
        "val.jsonl": val_set,
        "test.jsonl": test_set
    }

    for filename, dataset in splits.items():
        split_path = output_dir / filename
        with open(split_path, "w", encoding="utf-8") as out:
            for it in dataset:
                out.write(json.dumps(it, ensure_ascii=False) + "\n")
        logger.info("Gravado: %s (%d exemplos)", split_path, len(dataset))


if __name__ == "__main__":
    main()
