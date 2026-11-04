#!/usr/bin/env python3
"""
common_f5.py — caminhos, carga de dados e utilitários da F5 (alavancas de retrieval
+ compressão de contexto sobre o v4).

Duas réguas de retrieval (trava anti-mascaramento do retrieval4):
  - POC 151  : corpus/qa_pairs_v2.jsonl (check_hit oficial doc+seção, R@k/MRR);
  - BENCH    : UFGCEMIGONA/benchmark_mobile normas_regulamentadoras (1536 = 1136 com
               ouro em NORMA + 400 em MANUAL), métricas do mobile_eval (hit/evrec/
               nDCG/MRR por offsets de evidência), harness reimplementado do apêndice
               do smoke (data/lfm3-retrieval-smoke/report.md no firstmate home).

Interpretador: classifier/.venv (sklearn 1.9.1 por causa do classic_winner.pkl).
Comentários PT-BR; identificadores em inglês.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
CACHE = HERE / "cache"            # gitignored (reprodutível)

V4_DB = REPO / "android" / "app" / "src" / "main" / "assets" / "index.db"
QA151 = REPO / "corpus" / "qa_pairs_v2.jsonl"
GAB151 = REPO / "bench" / "regua" / "data" / "gabarito_151.jsonl"
GAB384 = REPO / "bench" / "lfm3_f0" / "data" / "gabarito_384.jsonl"

NRS_PARQUET = Path(os.environ.get(
    "NRS_PARQUET", "/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet"))
BENCH_PARQUET = Path(os.environ.get(
    "BENCH_PARQUET",
    str(Path.home() / ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/"
        "snapshots/5a5aa32ae4087481be828e071aff169827ec2993/"
        "normas_regulamentadoras/train-00000-of-00001.parquet")))

EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:8399")
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "

# recorte das 6 normas do F2/F3b (tratadas 10/18/34 + controle 13/31/33) = 344 queries
RECORTE_NORMS = (10, 18, 34, 13, 31, 33)
VOCABS = ("clean", "technical", "operational", "lay")


# Frente 4 — gate por INTENÇÃO para o sinal de manual (proposta se o manual diluir as
# normas): só liga quando a fala pede orientação/interpretação/exemplo/guia.
MANUAL_INTENT_RE = re.compile(
    r"\b(manual|guia|orienta[cç][aã]o|orienta[cç][oõ]es|interpreta|explica|exemplo|"
    r"como (se )?(aplica|fazer|deve ser feito|proceder)|na pr[aá]tica|recomenda)",
    re.IGNORECASE)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_TOK = re.compile(r"\w+")


def tokens_with_offsets(text: str) -> List[tuple]:
    """Tokens \\w+ sem acento/minúsculos com offsets de caractere no texto ORIGINAL.

    NFKD pode mudar o comprimento; por isso tokenizamos o texto original e só
    normalizamos cada token (offsets ficam exatos)."""
    return [(deaccent(m.group(0)).lower(), m.start(), m.end()) for m in _TOK.finditer(text)]


def nr_num(text: str) -> Optional[int]:
    m = re.search(r"nr[-_ ]?(\d+)", (text or "").lower())
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# Fontes (nrs.parquet): norma e manual por arquivo
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_sources() -> Dict[str, Any]:
    """{'by_file': {filename: texto}, 'norm_file_of_doc': {'nr-10': filename},
    'manual_file_of_doc': {'nr-10': manual_filename}}"""
    import pyarrow.parquet as pq
    rows = pq.read_table(str(NRS_PARQUET)).to_pylist()
    by_file, norm_of, man_of = {}, {}, {}
    for r in rows:
        doc = "nr-%02d" % int(r["numero"])
        by_file[r["filename"]] = r["conteudo"]
        norm_of[doc] = r["filename"]
        if r.get("manual_filename"):
            by_file[r["manual_filename"]] = r["manual"]
            man_of[doc] = r["manual_filename"]
    return {"by_file": by_file, "norm_file_of_doc": norm_of, "manual_file_of_doc": man_of}


@lru_cache(maxsize=1)
def load_bench() -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    cols = ["id", "base_qa_id", "question", "answer", "question_type", "source_id",
            "evidence", "vocabulary_level", "source_file"]
    rows = pq.read_table(str(BENCH_PARQUET), columns=cols).to_pylist()
    assert len(rows) == 1536
    man_files = set(load_sources()["manual_file_of_doc"].values())
    for r in rows:
        r["gold_kind"] = "manual" if r["source_file"] in man_files else "norma"
        r["ev"] = [(e["start_char"], e["end_char"]) for e in r["evidence"]]
    return rows


@lru_cache(maxsize=1)
def load_151() -> List[Dict[str, Any]]:
    return read_jsonl(QA151)


@lru_cache(maxsize=1)
def load_chunks() -> Dict[int, Dict[str, Any]]:
    """Metadados dos 2202 chunks do índice v4 embarcado (sem a expansão)."""
    con = sqlite3.connect(str(V4_DB))
    con.row_factory = sqlite3.Row
    out = {r["id"]: dict(r) for r in con.execute(
        "SELECT id, doc, section, title, page, text FROM chunks")}
    con.close()
    return out


def chunk_body(text: str) -> tuple:
    """Separa o prefixo 'NR-XX · seção título — ' do corpo. Retorna (prefixo, corpo, offset)."""
    i = text.find(" — ")
    if i < 0 or i > 300:
        return "", text, 0
    return text[: i + 3], text[i + 3:], i + 3
