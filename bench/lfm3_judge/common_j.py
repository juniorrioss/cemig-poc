"""
common_j.py — caminhos, IO e utilidades do bench/lfm3_judge (decision model × LLM-as-judge).

Conjuntos (todos com o MESMO schema de item, ver build_sets.py):
  V  variantes sintéticas de qualidade CONHECIDA (evolução do jaopredos/eval-vela);
  R  respostas REAIS do nosso pipeline (F6 recorte344, f4-r32-ep2) já julgadas pelo Qwen27b;
  B  benchmark_mobile_results (4 retrievers do mobilerag + o NOSSO como 5º) com veredito
     independente do gpt-oss-120b (árbitro) e do Qwen (rubrica binária).

Comentários em português; identificadores em inglês.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
CACHE = HERE / "cache"  # gitignored (reprodutível)

HF_SNAP = Path.home() / ".cache/huggingface/hub"
BENCH_PARQUET = Path(os.environ.get(
    "BENCH_PARQUET",
    str(HF_SNAP / "datasets--UFGCEMIGONA--benchmark_mobile/snapshots/"
        "5a5aa32ae4087481be828e071aff169827ec2993/normas_regulamentadoras/train-00000-of-00001.parquet")))
# UFGCEMIGONA/benchmark_mobile_results @ faeb942 (rodada 2: contexto + resposta dos 4 pipelines,
# veredito gpt-oss-120b e rubrica binária do Qwen3.8-27B-FP8). Baixado p/ cache/ (gitignored).
BMR_PARQUET = Path(os.environ.get("BMR_PARQUET", str(CACHE / "benchmark_mobile_results_nr.parquet")))
BMR_REV = "faeb942503f05a9c341141cff5ca317f11a851e8"
EVAL_VELA_REV = "0993e1ec3ff374e55042b1c32b51ebfd0b1278e7"  # jaopredos/eval-vela (base)

GAB384 = REPO / "bench" / "lfm3_f0" / "data" / "gabarito_384.jsonl"
F6_EVAL = REPO / "bench" / "lfm3_f6" / "data" / "eval"
V4_DB = REPO / "android" / "app" / "src" / "main" / "assets" / "index.db"
# Caches da F5 (gitignored, reprodutíveis por `make -C bench/lfm3_f5 ...`)
F5_CACHE = Path(os.environ.get("F5_CACHE", str(REPO / "bench" / "lfm3_f5" / "cache")))
OUR_VARIANT = "thr0.7_f2+man1.75+margin"  # ponto de operação da F6 (embarca)

JUDGE_URL = os.environ.get("REGUA_JUDGE_URL", "http://10.100.0.111:8005/v1")
JUDGE_MODEL = os.environ.get("REGUA_JUDGE_MODEL", "Qwen/Qwen3.8-27B-FP8")
DECIDER_URL = os.environ.get("DECIDER_URL", "http://10.100.0.111:8006")

METHODS = ("naive", "bm25", "mobilerag", "mobilerag_plus")


def read_text_any(path: Path | str) -> str | None:
    """Lê `x.jsonl` ou, se ausente, a cópia versionada `x.jsonl.gz` (os .jsonl crus são
    gitignored para não inflar o repo; `make pack` regrava os .gz)."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    gz = p.with_name(p.name + ".gz")
    if gz.exists():
        import gzip
        return gzip.decompress(gz.read_bytes()).decode("utf-8")
    return None


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    txt = read_text_any(path)
    if txt is None:
        return []
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def deaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def content_tokens(text: str) -> List[str]:
    """Tokens \\w+ sem acento/caixa, >=3 chars ou dígitos (cobertura lexical barata)."""
    toks = re.findall(r"[a-z0-9]+", deaccent((text or "").lower()))
    return [t for t in toks if len(t) >= 3 or t.isdigit()]


def evidence_coverage(evidence: str, context: str) -> float:
    """Fração dos tokens de conteúdo da evidência presentes no contexto (critério de
    conteúdo, idêntico para os 5 retrievers — não depende de offsets nem de chunking)."""
    ev = content_tokens(evidence)
    if not ev:
        return 0.0
    ctx = set(content_tokens(context))
    return sum(1 for t in ev if t in ctx) / len(ev)
