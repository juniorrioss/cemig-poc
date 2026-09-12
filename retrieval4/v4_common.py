#!/usr/bin/env python3
"""
v4_common.py — Utilidades compartilhadas do Retrieval v4 (POC CEMIG).

Reusa fielmente o harness honesto e as fontes de dados da v3/classifier para NÃO
reinventar métrica: importa `eval_common` (holdout, check_hit, app_fts_query, RankFn),
`nr_taxonomy` (36 NRs), `data_utils` (holdout do classificador) e a fusão RRF v3.

Adiciona SÓ o que a v4 precisa:
  - carga dos PARES-DE-GRAÇA (pergunta coloquial -> NR) das expansões da v3;
  - filtro anti-contaminação (TRAVA 1): remove pares derivados de chunks-ouro do holdout;
  - split por NR (TRAVA 2): reserva NRs inteiras (nr-33/16/26) fora do treino;
  - fábrica de classificador a partir de um pipeline sklearn já treinado, com
    predição top-k + probabilidades calibradas (para os gates de confiança).

Ambiente: classifier/.venv (sklearn 1.9.1). Comentários PT-BR, código em inglês.
Procedência: task poc-retrieval-v4 (branch fm/poc-retrieval-v4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))

from nr_taxonomy import CLASSES, NONE_LABEL, normalize_nr  # noqa: E402

# NRs reservadas INTEIRAS para o teste por-NR (TRAVA 2). São as mesmas 3 usadas no
# finetune2 como split de generalização: espaço confinado, periculosidade, sinalização.
HELD_OUT_NRS: List[str] = ["nr-33", "nr-16", "nr-26"]

EXPANSIONS_PATH = _ROOT / "retrieval3" / "data" / "expansions.jsonl"
LABELS_V1_PATH = _ROOT / "classifier" / "data" / "labels.jsonl"
QA_V2_PATH = _ROOT / "corpus" / "qa_pairs_v2.jsonl"
SMOKE_PATH = _ROOT / "bench" / "data" / "smoke_qa_20.jsonl"


def holdout_gold_chunk_ids() -> Set[int]:
    """IDs de chunk-ouro do holdout (151+20), p/ excluir pares de treino derivados deles.

    TRAVA 1 (anti-contaminação): embora as expansões sejam genéricas por chunk e o prompt
    nunca tenha visto as 151, as perguntas coloquiais de um chunk-ouro poderiam ficar
    próximas da pergunta real do holdout. Excluímos por precaução — barato e honesto.
    """
    ids: Set[int] = set()
    for p in (QA_V2_PATH, SMOKE_PATH):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r.get("chunk_id", -1)
            if isinstance(cid, int):
                ids.add(cid)
            for c in r.get("relevant_chunk_ids", []) or []:
                if isinstance(c, int):
                    ids.add(c)
    ids.discard(-1)
    return ids


def load_free_pairs(exclude_gold: bool = True) -> List[Dict[str, Any]]:
    """Carrega pares-de-graça (pergunta coloquial -> NR) das expansões da v3.

    Cada chunk gerou 4-6 perguntas coloquiais cuja resposta está naquele chunk; o
    rótulo de NR é o `doc` do chunk. Milhares de pares supervisionados sem custo novo.
    Retorna registros {text, primary, labels, kind='free', generator, chunk_id, doc}.
    """
    gold = holdout_gold_chunk_ids() if exclude_gold else set()
    out: List[Dict[str, Any]] = []
    gen = "Qwen/Qwen3.8-27B-FP8"  # gerador das expansões da v3 (procedência no header)
    for line in EXPANSIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = r.get("id", -1)
        if exclude_gold and cid in gold:
            continue
        doc = normalize_nr(r.get("doc", ""))
        if not doc or doc == NONE_LABEL:
            continue
        for q in r.get("perguntas", []) or []:
            q = str(q).strip()
            if not q:
                continue
            out.append({
                "text": q, "primary": doc, "labels": [doc],
                "kind": "free", "generator": gen, "chunk_id": cid, "doc": doc,
            })
    return out


def load_labels_v1() -> List[Dict[str, Any]]:
    """Dataset rotulado original do classificador (1970 falas via 27B)."""
    recs: List[Dict[str, Any]] = []
    for line in LABELS_V1_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Carga genérica de JSONL tolerante a cabeçalho de procedência (linha `# ...`)."""
    recs: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        recs.append(json.loads(line))
    return recs


def split_by_nr(records: List[Dict[str, Any]],
                held_out: Optional[List[str]] = None
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separa registros de treino em (in-split, held-out) por NR primária (TRAVA 2)."""
    ho = set(held_out if held_out is not None else HELD_OUT_NRS)
    keep, drop = [], []
    for r in records:
        (drop if r.get("primary") in ho else keep).append(r)
    return keep, drop


if __name__ == "__main__":
    import collections
    gold = holdout_gold_chunk_ids()
    free = load_free_pairs(exclude_gold=True)
    free_all = load_free_pairs(exclude_gold=False)
    v1 = load_labels_v1()
    print(f"holdout gold chunks: {len(gold)}")
    print(f"free pairs (excl gold): {len(free)} | (incl gold): {len(free_all)} "
          f"| removidos p/ trava-1: {len(free_all) - len(free)}")
    print(f"labels v1: {len(v1)}")
    keep, drop = split_by_nr(v1 + free)
    print(f"split por NR -> treino {len(keep)} | held-out {len(drop)} ({HELD_OUT_NRS})")
    print("free por NR (top):", dict(sorted(collections.Counter(r["primary"] for r in free).items(),
                                            key=lambda x: -x[1])[:10]))
