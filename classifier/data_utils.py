#!/usr/bin/env python3
"""
data_utils.py — Carga de dados compartilhada (treino sintético + holdout real).

Holdout FIXO (nunca entra em treino):
  - 151 perguntas reais de corpus/qa_pairs_v2.jsonl (têm doc-ouro);
  - 20 perguntas de bench/data/smoke_qa_20.jsonl (têm doc-ouro).
Cada item de holdout expõe `gold` (nr-XX) e `text` (a fala/pergunta).

Treino: classifier/data/labels.jsonl (falas sintéticas rotuladas), com `labels`
(pode ter 1 ou 2 NRs), `primary` (rótulo principal p/ treino single-label) e `kind`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from nr_taxonomy import NONE_LABEL, normalize_nr

# Caminhos relativos à raiz do repositório (execução a partir de classifier/ ou raiz).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

TRAIN_PATH = _HERE / "data" / "labels.jsonl"
QA_V2_PATH = _ROOT / "corpus" / "qa_pairs_v2.jsonl"
SMOKE_PATH = _ROOT / "bench" / "data" / "smoke_qa_20.jsonl"


def load_train() -> List[Dict[str, Any]]:
    """Carrega falas sintéticas rotuladas (treino)."""
    recs = []
    for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def _gold(doc: str) -> str:
    """Normaliza o doc-ouro; 'fora_do_escopo' vira o rótulo especial 'nenhuma'."""
    norm = normalize_nr(doc)
    return norm if norm else NONE_LABEL


def load_holdout() -> List[Dict[str, Any]]:
    """Carrega o holdout real fixo (151 + 20), normalizando para {text, gold, source}."""
    items: List[Dict[str, Any]] = []
    for line in QA_V2_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({"id": r["id"], "text": r["question"], "gold": _gold(r["doc"]), "source": "qa_v2"})
    for line in SMOKE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({"id": r["id"], "text": r["question"], "gold": _gold(r["doc"]), "source": "smoke"})
    return items


if __name__ == "__main__":
    tr = load_train()
    ho = load_holdout()
    print(f"treino: {len(tr)} falas | holdout: {len(ho)} perguntas")
    import collections
    print("holdout por fonte:", dict(collections.Counter(h["source"] for h in ho)))
    print("holdout por gold :", dict(collections.Counter(h["gold"] for h in ho)))
