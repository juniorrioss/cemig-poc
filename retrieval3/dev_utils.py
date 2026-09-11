#!/usr/bin/env python3
"""
dev_utils.py — carga do dev-set sintético (calibração). NUNCA usa holdout.

O dev-set (retrieval3/data/dev_set.jsonl) tem o mesmo formato de QAItem, com
gold em nível de chunk (chunk_id + relevant_chunk_ids). Serve só p/ calibrar
hiperparâmetros (k do RRF, pesos de fusão, peso do campo expansion, thresholds).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from eval_common import QAItem

_HERE = Path(__file__).resolve().parent
DEV_PATH = _HERE / "data" / "dev_set.jsonl"


def load_devset(path: Path = DEV_PATH) -> List[QAItem]:
    items: List[QAItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        items.append(QAItem(
            id=r["id"], question=r["question"],
            doc=str(r.get("doc", "") or "").lower(),
            section=str(r.get("section", "") or ""),
            chunk_id=r.get("chunk_id", -1),
            relevant_chunk_ids=r.get("relevant_chunk_ids", []),
            query_terms=r.get("query_terms", ""),
            source="dev",
        ))
    return items


if __name__ == "__main__":
    d = load_devset()
    print(f"dev-set: {len(d)} perguntas")
