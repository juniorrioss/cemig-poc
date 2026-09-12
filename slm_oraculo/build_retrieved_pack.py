#!/usr/bin/env python3
"""
build_retrieved_pack.py — Pack de contexto RECUPERADO pela busca v4 embarcada (FASE 3).

Reusa o cache do retrieval v4 embarcado (bench/prompt_teto/data/chunks_v4.json, R@2 51%),
formatando o contexto igual ao AskPipeline.kt. É o "o que sobra na prática" — mede a
aprovação com a busca real, não em oráculo.

Aplica o reparo do corpus corrigido aos chunks recuperados (substitui glifos PUA do Anexo II
pelo texto DocAI limpo) para paridade com o oráculo corrigido.

Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from corpus_fix import has_pua, _clean_anexo_ii_text  # noqa: E402

CACHE = _ROOT / "bench" / "prompt_teto" / "data" / "chunks_v4.json"
OUT = _HERE / "data" / "retrieved_pack.json"


def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def build() -> Dict[str, Any]:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    anexo = _clean_anexo_ii_text()
    items: List[Dict[str, Any]] = []
    n_repaired = 0
    for it in cache["items"].values():
        chunks = []
        for c in it["chunks"]:
            if has_pua(c["text"]):
                chunks.append({**c, "text": anexo})
                n_repaired += 1
            else:
                chunks.append(c)
        items.append({
            "item_id": it["item_id"], "question": it["question"],
            "doc": it["doc"], "section": it["section"],
            "golden_answer": it.get("golden_answer", ""),
            "context": format_context(chunks),
            "retrieval_hit": it.get("retrieval_hit"),
            "captain_case": False,
        })
    meta = {
        "task": "poc-slm-oraculo", "context": "retrieved_v4_embarcado",
        "retriever": cache["metadata"]["retriever"],
        "recall_at2_pct": cache["metadata"]["recall_at_k_pct"],
        "n": len(items), "n_pua_repaired": n_repaired,
    }
    return {"metadata": meta, "items": items}


if __name__ == "__main__":
    pack = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(pack["metadata"], ensure_ascii=False, indent=1))
