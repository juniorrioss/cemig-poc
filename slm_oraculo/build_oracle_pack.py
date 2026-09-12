#!/usr/bin/env python3
"""
build_oracle_pack.py — Empacota o ORÁCULO (trecho-ouro do corpus CORRIGIDO) das 151.

Saída: data/oracle_pack.json — portátil (sem dependências de índice), para que o gerador
possa rodar em QUALQUER host (Spark p/ o 2.6B; este host p/ o 27B). Cada item traz:
  - item_id, question
  - context: trecho-ouro formatado igual ao AskPipeline.kt (corpus CORRIGIDO)
  - doc, section, golden_answer (referência)

Também empacota os 2 casos fixos do capitão (contexto oráculo montado do Anexo II limpo
para o 13,8 kV; NR-35/NR-01 para o poste bambo).

Régua e retrieval ficam fora daqui — o pack é só o INSUMO de geração em oráculo.
Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "bench" / "regua"))
sys.path.insert(0, str(_HERE))

from common import load_holdout  # noqa: E402 (bench/regua)
from corpus_fix import CorrectedGoldResolver, _clean_anexo_ii_text  # noqa: E402

QA = _ROOT / "corpus" / "qa_pairs_v2.jsonl"
OUT = _HERE / "data" / "oracle_pack.json"


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Idêntico ao AskPipeline.kt (Turno 2)."""
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    parts = []
    for idx, c in enumerate(chunks):
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


def build() -> Dict[str, Any]:
    qa_full = {json.loads(l)["id"]: json.loads(l)
               for l in QA.read_text(encoding="utf-8").splitlines() if l.strip()}
    resolver = CorrectedGoldResolver()
    items: List[Dict[str, Any]] = []
    n_repaired = n_injected = 0
    for q in load_holdout():
        gc = resolver.resolve(q["doc"], q["section"], q["question"])
        n_repaired += int(any(c.get("repaired") for c in gc))
        n_injected += int(any(c.get("injected") for c in gc))
        items.append({
            "item_id": q["id"],
            "question": q["question"],
            "doc": q["doc"],
            "section": q["section"],
            "golden_answer": qa_full[q["id"]].get("golden_answer", ""),
            "context": format_context(gc),
            "retrieval_hit": True,   # oráculo: chunk-ouro entregue de bandeja
            "captain_case": False,
        })

    # Casos do capitão em oráculo (contexto ideal montado).
    anexo = _clean_anexo_ii_text()
    captain = [
        {
            "item_id": "cap-13kv",
            "question": "Qual a distancia segura para media tensao de 13,8kV?",
            "doc": "nr-10", "section": "Anexo II", "golden_answer": "",
            "context": format_context([{
                "doc": "nr-10", "section": "Anexo II",
                "title": "NR-10 · Anexo II (corpus corrigido)", "text": anexo}]),
            "retrieval_hit": True, "captain_case": True,
        },
        {
            "item_id": "cap-poste",
            "question": "o poste que preciso subir nao me parece firme, o que eu devo fazer?",
            "doc": "nr-35", "section": "35.4", "golden_answer": "",
            # Oráculo do poste bambo: NR-35 (planejamento/análise de risco de altura) +
            # NR-01 (direito de recusa). Trechos-ouro do índice canônico.
            "context": _poste_context(),
            "retrieval_hit": True, "captain_case": True,
        },
    ]

    meta = {
        "task": "poc-slm-oraculo",
        "corpus": "CORRIGIDO (reparo cirurgico Anexo II NR-10 via DocAI)",
        "n": len(items), "n_captain": len(captain),
        "n_gold_repaired": n_repaired, "n_anexo_injected": n_injected,
    }
    return {"metadata": meta, "items": items + captain}


def _poste_context() -> str:
    """Trecho-ouro do caso 'poste bambo': NR-35 análise de risco + NR-01 recusa."""
    import sqlite3
    db = sqlite3.connect(str(_ROOT / "corpus" / "index_hf_36nr.db"))
    db.row_factory = sqlite3.Row
    chunks = []
    for doc, like in (("nr-35", "35.4%"), ("nr-01", "1.4.3%")):
        rows = db.execute(
            "SELECT doc,section,title,text FROM chunks WHERE doc=? AND section LIKE ? LIMIT 1",
            (doc, like)).fetchall()
        for r in rows:
            chunks.append({"doc": r["doc"], "section": r["section"],
                           "title": r["title"], "text": r["text"]})
    db.close()
    if not chunks:  # fallback defensivo
        chunks = [{"doc": "nr-35", "section": "35.4", "title": "NR-35",
                   "text": "Trabalho em altura exige análise de risco e planejamento."}]
    return format_context(chunks)


if __name__ == "__main__":
    pack = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(pack["metadata"], ensure_ascii=False, indent=1))
