#!/usr/bin/env python3
"""
corpus_fix.py — Trecho-ouro do corpus CORRIGIDO (reparo cirúrgico do Anexo II da NR-10).

Contexto (ordem do capitão): a FASE 1 deve medir o teto em ORÁCULO com os chunks-ouro do
corpus CORRIGIDO. O reparo do Anexo II da NR-10 (DocAI Form Parser) NÃO foi mesclado ao
índice de produção `corpus/index_hf_36nr.db` — a decisão registrada em docai/README.md foi
"não substituir o índice" (o aditivo polui o BM25). O texto reconstruído e verificado
(43.803 números conferidos, 18/18 faixas) vive em:
    docai/results/reconstructed_nr10_anexo_ii.json

Este módulo entrega o trecho-ouro por pergunta com DUAS camadas:
  1. resolução normal (bench/regua GoldResolver sobre o índice canônico);
  2. reparo/injeção: quando a pergunta cobra a tabela de zonas/distâncias da NR-10 (kV,
     zona de risco/controlada, Anexo II), anexamos o texto LIMPO reconstruído, substituindo
     qualquer chunk corrompido por glifo PUA.

ACHADO HONESTO (registrado no README da tarefa): NENHUM dos 151 chunks-ouro resolvidos
aponta para os chunks corrompidos 274/275/276; as perguntas de zona/distância da NR-10
resolvem para seções definitórias (10.1/10.2/10.6/Glossário). Logo o reparo do corpus
move POUCO o teto das 151 — mas o oráculo corrigido é o correto a medir, e para as
perguntas de zona/distância a tabela limpa (13,8 kV -> 0,38/1,38 m) passa a estar
disponível ao modelo, o que antes era IMPOSSÍVEL.

Comentários em PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "bench" / "regua"))

from common import GoldResolver, load_holdout  # noqa: E402

RECONSTRUCTED = _ROOT / "docai" / "results" / "reconstructed_nr10_anexo_ii.json"

# Glifos de fonte privada (PUA) que sinalizam chunk corrompido do parquet.
_PUA = ("\uf03c", "\uf03e", "\uf0b3", "\uf0a3", "\uf0b7", "\uf0b1", "\uf0d7", "\uf0b0")

# Sinais de que a pergunta cobra a tabela de zona/distância da NR-10.
_ZONE_HINT = re.compile(
    r"(kv|zona (de )?(risco|controlad|livre)|dist[aâ]ncia|anexo\s*ii|raio)",
    re.IGNORECASE)


def _clean_anexo_ii_text() -> str:
    """Monta o texto LIMPO do Anexo II a partir do artefato DocAI verificado."""
    recs = json.loads(RECONSTRUCTED.read_text(encoding="utf-8"))
    zona = [r["sentence"] for r in recs if r.get("kind") == "zona_risco"]
    header = ("NR-10 · Anexo II — Zona de risco e zona controlada. Tabela de raios de "
              "delimitação por faixa de tensão nominal (corpus corrigido via DocAI, "
              "18/18 faixas verificadas):")
    return header + "\n" + "\n".join(zona)


def has_pua(text: str) -> bool:
    return any(g in text for g in _PUA)


class CorrectedGoldResolver:
    """Resolve o trecho-ouro do corpus CORRIGIDO (com reparo cirúrgico do Anexo II NR-10)."""

    def __init__(self) -> None:
        self.base = GoldResolver()
        self.anexo_clean = _clean_anexo_ii_text()

    def resolve(self, doc: str, section: str, question: str = "") -> List[Dict[str, Any]]:
        chunks = self.base.resolve(doc, section)
        repaired: List[Dict[str, Any]] = []
        injected_anexo = False
        for c in chunks:
            if has_pua(c["text"]):
                # Chunk corrompido: substitui pelo Anexo II limpo (reparo cirúrgico).
                repaired.append({**c, "text": self.anexo_clean, "repaired": True})
                injected_anexo = True
            else:
                repaired.append(c)
        # Injeção: pergunta de zona/distância da NR-10 cujo trecho-ouro NÃO inclui a
        # tabela — anexa o Anexo II limpo (o dado que o corpus corrigido destravou).
        doc_l = (doc or "").lower()
        if (doc_l == "nr-10" and not injected_anexo
                and _ZONE_HINT.search(question or "")):
            repaired.append({
                "id": -1010, "doc": "nr-10", "section": "Anexo II",
                "title": "NR-10 · Anexo II (corpus corrigido)",
                "text": self.anexo_clean, "injected": True,
            })
        return repaired


def audit() -> Dict[str, Any]:
    """Diagnóstico: quantos trechos-ouro das 151 mudam com o corpus corrigido."""
    r = CorrectedGoldResolver()
    base = GoldResolver()
    ho = load_holdout()
    changed = []
    for q in ho:
        b = base.resolve(q["doc"], q["section"])
        c = r.resolve(q["doc"], q["section"], q["question"])
        btxt = "\n".join(x["text"] for x in b)
        ctxt = "\n".join(x["text"] for x in c)
        if btxt != ctxt:
            changed.append({"id": q["id"], "doc": q["doc"], "section": q["section"],
                            "repaired": any(x.get("repaired") for x in c),
                            "injected": any(x.get("injected") for x in c)})
    return {"n_holdout": len(ho), "n_changed": len(changed), "changed": changed}


if __name__ == "__main__":
    rep = audit()
    print(json.dumps(rep, ensure_ascii=False, indent=1))
