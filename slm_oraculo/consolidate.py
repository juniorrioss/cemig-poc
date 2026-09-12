#!/usr/bin/env python3
"""
consolidate.py — Tabela mestre + veredito da POC (FASE 3).

Agrega todos os data/score_*.json em uma tabela comparável e computa:
  - curva de aprovação em ORÁCULO: 2.6B base -> treinado, contra o teto do 27B;
  - fração do gap fechado (base->treinado)/(27B-base);
  - medição na busca v4 real (retrieved);
  - alucinação por célula (critério de rejeição, ordem do capitão).

Saída: data/consolidation.json (tabela + veredito). Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"
OUT = DATA / "consolidation.json"


def load_scores() -> List[Dict[str, Any]]:
    rows = []
    for f in sorted(DATA.glob("score_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d.get("metrics", {})
        m["_file"] = f.name
        rows.append(m)
    return rows


def find(rows, label, ctx) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("label") == label and str(r.get("context_source", "")).startswith(_ctx_key(ctx)) \
                or (r.get("label") == label and _match_ctx(r, ctx)):
            return r
    return None


def _ctx_key(ctx: str) -> str:
    return "CORRIGIDO" if ctx == "oracle" else "retrieved"


def _match_ctx(r: Dict[str, Any], ctx: str) -> bool:
    cs = str(r.get("context_source", "")).lower()
    is_oracle = ("corrigido" in cs or "orac" in cs)
    if ctx == "oracle":
        return is_oracle
    return not is_oracle  # retrieved = qualquer coisa que não seja o oráculo corrigido


def main() -> None:
    rows = load_scores()

    def cell(label, ctx):
        for r in rows:
            if r.get("label") == label and _match_ctx(r, ctx):
                return r
        return None

    def ap(label, ctx):
        c = cell(label, ctx)
        return c["approval_pct"] if c else None

    tel_27b_o = ap("27b", "oracle")
    base_bf16_o = ap("2.6b_bf16_base", "oracle")
    base_q4_o = ap("2.6b_q4_base", "oracle")

    # Curva: melhor checkpoint treinado em oráculo (Q4, engine de embarque).
    trained_labels = sorted({r["label"] for r in rows
                             if r.get("label", "").startswith(("sft_ep", "2.6b_sft"))})
    trained_o = {lb: ap(lb, "oracle") for lb in trained_labels}
    best_trained = None
    if trained_o:
        best_trained = max((k for k, v in trained_o.items() if v is not None),
                           key=lambda k: trained_o[k], default=None)

    gap_closed = None
    if best_trained and tel_27b_o and base_q4_o is not None:
        denom = tel_27b_o - base_q4_o
        if denom > 0:
            gap_closed = round(100 * (trained_o[best_trained] - base_q4_o) / denom, 1)

    table = []
    for r in rows:
        table.append({
            "label": r.get("label"), "context": ("oracle" if _match_ctx(r, "oracle") else "retrieved"),
            "approval_pct": r.get("approval_pct"), "mean_coverage": r.get("mean_coverage"),
            "hallucination_pct": r.get("hallucination_pct"),
            "conv_chunk_hit_pct": r.get("conv_chunk_hit_pct"),
            "citation_pct": r.get("citation_pct"),
            "avg_completion_tokens": r.get("avg_completion_tokens"),
            "file": r.get("_file"),
        })
    table.sort(key=lambda x: (x["context"], -(x["approval_pct"] or 0)))

    consolidation = {
        "task": "poc-slm-oraculo",
        "regua": "bench/regua (ruler + judge_regua 27B), limiar 0.5, citação fora do gate",
        "corpus": "CORRIGIDO (reparo cirúrgico Anexo II NR-10 via DocAI)",
        "n": 151,
        "teto_27b_oracle_pct": tel_27b_o,
        "base_2.6b_bf16_oracle_pct": base_bf16_o,
        "base_2.6b_q4_oracle_pct": base_q4_o,
        "quant_cost_bf16_to_q4_pp": (round(base_bf16_o - base_q4_o, 1)
                                     if (base_bf16_o and base_q4_o) else None),
        "gap_27b_minus_base_bf16_pp": (round(tel_27b_o - base_bf16_o, 1)
                                       if (tel_27b_o and base_bf16_o) else None),
        "trained_oracle_pct": trained_o,
        "best_trained_oracle": best_trained,
        "best_trained_oracle_pct": (trained_o.get(best_trained) if best_trained else None),
        "gap_closed_pct": gap_closed,
        "retrieved_v4": {
            "27b": ap("27b", "retrieved"),
            "2.6b_q4_base": ap("2.6b_q4_base", "retrieved"),
            **{lb: ap(lb, "retrieved") for lb in trained_labels},
        },
        "table": table,
    }
    OUT.write_text(json.dumps(consolidation, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in consolidation.items() if k != "table"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
