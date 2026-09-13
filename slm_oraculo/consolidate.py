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

    # Gap fechado separado por precisão (não misturar SFT com quantização).
    def _best(prefix):
        cands = {lb: v for lb, v in trained_o.items() if lb.endswith(prefix) and v is not None}
        if not cands:
            return None, None
        k = max(cands, key=lambda x: cands[x])
        return k, cands[k]

    best_bf16_lb, best_bf16 = _best("bf16")
    best_q4_lb, best_q4 = _best("q4")
    gap_bf16 = (round(100 * (best_bf16 - base_bf16_o) / (tel_27b_o - base_bf16_o), 1)
                if (best_bf16 and base_bf16_o and tel_27b_o) else None)
    gap_q4 = (round(100 * (best_q4 - base_q4_o) / (tel_27b_o - base_q4_o), 1)
              if (best_q4 and base_q4_o and tel_27b_o) else None)

    # Alucinação por célula em oráculo (critério de rejeição — ordem do capitão).
    halluc_oracle = {}
    for r in rows:
        if _match_ctx(r, "oracle"):
            halluc_oracle[r.get("label")] = r.get("hallucination_pct")

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
        "best_bf16_oracle": {"label": best_bf16_lb, "pct": best_bf16, "gap_closed_pct": gap_bf16},
        "best_q4_oracle": {"label": best_q4_lb, "pct": best_q4, "gap_closed_pct": gap_q4},
        "hallucination_oracle_pct": halluc_oracle,
        "embarque_recomendado": {
            "label": "sft_ep2_q4", "gguf": "lfm2.5-2.6b-sft_ep2-Q4_0.gguf",
            "criterio": "melhor Q4 que NAO piora alucinacao vs base (25.2% vs 25.8%)",
        },
        "veredito": (
            "Com contexto perfeito e o melhor treino, o 2.6B fecha ~36-38% do gap para o "
            "teto do 27B (Q4 45->58%, bf16 49->62%) SEM piorar alucinacao (ep2). Restam ~24 "
            "p.p. de capacidade (ex.: ler a faixa certa da tabela 13,8kV, que o treinado "
            "ainda erra: 0,25m vs 0,38m do 27B). Na busca v4 real o ganho embarcavel quase "
            "some (Q4 23.8->24.5%): o RETRIEVAL domina o teto pratico. No aparelho o 2.6B "
            "(treinado ou nao) nao cabe no orcamento de voz (~23s/4.3GB). CONCLUSAO: o SFT de "
            "destilacao e a receita certa e da salto real de qualidade, mas a POC em VOZ com "
            "2.6B e proibitiva HOJE por latencia/RAM, e o teto util esta travado pelo "
            "retrieval, nao pelo SLM. Embarcado segue 1.2B; priorizar retrieval."
        ),
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
