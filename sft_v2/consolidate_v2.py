#!/usr/bin/env python3
"""
consolidate_v2.py — Tabela mestre + veredito do SFT v2 (Parte 3).

Agrega data/score_*.json em uma tabela comparável e computa:
  - ORÁCULO 151: aprovação / cobertura / ALUCINAÇÃO por variante (v1 r16, v2 r16/r32/r64, base, 27b);
  - RETRIEVED 151 (busca v4 real): aprovação / ALUCINAÇÃO (o número que mais importa);
  - VAL pack: aprovação/alucinação na validação (fora do treino);
  - REFUSAL pack: recusa-correta% / recusa-indevida% (a NOVA MÉTRICA);
  - varredura de rank: o ganho de r é sinal ou ruído?

Saída: data/consolidation.json. Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"
OUT = DATA / "consolidation.json"

# Sufixos das suites (do eval_checkpoint_v2.sh).
SUITES = ["oracle151", "retr151", "val", "refusal"]


def load_scores() -> Dict[str, Dict[str, Any]]:
    """Mapa {score_file_stem: metrics}."""
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(DATA.glob("score_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = d.get("metrics", {})
        m["_file"] = f.name
        out[f.stem] = m
    return out


def suite_of(stem: str) -> Optional[str]:
    for s in SUITES:
        if stem.endswith("_" + s):
            return s
    return None


def variant_of(stem: str) -> str:
    """score_<variant>_<suite> -> <variant>."""
    v = stem[len("score_"):]
    for s in SUITES:
        if v.endswith("_" + s):
            return v[: -(len(s) + 1)]
    return v


def cell(scores: Dict[str, Dict[str, Any]], variant: str, suite: str) -> Optional[Dict[str, Any]]:
    return scores.get(f"score_{variant}_{suite}")


def main() -> None:
    scores = load_scores()
    variants = sorted({variant_of(k) for k in scores})

    # tabela plana.
    table: List[Dict[str, Any]] = []
    for stem, m in scores.items():
        table.append({
            "variant": variant_of(stem), "suite": suite_of(stem),
            "approval_pct": m.get("approval_pct"),
            "mean_coverage": m.get("mean_coverage"),
            "hallucination_pct": m.get("hallucination_pct"),
            "recusa_correta_pct": m.get("recusa_correta_pct"),
            "recusa_indevida_pct": m.get("recusa_indevida_pct"),
            "avg_completion_tokens": m.get("avg_completion_tokens"),
            "file": m.get("_file"),
        })
    table.sort(key=lambda x: (x["suite"] or "", -(x["approval_pct"] or 0)))

    def grab(variant: str, suite: str, key: str) -> Optional[Any]:
        c = cell(scores, variant, suite)
        return c.get(key) if c else None

    # resumo por variante nas 4 suites.
    summary: Dict[str, Any] = {}
    for v in variants:
        summary[v] = {
            "oracle151": {"approval": grab(v, "oracle151", "approval_pct"),
                          "coverage": grab(v, "oracle151", "mean_coverage"),
                          "hallucination": grab(v, "oracle151", "hallucination_pct")},
            "retrieved151": {"approval": grab(v, "retr151", "approval_pct"),
                             "hallucination": grab(v, "retr151", "hallucination_pct")},
            "val": {"approval": grab(v, "val", "approval_pct"),
                    "hallucination": grab(v, "val", "hallucination_pct")},
            "refusal": {"recusa_correta": grab(v, "refusal", "recusa_correta_pct"),
                        "recusa_indevida": grab(v, "refusal", "recusa_indevida_pct")},
        }

    # varredura de rank: r16 vs r32 vs r64 (v2).
    rank_variants = [v for v in variants if v.startswith("v2_r")]
    rank_sweep = {v: summary[v] for v in rank_variants}

    consolidation = {
        "task": "poc-sft-v2",
        "regua": "bench/regua (ruler + judge_regua 27B) + juiz de recusa (approve_nonoracle), limiar 0.5",
        "n_151": 151,
        "families": "oraculo ~50% + recusa ~20% + parcial ~20% + distrator ~10%",
        "note_v1": ("v1 (slm_oraculo r=16 SEM recusa): oraculo Q4 45->58.3%, alucinacao "
                    "25.2%; retrieved Q4 23.8->24.5%, alucinacao 41.7%. Sem metrica de recusa."),
        "summary_by_variant": summary,
        "rank_sweep_v2": rank_sweep,
        "table": table,
    }
    OUT.write_text(json.dumps(consolidation, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in consolidation.items() if k != "table"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
