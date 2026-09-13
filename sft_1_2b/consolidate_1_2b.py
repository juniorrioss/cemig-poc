#!/usr/bin/env python3
"""
consolidate_1_2b.py — Tabela mestre + veredito do SFT do 1.2B.

Agrega os scores do 1.2B (sft_1_2b/data/score_*.json) + reusa, para comparacao maca-com-maca:
  - o 27B (teto): sft_v2/data/score_27b_*
  - o 2.6B v3_r64 (melhor atual): sft_v3/data/score_v3_r64_{q4,bf16}_*

Compara TODAS as variantes 1.2B (bases + treinados por rank, bf16 E Q4) em:
  - ORACULO 151: aprovacao / cobertura / ALUCINACAO;
  - RETRIEVED 151 (busca v4 real): aprovacao / ALUCINACAO;
  - VAL pack: aprovacao;
  - REFUSAL: recusa-correta%(recall) / recusa-indevida%(FP) / precision / F1 / matriz.

O numero CENTRAL (ordem do capitao: "quanto o treino vai auxiliar o modelo pequeno") e o
DELTA de cada variante treinada contra o proprio base (base_qad_q4, o embarcado). E mede o
RISCO QAD: base_qad_q4 vs base_q4 (requant simples) vs base_bf16, e treinado bf16 vs Q4.

Saida: data/consolidation.json. Comentarios PT-BR; codigo em ingles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
V2 = _HERE.parent / "sft_v2"
V3 = _HERE.parent / "sft_v3"
DATA = _HERE / "data"
OUT = DATA / "consolidation.json"
sys.path.insert(0, str(V3))
from refusal_metrics import confusion  # noqa: E402

# Variantes locais do 1.2B (bases + varredura de rank).
BASES_1_2B = ["base_qad_q4", "base_bf16", "base_q4"]
TRAINED_1_2B = [f"sft_r{r}_{q}" for r in (16, 32, 64, 128) for q in ("q4", "bf16")]

# Reuso p/ comparacao (2.6B melhor atual + teto).
REUSE = {
    "v3_r64_q4": V3 / "data",
    "v3_r64_bf16": V3 / "data",
    "27b": V2 / "data",
}


def load_score(base: Path, variant: str, suite: str) -> Optional[Dict[str, Any]]:
    f = base / f"score_{variant}_{suite}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def metrics_of(base: Path, variant: str, suite: str) -> Dict[str, Any]:
    d = load_score(base, variant, suite)
    return (d.get("metrics") or {}) if d else {}


def refusal_confusion(base: Path, variant: str) -> Dict[str, Any]:
    d = load_score(base, variant, "refusal")
    return confusion(d.get("items", [])) if d else {}


def summarize(variant: str, base: Path) -> Dict[str, Any]:
    o = metrics_of(base, variant, "oracle151")
    r = metrics_of(base, variant, "retr151")
    v = metrics_of(base, variant, "val")
    conf = refusal_confusion(base, variant)
    return {
        "oracle151": {"approval": o.get("approval_pct"), "coverage": o.get("mean_coverage"),
                      "hallucination": o.get("hallucination_pct"),
                      "avg_tok": o.get("avg_completion_tokens")},
        "retrieved151": {"approval": r.get("approval_pct"),
                         "hallucination": r.get("hallucination_pct")},
        "val": {"approval": v.get("approval_pct")},
        "refusal_classifier": conf,
    }


def main() -> None:
    summary: Dict[str, Any] = {}
    for v in BASES_1_2B + TRAINED_1_2B:
        summary[v] = summarize(v, DATA)
    for v, base in REUSE.items():
        summary[v] = summarize(v, base)

    order = BASES_1_2B + TRAINED_1_2B + list(REUSE.keys())
    table: List[Dict[str, Any]] = []
    for v in order:
        s = summary[v]
        conf = s.get("refusal_classifier") or {}
        table.append({
            "variant": v,
            "oracle_appr": s["oracle151"]["approval"],
            "oracle_cov": s["oracle151"]["coverage"],
            "oracle_hall": s["oracle151"]["hallucination"],
            "retr_appr": s["retrieved151"]["approval"],
            "retr_hall": s["retrieved151"]["hallucination"],
            "val_appr": s["val"]["approval"],
            "recusa_correta": conf.get("recusa_correta_recall_pct"),
            "recusa_indevida": conf.get("recusa_indevida_fp_pct"),
            "precision": conf.get("precision_pct"),
            "f1": conf.get("f1_pct"),
            "avg_tok": s["oracle151"]["avg_tok"],
            "confusion": {k: conf.get(k) for k in ("TP", "FN", "FP", "TN")},
        })

    # DELTA central: cada treinado vs o base EMBARCADO (base_qad_q4).
    base_ref = summary.get("base_qad_q4", {})

    def _d(a: Any, b: Any) -> Optional[float]:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(a - b, 1)
        return None

    deltas: Dict[str, Any] = {}
    for v in TRAINED_1_2B:
        s = summary[v]
        deltas[v] = {
            "d_oracle_appr": _d(s["oracle151"]["approval"], base_ref.get("oracle151", {}).get("approval")),
            "d_oracle_hall": _d(s["oracle151"]["hallucination"], base_ref.get("oracle151", {}).get("hallucination")),
            "d_retr_appr": _d(s["retrieved151"]["approval"], base_ref.get("retrieved151", {}).get("approval")),
        }

    consolidation = {
        "task": "poc-sft-1.2b",
        "regua": "bench/regua (ruler + judge_regua 27B) + juiz de recusa, limiar 0.5",
        "base_hf": "LiquidAI/LFM2.5-1.2B-Instruct (bf16; NAO existe checkpoint QAD em HF)",
        "embarcado": "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf (QAD; referencia base_qad_q4)",
        "families": "MESMOS dados do v3: oraculo 75.5% + recusa 9.8% + parcial 9.8% + distrator 4.9%",
        "delta_ref": "base_qad_q4 (o modelo embarcado hoje)",
        "deltas_vs_qad": deltas,
        "summary_by_variant": summary,
        "table": table,
    }
    OUT.write_text(json.dumps(consolidation, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'variant':16s} {'orcAppr':>7s} {'orcCov':>6s} {'orcHall':>7s} {'retAppr':>7s} "
           f"{'retHall':>7s} {'valAppr':>7s} {'recCorr':>7s} {'recInd':>6s} {'prec':>6s} "
           f"{'F1':>6s} {'tok':>5s}")
    print(hdr)
    for row in table:
        def f(x: Any, d: int = 1) -> str:
            return f"{x:.{d}f}" if isinstance(x, (int, float)) else "-"
        print(f"{row['variant']:16s} {f(row['oracle_appr']):>7s} {f(row['oracle_cov'],2):>6s} "
              f"{f(row['oracle_hall']):>7s} {f(row['retr_appr']):>7s} {f(row['retr_hall']):>7s} "
              f"{f(row['val_appr']):>7s} {f(row['recusa_correta']):>7s} "
              f"{f(row['recusa_indevida']):>6s} {f(row['precision']):>6s} {f(row['f1']):>6s} "
              f"{f(row['avg_tok'],0):>5s}")
    print("\n=== DELTA vs base_qad_q4 (o embarcado) — o numero central ===")
    for v, d in deltas.items():
        print(f"{v:16s} d_orcAppr={str(d['d_oracle_appr']):>6s}  "
              f"d_orcHall={str(d['d_oracle_hall']):>6s}  d_retAppr={str(d['d_retr_appr']):>6s}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
