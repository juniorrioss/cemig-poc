#!/usr/bin/env python3
"""
consolidate_v3.py — Tabela mestre + veredito do SFT v3 (Partes 3 e 4).

Agrega os scores do v3 (sft_v3/data/score_*.json) E do v2 (sft_v2/data/score_v2_r64_*,
para comparação maçã-com-maçã) + o teto 27B e a base 2.6B (reusados do sft_v2). Junta a
métrica de recusa como classificador (refusal_metrics.confusion) na MESMA tabela.

Compara: v3_r64, v3_r128 e v2_r64 (bf16 E Q4) em:
  - ORÁCULO 151: aprovação / cobertura / ALUCINAÇÃO;
  - RETRIEVED 151 (busca v4 real): aprovação / ALUCINAÇÃO;
  - VAL pack: aprovação/alucinação na validação;
  - REFUSAL: recusa-correta%(recall) / recusa-indevida%(FP) / precision / F1 / matriz TP-FP-TN-FN.

Saída: data/consolidation.json. Comentários PT-BR; código em inglês.

Uso: classifier/.venv/bin/python consolidate_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
V2 = _HERE.parent / "sft_v2"
DATA = _HERE / "data"
OUT = DATA / "consolidation.json"
sys.path.insert(0, str(_HERE))
from refusal_metrics import confusion  # noqa: E402

SUITES = ["oracle151", "retr151", "val", "refusal"]

# Variantes do v3 (locais) + reusadas do v2 p/ comparação maçã-com-maçã.
V3_VARIANTS = ["v3_r64_q4", "v3_r64_bf16", "v3_r128_q4", "v3_r128_bf16"]
V2_REUSE = ["v2_r64_q4", "v2_r64_bf16", "base_q4_v2", "base_bf16_v2", "27b"]


def load_score(base: Path, variant: str, suite: str) -> Optional[Dict[str, Any]]:
    f = base / f"score_{variant}_{suite}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def metrics_of(base: Path, variant: str, suite: str) -> Optional[Dict[str, Any]]:
    d = load_score(base, variant, suite)
    return d.get("metrics") if d else None


def refusal_confusion(base: Path, variant: str) -> Optional[Dict[str, Any]]:
    d = load_score(base, variant, "refusal")
    if not d:
        return None
    return confusion(d.get("items", []))


def summarize(variant: str, base: Path) -> Dict[str, Any]:
    o = metrics_of(base, variant, "oracle151") or {}
    r = metrics_of(base, variant, "retr151") or {}
    v = metrics_of(base, variant, "val") or {}
    conf = refusal_confusion(base, variant) or {}
    return {
        "oracle151": {"approval": o.get("approval_pct"), "coverage": o.get("mean_coverage"),
                      "hallucination": o.get("hallucination_pct"),
                      "avg_tok": o.get("avg_completion_tokens")},
        "retrieved151": {"approval": r.get("approval_pct"), "hallucination": r.get("hallucination_pct")},
        "val": {"approval": v.get("approval_pct"), "hallucination": v.get("hallucination_pct")},
        "refusal_classifier": conf,
    }


def main() -> None:
    summary: Dict[str, Any] = {}
    for v in V3_VARIANTS:
        summary[v] = summarize(v, DATA)
    for v in V2_REUSE:
        summary[v] = summarize(v, V2 / "data")

    # tabela plana p/ leitura rápida.
    table: List[Dict[str, Any]] = []
    for v, s in summary.items():
        conf = s.get("refusal_classifier") or {}
        table.append({
            "variant": v,
            "oracle_appr": s["oracle151"]["approval"],
            "oracle_hall": s["oracle151"]["hallucination"],
            "retr_appr": s["retrieved151"]["approval"],
            "retr_hall": s["retrieved151"]["hallucination"],
            "val_appr": s["val"]["approval"],
            "recusa_correta": conf.get("recusa_correta_recall_pct"),
            "recusa_indevida": conf.get("recusa_indevida_fp_pct"),
            "precision": conf.get("precision_pct"),
            "f1": conf.get("f1_pct"),
            "confusion": {k: conf.get(k) for k in ("TP", "FN", "FP", "TN")},
        })

    consolidation = {
        "task": "poc-sft-v3",
        "regua": "bench/regua (ruler + judge_regua 27B) + juiz de recusa, limiar 0.5",
        "families_v3": "oráculo 75.5% + recusa 9.8% + parcial 9.8% + distrator 4.9% (recusa <= 25%)",
        "families_v2": "oráculo 48% + recusa 20.8% + parcial 20.8% + distrator 10.4% (recusa 52%)",
        "eval_loss": {"r64": 0.5366, "r128": 0.5098},
        "summary_by_variant": summary,
        "table": table,
    }
    OUT.write_text(json.dumps(consolidation, ensure_ascii=False, indent=1), encoding="utf-8")

    # imprime tabela legível.
    print(f"{'variant':16s} {'orcAppr':>7s} {'orcHall':>7s} {'retAppr':>7s} {'retHall':>7s} "
          f"{'valAppr':>7s} {'recCorr':>7s} {'recIndev':>8s} {'prec':>6s} {'F1':>6s}")
    for row in table:
        def f(x: Any) -> str:
            return f"{x:.1f}" if isinstance(x, (int, float)) else "-"
        print(f"{row['variant']:16s} {f(row['oracle_appr']):>7s} {f(row['oracle_hall']):>7s} "
              f"{f(row['retr_appr']):>7s} {f(row['retr_hall']):>7s} {f(row['val_appr']):>7s} "
              f"{f(row['recusa_correta']):>7s} {f(row['recusa_indevida']):>8s} "
              f"{f(row['precision']):>6s} {f(row['f1']):>6s}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
