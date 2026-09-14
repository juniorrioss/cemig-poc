#!/usr/bin/env python3
"""
consolidate.py — junta todos os eval_<label>.json + e2e_<label>.json numa tabela mestre.

Também computa média e desvio dos 3 runs do candidato final (emenda 1: barra de erro só no
final). Gera data/consolidation.json com:
  - tabela por candidato (decisão F1, sintaxe, args recall@2 raw/model/27b + delta, reuso, e2e);
  - curva de saturação (degraus 750/1500/3000/4500) por componente;
  - média±desvio do candidato final (3 runs).

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent


def _load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_eval(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Extrai os números-chave de um eval_<label>.json."""
    row: Dict[str, Any] = {"label": ev.get("label")}
    if "decision" in ev:
        c = ev["decision"]["confusion"]
        s = ev["decision"]["syntactic"]
        row["decision_f1"] = c["f1"]
        row["decision_prec"] = c["precision"]
        row["decision_rec"] = c["recall"]
        row["call_when_shouldnt_pct"] = c["call_when_should_not_pct"]
        row["syntactic_valid_pct"] = s["valid_pct"]
    if "args" in ev and "table" in ev["args"]:
        t = ev["args"]["table"]
        row["args_raw_r2"] = t.get("raw_question", {}).get("recall_at_2")
        if "model_rewrite" in t:
            row["args_model_r2"] = t["model_rewrite"].get("recall_at_2")
            row["args_delta_model_raw_r2"] = t.get("delta_model_minus_raw", {}).get("recall_at_2")
        row["args_27b_r2"] = t.get("llm27b_rewrite", {}).get("recall_at_2")
    if "reuse" in ev:
        rm = ev["reuse"]["metrics"]
        row["reuse_correct_pct"] = rm["reuse_correct_pct"]
        row["called_when_reuse_pct"] = rm["called_when_should_reuse_pct"]
        row["newtopic_correct_pct"] = rm["newtopic_correct_pct"]
    return row


def summarize_e2e(e: Dict[str, Any]) -> Dict[str, Any]:
    return {"label": e.get("label"), "e2e_approval_pct": e.get("approval_pct"),
            "e2e_hallucination_pct": e.get("hallucination_pct"),
            "e2e_got_gold_pct": e.get("got_gold_pct"),
            "e2e_conv_pct": e.get("conv_goldcorrect_to_approved_pct")}


def mean_std(vals: List[float]) -> Dict[str, Any]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(statistics.mean(vals), 2),
            "std": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-label", default="", help="prefixo dos 3 runs finais, ex.: tools_r32_final")
    args = ap.parse_args()

    evals = {}
    for p in sorted(glob.glob(str(_HERE / "data" / "eval_*.json"))):
        name = Path(p).stem[len("eval_"):]
        if name in ("decision", "args", "reuse", "lexical_discards"):
            continue  # são suites, não resultados
        ev = _load(Path(p))
        if ev and "label" in ev:
            evals[ev["label"]] = summarize_eval(ev)
    e2es = {}
    for p in sorted(glob.glob(str(_HERE / "data" / "e2e_*.json"))):
        e = _load(Path(p))
        if e and "label" in e:
            e2es[e["label"]] = summarize_e2e(e)

    # junta eval + e2e por label
    table = {}
    for lbl, row in evals.items():
        table[lbl] = {**row, **{k: v for k, v in e2es.get(lbl, {}).items() if k != "label"}}
    for lbl, row in e2es.items():
        if lbl not in table:
            table[lbl] = row

    # curva de saturação (labels contendo 'step')
    curve = {lbl: row for lbl, row in table.items() if "step" in lbl}

    # média/desvio dos 3 runs finais
    final_stats = {}
    if args.final_label:
        runs = [row for lbl, row in table.items() if lbl.startswith(args.final_label)]
        if runs:
            for key in ("decision_f1", "syntactic_valid_pct", "args_model_r2",
                        "args_delta_model_raw_r2", "reuse_correct_pct", "newtopic_correct_pct",
                        "e2e_approval_pct", "e2e_hallucination_pct", "e2e_conv_pct"):
                final_stats[key] = mean_std([r.get(key) for r in runs])
            final_stats["n_runs"] = len(runs)

    out = {"table": table, "saturation_curve": curve, "final_mean_std": final_stats}
    p = _HERE / "data" / "consolidation.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nsalvo {p}")


if __name__ == "__main__":
    main()
