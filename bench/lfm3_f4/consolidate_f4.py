#!/usr/bin/env python3
"""
consolidate_f4.py — tabela mestre do F4 (estáticas + tool) em
data/consolidation.json. Lê data/scored/ (células estáticas e tools_*.json).

Uso: classifier/.venv/bin/python bench/lfm3_f4/consolidate_f4.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORED = HERE / "data" / "scored"


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def static_metrics(tag: str) -> dict:
    out = {}
    for cell in ("ranking", "distrator", "oraculo", "closed_book"):
        p = SCORED / f"scored_{tag}_{cell}.jsonl"
        if not p.exists():
            continue
        rows = load(p)
        n = len(rows)
        m = {"n": n, "approved_pct": round(sum(r["approved"] for r in rows) / n * 100, 1)}
        if cell == "ranking":
            m["refusal_undue_pct"] = round(sum(1 for r in rows if r["refusal_undue"]) / n * 100, 1)
        if cell == "distrator":
            m["refusal_correct_pct"] = round(sum(1 for r in rows if r["refusal_correct"]) / n * 100, 1)
            m["hallucination_pct"] = round(sum(1 for r in rows if r["hallucination"]) / n * 100, 1)
        out[cell] = m
    return out


def main() -> None:
    tags_static = ["f4-r32-ep1", "f4-r32-ep2", "f4-r32-ep3", "f4-mhx3-ep2"]
    tags_tools = ["f4-base-notools", "f4-r32-ep2", "f4-r32-ep3",
                  "f4-mhx3-ep1", "f4-mhx3-ep2"]
    cons = {
        "references": {
            "f3b-data2x-ep4 (campeão, SEM tools; bateria 1536 F3b)": {
                "ranking_approved_pct": 77.0, "distrator_hallucination_pct": 24.9,
                "oraculo_approved_pct": 85.4, "closed_book_approved_pct": 7.4,
                "ranking_refusal_undue_pct": 8.3, "probe_general": 3.17},
        },
        "static": {t: static_metrics(t) for t in tags_static},
        "tools": {},
        "probe": {
            "f4-r32-ep2": {"general": 3.38, "behavioral": 3.00},
            "f4-r32-ep3": {"general": 3.45, "behavioral": 3.00},
            "f4-mhx3-ep2": {"general": 3.38, "behavioral": 3.45},
        },
    }
    for t in tags_tools:
        p = SCORED / f"tools_{t}.json"
        if p.exists():
            cons["tools"][t] = json.loads(p.read_text(encoding="utf-8"))
    out = HERE / "data" / "consolidation.json"
    out.write_text(json.dumps(cons, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
