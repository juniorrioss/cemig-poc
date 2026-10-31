#!/usr/bin/env python3
"""
analyze_f3b.py — tabela de dominância do sweep F3b.

Agrega, por célula do sweep (eixo, valor, época), as métricas da avaliação
rápida padronizada: ranking-aprov + recusa-indevida (célula ranking) e
alucinação + recusa-correta (célula distrator), no RECORTE das 6 normas do F2
(344 queries), mais o probe de capacidade geral quando existir.

Referências F0/F3 são recomputadas NO MESMO recorte a partir dos scored
originais (comparação maçã-com-maçã; os números de 1536 do RESULTS da F3 não
são comparáveis ao recorte).

Saída: data/dominance.json + tabela markdown no stdout.
Uso:   classifier/.venv/bin/python bench/lfm3_f3b/analyze_f3b.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
F0_DATA = REPO / "bench" / "lfm3_f0" / "data"
F3_SCORED = REPO / "bench" / "lfm3_f3" / "data" / "scored"
F3B_SCORED = HERE / "data" / "scored"
PROBE_DIR = REPO / "bench" / "lfm3_f2" / "probe"

NORMS = {10, 18, 34, 13, 31, 33}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


GAB = {r["base_qa_id"]: r["source_file"]
       for r in read_jsonl(F0_DATA / "gabarito_384.jsonl")}


def nr_of(sf: str) -> int | None:
    m = re.search(r"nr[-_ ]?(\d+)", sf.lower())
    return int(m.group(1)) if m else None


def subset(rows: list[dict]) -> list[dict]:
    return [r for r in rows if nr_of(GAB[r["base_qa_id"]]) in NORMS]


def cell_metrics(ranking: list[dict], distrator: list[dict]) -> dict | None:
    r, d = subset(ranking), subset(distrator)
    if not r or not d:
        return None
    out = {
        "n_ranking": len(r), "n_distrator": len(d),
        "ranking_aprov": sum(x["approved"] for x in r) / len(r),
        "recusa_indevida": sum(bool(x["refusal_undue"]) for x in r) / len(r),
        "haluc": sum(x["hallucination"] for x in d) / len(d),
        "recusa_correta": sum(bool(x["refusal_correct"]) for x in d) / len(d),
        "trunc": sum(x["truncated"] for x in d) / len(d),
    }
    return out


def probe_metrics(tag: str) -> dict | None:
    p = PROBE_DIR / f"probe_scored_{tag}.jsonl"
    alt = HERE / "data" / "probe" / f"probe_scored_{tag}.jsonl"
    rows = read_jsonl(p) or read_jsonl(alt)
    if not rows:
        return None
    gen = [r for r in rows if r["suite"] == "general"]
    beh = [r for r in rows if r["suite"] == "behavioral"]
    return {
        "probe_general": sum(r["qualidade"] for r in gen) / max(1, len(gen)),
        "probe_behavioral": sum(r["qualidade"] for r in beh) / max(1, len(beh)),
        "probe_pt": sum(r["pt"] for r in rows) / len(rows),
    }


def main() -> None:
    table: dict[str, dict] = {}

    # referências no recorte
    table["F0-baseline"] = cell_metrics(
        read_jsonl(F0_DATA / "scored_ranking.jsonl"),
        read_jsonl(F0_DATA / "scored_distrator.jsonl"))
    table["F0-baseline"].update(
        {"probe_general": 3.92, "probe_behavioral": 3.80})  # RESULTS F2/F3
    for ep in range(1, 9):
        t = f"f3-r32-ep{ep}"
        m = cell_metrics(read_jsonl(F3_SCORED / f"scored_{t}_ranking.jsonl"),
                         read_jsonl(F3_SCORED / f"scored_{t}_distrator.jsonl"))
        if m:
            pm = probe_metrics(f"F3-r32-ep{ep}")
            if pm:
                m.update(pm)
            table[t] = m

    # células do sweep
    tags = sorted({re.match(r"scored_(.+)_(ranking|distrator)\.jsonl", p.name).group(1)
                   for p in F3B_SCORED.glob("scored_*_*.jsonl")})
    for t in tags:
        m = cell_metrics(read_jsonl(F3B_SCORED / f"scored_{t}_ranking.jsonl"),
                         read_jsonl(F3B_SCORED / f"scored_{t}_distrator.jsonl"))
        if m:
            pm = probe_metrics(t)
            if pm:
                m.update(pm)
            table[t] = m

    out = HERE / "data" / "dominance.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False))

    ref = table.get("f3-r32-ep4", {})
    print("| célula | ranking | Δrk | haluc | Δhal | recusa-cor | rec-indev | trunc | probe-g |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, m in table.items():
        if not m:
            continue
        drk = (m["ranking_aprov"] - ref.get("ranking_aprov", 0)) * 100
        dh = (m["haluc"] - ref.get("haluc", 0)) * 100
        pg = m.get("probe_general")
        pg_s = f"{pg:.2f}" if pg else "—"
        print(f"| {tag} | {m['ranking_aprov']*100:.1f}% | {drk:+.1f} | "
              f"{m['haluc']*100:.1f}% | {dh:+.1f} | {m['recusa_correta']*100:.1f}% | "
              f"{m['recusa_indevida']*100:.1f}% | {m['trunc']*100:.1f}% | {pg_s} |")
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
