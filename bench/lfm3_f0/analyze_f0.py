"""
analyze_f0.py — agrega as métricas do baseline F0 em tabela célula x vocabulário.

Métricas (por célula x vocabulário):
  aprovacao, cobertura média, alucinacao, recusa (correta no distrator /
  indevida nas respondíveis), citacao (informativa, fora do gate), tautologia,
  tokens médios gerados.

Saída: data/consolidation.json + RESULTS.md (markdown).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from common_f0 import CELLS, VOCABS, read_jsonl

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def agg(items: list) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    answerable = items[0]["cell"] != "distrator"
    out = {
        "n": n,
        "aprovacao": sum(i["approved"] for i in items) / n,
        "cobertura": sum(i["coverage"] for i in items) / n,
        "alucinacao": sum(i["hallucination"] for i in items) / n,
        "tautologia": sum(i["tautology"] for i in items) / n,
        "citacao": sum(i["citation_present"] for i in items) / n,
        "tokens_medios": sum(i["n_gen_tokens"] for i in items) / n,
        "truncadas": sum(i.get("truncated", False) for i in items) / n,
    }
    if answerable:
        out["recusa_indevida"] = sum(bool(i["refusal_undue"]) for i in items) / n
    else:
        out["recusa_correta"] = sum(bool(i["refusal_correct"]) for i in items) / n
        out["respondeu_certo_mesmo_assim"] = sum(
            i.get("answered_correct_anyway", False) for i in items) / n
    return out


def main() -> None:
    consolidation = {}
    lines = ["# Resultados F0 — LFM3-1.7B bf16 baseline (benchmark_mobile, chunks_512)", ""]

    for cell in CELLS:
        scored = read_jsonl(DATA / f"scored_{cell}.jsonl")
        if not scored:
            continue
        by_vocab = defaultdict(list)
        for s in scored:
            by_vocab[s["vocabulary_level"]].append(s)

        cell_stats = {"all": agg(scored)}
        for v in VOCABS:
            cell_stats[v] = agg(by_vocab[v])
        # corte por question_type (informativo)
        by_qt = defaultdict(list)
        for s in scored:
            by_qt[s["question_type"]].append(s)
        cell_stats["by_question_type"] = {k: agg(v) for k, v in sorted(by_qt.items())}
        consolidation[cell] = cell_stats

        answerable = cell != "distrator"
        lines.append(f"## Célula: {cell} (n={len(scored)})")
        lines.append("")
        if answerable:
            lines.append("| vocabulário | n | aprovação | cobertura | alucinação | recusa indevida | citação | truncadas | tokens méd. |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for v in ("all",) + VOCABS:
                s = cell_stats[v]
                if s["n"] == 0:
                    continue
                lines.append(
                    f"| {v} | {s['n']} | {pct(s['aprovacao'])} | {s['cobertura']:.3f} "
                    f"| {pct(s['alucinacao'])} | {pct(s['recusa_indevida'])} "
                    f"| {pct(s['citacao'])} | {pct(s['truncadas'])} | {s['tokens_medios']:.0f} |")
        else:
            lines.append("| vocabulário | n | recusa correta (=aprov.) | alucinação | acertou mesmo assim | citação | truncadas | tokens méd. |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for v in ("all",) + VOCABS:
                s = cell_stats[v]
                if s["n"] == 0:
                    continue
                lines.append(
                    f"| {v} | {s['n']} | {pct(s['recusa_correta'])} "
                    f"| {pct(s['alucinacao'])} | {pct(s['respondeu_certo_mesmo_assim'])} "
                    f"| {pct(s['citacao'])} | {pct(s['truncadas'])} | {s['tokens_medios']:.0f} |")
        lines.append("")

    (DATA / "consolidation.json").write_text(
        json.dumps(consolidation, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] data/consolidation.json + RESULTS.md")
    for cell in consolidation:
        a = consolidation[cell]["all"]
        key = "aprovacao" if cell != "distrator" else "recusa_correta"
        print(f"  {cell}: n={a['n']} {key}={pct(a[key])}")


if __name__ == "__main__":
    main()
