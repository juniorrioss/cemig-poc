#!/usr/bin/env python3
"""
analyze_f3.py — consolida a grade F3 vs F0 (e opcionalmente vs A-r32-ep2 da
F2) nas 4 células, com as 2 métricas-alvo e as 3 guardas do brief:

  alvo 1: ranking (1-de-5) aprovação  >= 78%  (F0: 65,8%)
  alvo 2: distrator alucinação        < 10%   (F0: 28,3%)
  guardas: oráculo ~83,7% | closed-book ~5,9% | probe capacidade (à parte)
  + recusa-correta (distrator) e recusa-indevida (respondíveis) — trade-off
    da sft_v3 reportado explicitamente.

Fontes: data/scored/scored_{tag}_{cell}.jsonl (F3) e
bench/lfm3_f0/data/scored_{cell}.jsonl (baseline).

Saída: data/consolidation.json + tabela markdown no stdout (cola no RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
from analyze_f0 import agg  # noqa: E402
from common_f0 import CELLS, VOCABS, read_jsonl  # noqa: E402

F0_DATA = REPO / "bench" / "lfm3_f0" / "data"
DATA = HERE / "data"


def collect(tag: str) -> dict:
    """Agrega por célula (all + por vocabulário)."""
    out = {}
    for cell in CELLS:
        if tag == "F0-baseline":
            items = read_jsonl(F0_DATA / f"scored_{cell}.jsonl")
        else:
            items = read_jsonl(DATA / "scored" / f"scored_{tag}_{cell}.jsonl")
        if not items:
            continue
        cell_out = {"all": agg(items)}
        by_voc = defaultdict(list)
        for i in items:
            by_voc[i["vocabulary_level"]].append(i)
        for v in VOCABS:
            cell_out[v] = agg(by_voc.get(v, []))
        out[cell] = cell_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="", help="tags F3, vírgula (default: tudo em data/scored)")
    args = ap.parse_args()

    if args.tags:
        tags = args.tags.split(",")
    else:
        import re
        tags = sorted({m.group(1) for p in (DATA / "scored").glob("scored_*.jsonl")
                       if (m := re.match(r"scored_(.+)_(closed_book|oraculo|ranking|distrator)$", p.stem))})
    all_tags = ["F0-baseline"] + tags
    consolidation = {t: collect(t) for t in all_tags}
    (DATA / "consolidation.json").write_text(
        json.dumps(consolidation, indent=1, ensure_ascii=False), encoding="utf-8")

    # tabela mestre: linhas = checkpoint, colunas = métricas-alvo + guardas
    print("\n| checkpoint | ranking aprov. (alvo >=78%) | distrator alucin. (alvo <10%) | "
          "distrator recusa-correta | oráculo aprov. (guarda ~83,7%) | "
          "closed-book aprov. (guarda ~5,9%) | recusa-indevida (ranking) | recusa-indevida (oráculo) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in all_tags:
        c = consolidation.get(t, {})
        def g(cell, key):
            v = c.get(cell, {}).get("all", {}).get(key)
            return f"{100*v:.1f}%" if v is not None else "—"
        print(f"| {t} | {g('ranking','aprovacao')} | {g('distrator','alucinacao')} | "
              f"{g('distrator','recusa_correta')} | {g('oraculo','aprovacao')} | "
              f"{g('closed_book','aprovacao')} | {g('ranking','recusa_indevida')} | "
              f"{g('oraculo','recusa_indevida')} |")
    print("\n[ok] data/consolidation.json")


if __name__ == "__main__":
    main()
