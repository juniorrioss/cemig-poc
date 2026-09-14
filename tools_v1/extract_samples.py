#!/usr/bin/env python3
"""
extract_samples.py — amostras LITERAIS renderizadas para o README (uma de cada família).

Escolhe 1 diálogo de cada família do dataset e o renderiza no formato nativo do LFM2.5
(render_jinja == apply_chat_template) para colar no relatório. Prioriza, quando existirem,
diálogos que toquem os casos do capitão (13,8 kV, poste, camiseta).

Uso: ../classifier/.venv/bin/python extract_samples.py --in data/dialogs.jsonl \
        --out data/samples_dataset.txt
Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import render_jinja as RJ
from tool_schema import TOOLS

_HERE = Path(__file__).resolve().parent

CAPTAIN_KEYS = {
    "13,8 kv": ["13,8", "13.8", "13800"],
    "poste": ["poste"],
    "camiseta": ["camiseta", "roupa", "vestiment", "algod"],
}


def q_of(d: Dict[str, Any]) -> str:
    for m in d["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def pick_family(rows: List[Dict[str, Any]], fam: str,
                prefer: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    cands = [d for d in rows if d["meta"]["family"] == fam]
    if not cands:
        return None
    if prefer:
        for d in cands:
            ql = q_of(d).lower()
            if any(p in ql for p in prefer):
                return d
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(_HERE / "data" / "dialogs.jsonl"))
    ap.add_argument("--out", default=str(_HERE / "data" / "samples_dataset.txt"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

    fams = ["chamada_simples", "sem_ferramenta", "multiturno_reuso",
            "multiturno_nova_busca", "recusa_apos_busca"]
    out_lines: List[str] = []
    for fam in fams:
        prefer = None
        if fam == "chamada_simples":
            prefer = CAPTAIN_KEYS["13,8 kv"]
        d = pick_family(rows, fam, prefer)
        if not d:
            continue
        out_lines.append("=" * 78)
        out_lines.append(f"FAMÍLIA: {fam}   (src: {d['meta'].get('doc')} {d['meta'].get('section','')})")
        out_lines.append("=" * 78)
        out_lines.append(RJ.render(d["messages"], tools=TOOLS))
        out_lines.append("")

    # casos do capitão explícitos (busca por chunk 13,8 / poste / camiseta em qualquer família)
    for name, keys in CAPTAIN_KEYS.items():
        found = None
        for d in rows:
            if any(k in q_of(d).lower() for k in keys):
                found = d
                break
        out_lines.append("#" * 78)
        out_lines.append(f"CASO DO CAPITÃO: {name}")
        out_lines.append("#" * 78)
        if found:
            out_lines.append(f"(família {found['meta']['family']}, {found['meta'].get('doc')})")
            out_lines.append(RJ.render(found["messages"], tools=TOOLS))
        else:
            out_lines.append("(nenhum diálogo do dataset toca este caso — ver eval no README)")
        out_lines.append("")

    Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"amostras salvas em {args.out} ({len(out_lines)} linhas)")


if __name__ == "__main__":
    main()
