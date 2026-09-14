#!/usr/bin/env python3
"""
extract_samples.py — Amostras literais dos casos do capitão, lado a lado nos 3 modos.

Para cada caso do capitão (13,8 kV / poste bambo / camiseta rasgada / aliança), mostra a
resposta LITERAL do tools_r128 nos modos oracle / hibrido / pure (Q4, run1) com a decisão de
chamar, a consulta gerada, o got_gold e a norma citada. Base para a seção de amostras do README.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"

CASES = {
    "qa-013": "13,8 kV (trabalhar sozinho na rede de 13,8 kV)",
    "qa-025": "poste podre/bambo (encarregado mandando subir)",
    "qa-029": "13,8 kV / distância segura do cabo energizado",
    "qa-003": "camiseta rasgada no poste (adornos)",
    "qa-004": "aliança no barramento",
}

MODES = {
    "oracle": "gen_tools_r128_oracle_q4_run1.json",
    "hibrido": "gen_tools_r128_hibrido_q4_run1.json",
    "pure": "gen_tools_r128_pure_q4_run1.json",
}


def load(fname: str) -> dict:
    d = json.loads((DATA / fname).read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in d["items"]}


def main() -> None:
    packs = {m: load(f) for m, f in MODES.items()}
    lines = []
    for cid, desc in CASES.items():
        q = packs["oracle"].get(cid, {}).get("question", "")
        lines.append(f"### {cid} — {desc}")
        lines.append(f"PERGUNTA: {q}\n")
        for m in ("oracle", "hibrido", "pure"):
            it = packs[m].get(cid)
            if not it:
                continue
            lines.append(f"[{m.upper()}] called={it.get('called_tool')} "
                         f"got_gold={it.get('got_gold')} nr={it.get('nr')!r}")
            if it.get("consulta"):
                lines.append(f"  consulta_modelo: {it['consulta']!r}")
            if it.get("retrieved"):
                secs = ", ".join(f"{c['doc']}/{c['section']}" for c in it["retrieved"])
                lines.append(f"  contexto_devolvido: {secs}")
            lines.append(f"  RESPOSTA: {it.get('response', '').strip()}")
            lines.append("")
        lines.append("-" * 80)
    out = "\n".join(lines)
    (DATA / "samples_captain.txt").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
