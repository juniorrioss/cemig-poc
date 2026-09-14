#!/usr/bin/env python3
"""
extract_samples_2_6b.py — Amostras literais dos casos do capitão, 1.2B vs 2.6B lado a lado.

Para cada caso do capitão (13,8 kV / poste / camiseta), mostra a resposta LITERAL do 2.6B tool
(modos oracle/hibrido/pure, Q4 run1) e, ao lado, a do 1.2B tool (de tools_oraculo/data), para o
capitão comparar diretamente qualidade e comportamento (chamar/citar/recusar).

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"
DATA_12B = _HERE.parent / "tools_oraculo" / "data"

CASES = {
    "qa-013": "13,8 kV (trabalhar sozinho na rede de 13,8 kV)",
    "qa-025": "poste podre/bambo (encarregado mandando subir)",
    "qa-029": "13,8 kV / distância segura do cabo energizado",
    "qa-003": "camiseta rasgada no poste (adornos)",
    "qa-004": "aliança no barramento",
}

LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "tools_2_6b_r64"

MODES_26 = {m: DATA / f"gen_{LABEL}_{m}_q4_run1.json" for m in ("oracle", "hibrido", "pure")}
MODES_12 = {m: DATA_12B / f"gen_tools_r128_{m}_q4_run1.json" for m in ("oracle", "hibrido", "pure")}


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in d["items"]}


def fmt(it: dict) -> list:
    if not it:
        return ["  (ausente)"]
    out = [f"  called={it.get('called_tool')} got_gold={it.get('got_gold')} nr={it.get('nr')!r}"]
    if it.get("consulta"):
        out.append(f"  consulta_modelo: {it['consulta']!r}")
    if it.get("retrieved"):
        secs = ", ".join(f"{c['doc']}/{c['section']}" for c in it["retrieved"])
        out.append(f"  contexto: {secs}")
    out.append(f"  RESPOSTA: {it.get('response', '').strip()}")
    return out


def main() -> None:
    p26 = {m: load(pth) for m, pth in MODES_26.items()}
    p12 = {m: load(pth) for m, pth in MODES_12.items()}
    lines = []
    for cid, desc in CASES.items():
        q = (p26["oracle"].get(cid) or p12["oracle"].get(cid) or {}).get("question", "")
        lines.append("=" * 80)
        lines.append(f"### {cid} — {desc}")
        lines.append(f"PERGUNTA: {q}")
        for m in ("oracle", "hibrido", "pure"):
            lines.append(f"\n--- modo {m.upper()} ---")
            lines.append("[2.6B tool]")
            lines += fmt(p26[m].get(cid))
            lines.append("[1.2B tool r128]")
            lines += fmt(p12[m].get(cid))
        lines.append("")
    out = "\n".join(lines)
    (DATA / "samples_captain.txt").write_text(out, encoding="utf-8")
    print(out[:3000])
    print(f"\n[ok] -> {DATA / 'samples_captain.txt'}")


if __name__ == "__main__":
    main()
