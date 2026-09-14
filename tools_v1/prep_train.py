#!/usr/bin/env python3
"""
prep_train.py — converte os diálogos em dataset de treino do LFM2.5 (tools no system).

EMENDA 2 (paridade prod=treino): a lista de tools é ANEXADA ao system EXATAMENTE como o
template faria com tools=[...] ("List of tools: [json, ...]"). Provado byte-a-byte em
test_render_parity (bake == tools=). Assim reusamos o motor de treino do sft_v2/train_v2
SEM tocá-lo: cada exemplo é {"messages": [...multiturno...], "meta": {...}} e o SFTTrainer
renderiza com o chat_template oficial (tool_calls + role tool + {% generation %} ->
assistant_only_loss nativo, mascarando o turno 'tool').

Também gera os DEGRAUS aninhados da curva de saturação (emenda 1): 750/1500/3000/4500,
estratificados por família, cada degrau SUBCONJUNTO do maior (mesma seed) — isola a variável
volume. O degrau 4500 é o dataset completo.

Saídas em data/: train_full.jsonl (4500) + train_step{750,1500,3000}.jsonl (aninhados) +
splits de avaliação (val/refusal FORA do treino, herdados por chunk do sft_v2.split_chunks
via meta.src_chunk_id — garantimos que val/refusal-test não entram no treino).

Comentários PT-BR; identificadores em inglês.
Uso: ../classifier/.venv/bin/python prep_train.py --in data/dialogs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tool_schema import SYSTEM_PROMPT_TOOLS, TOOLS

_HERE = Path(__file__).resolve().parent

STEPS = [750, 1500, 3000, 4500]
FRACTION = {"chamada_simples": 0.35, "sem_ferramenta": 0.15, "multiturno_reuso": 0.20,
            "multiturno_nova_busca": 0.15, "recusa_apos_busca": 0.15}


def bake_tools_into_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anexa a lista de tools ao system (idêntico ao template com tools=). messages[0]=system."""
    tool_list = "List of tools: [" + ", ".join(
        json.dumps(t, ensure_ascii=False) for t in TOOLS) + "]"
    out = [dict(m) for m in messages]
    assert out[0]["role"] == "system", "primeiro turno deve ser system"
    base = out[0]["content"] or ""
    out[0] = {"role": "system", "content": base + ("\n" if base else "") + tool_list}
    return out


def load_dialogs(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))
    return rows


def to_train_row(d: Dict[str, Any]) -> Dict[str, Any]:
    """Um exemplo de treino: messages com tools no system + meta (família, procedência)."""
    return {"messages": bake_tools_into_system(d["messages"]),
            "meta": {"family": d["meta"]["family"],
                     "src_chunk_id": d["meta"].get("src_chunk_id"),
                     "src_doc": d["meta"].get("doc") or d["meta"].get("src_doc")}}


def stratified_nested(rows: List[Dict[str, Any]], steps: List[int], seed: int
                      ) -> Dict[int, List[Dict[str, Any]]]:
    """Degraus ANINHADOS estratificados por família (cada menor ⊂ maior; mesma seed)."""
    by_fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    for r in shuffled:
        by_fam[r["meta"]["family"]].append(r)
    # ordem fixa por família (já embaralhada) -> prefixos aninhados
    out: Dict[int, List[Dict[str, Any]]] = {}
    for step in steps:
        picked: List[Dict[str, Any]] = []
        for fam, frac in FRACTION.items():
            k = round(frac * step)
            picked.extend(by_fam[fam][:k])
        rng2 = random.Random(seed + step)
        rng2.shuffle(picked)
        out[step] = picked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(_HERE / "data" / "dialogs.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", default="750,1500,3000,4500")
    args = ap.parse_args()

    steps = [int(x) for x in args.steps.split(",")]
    dialogs = load_dialogs(Path(args.inp))
    print(f"diálogos carregados: {len(dialogs)}")
    fam_counts = Counter(d["meta"]["family"] for d in dialogs)
    print(f"por família: {dict(fam_counts)}")

    train_rows = [to_train_row(d) for d in dialogs]

    # dataset completo
    full = _HERE / "data" / "train_full.jsonl"
    with full.open("w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_full.jsonl: {len(train_rows)} exemplos")

    # degraus aninhados
    nested = stratified_nested(train_rows, [s for s in steps if s < len(train_rows)], args.seed)
    for step, rows in nested.items():
        p = _HERE / "data" / f"train_step{step}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fc = Counter(r["meta"]["family"] for r in rows)
        print(f"train_step{step}.jsonl: {len(rows)} | {dict(fc)}")

    # prova de aninhamento: cada degrau menor ⊂ maior (por identidade de linha)
    ordered = sorted(nested.keys())
    for i in range(1, len(ordered)):
        smaller = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in nested[ordered[i-1]]}
        larger = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in nested[ordered[i]]}
        assert smaller.issubset(larger), f"degrau {ordered[i-1]} NÃO é subconjunto de {ordered[i]}"
    print("aninhamento verificado: degraus são subconjuntos crescentes.")

    # manifest
    manifest = {"n_dialogs": len(dialogs), "by_family": dict(fam_counts),
                "steps": {str(s): len(rows) for s, rows in nested.items()},
                "full": len(train_rows), "seed": args.seed,
                "system_prompt_sha": __import__("hashlib").sha256(
                    SYSTEM_PROMPT_TOOLS.encode()).hexdigest()[:16]}
    (_HERE / "data" / "prep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest salvo em data/prep_manifest.json")


if __name__ == "__main__":
    main()
