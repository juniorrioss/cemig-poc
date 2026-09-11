#!/usr/bin/env python3
"""
build_judge_input.py — Empacota as respostas do shootout (data/responses_<cand>_<cond>.json)
no formato do bench/judge.py (models -> results -> classic_rag -> items).

Cada CÉLULA (candidato × condição) vira um "modelo" com chave '<cand>__<cond>', para o
analyze.py separar os eixos por célula. Modo fixo 'classic_rag' (síntese de 1 turno).

A resposta enviada ao juiz é a `response` FINAL (com a fonte estruturada já anexada quando
a condição pede — o brief exige que a resposta seja julgada COM a citação anexada).

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=str(_HERE / "data" / "responses_*.json"))
    ap.add_argument("--out", default=str(_HERE / "data" / "harness_for_judge.json"))
    args = ap.parse_args()

    models = {}
    for path in sorted(glob.glob(args.pattern)):
        p = Path(path)
        # nome: responses_<cand>_<cond>.json ; cond ∈ {baseline,full,fewshot,citation,ordering}
        stem = p.stem[len("responses_"):]
        for cond in ("baseline", "full", "fewshot", "citation", "ordering"):
            suf = "_" + cond
            if stem.endswith(suf):
                cand = stem[: -len(suf)]
                cell = f"{cand}__{cond}"
                break
        else:
            print(f"AVISO: ignorando {p.name} (condição não reconhecida)")
            continue

        data = json.loads(p.read_text(encoding="utf-8"))
        items = []
        for it in data["items"]:
            items.append({
                "item_id": it["item_id"],
                "question": it["question"],
                "golden_answer": it["golden_answer"],
                "doc": it["doc"],
                "section": it["section"],
                "response": it["response"],  # FINAL, com fonte anexada quando aplicável
                "retrieval_hit": it["retrieval_hit"],
                "completion_tokens": it["completion_tokens"],
                "reasoning_tokens": it.get("reasoning_tokens", 0),
                "wall_s_gpu": it.get("wall_s_gpu", 0),
            })
        models[cell] = {
            "description": f"{cand} × condição {cond} (retrieval v3, thinking-OFF)",
            "results": {"classic_rag": items},
        }

    out = {
        "metadata": {
            "benchmark": "sintese2 — shootout de sintetizadores (151 reais, retrieval v3)",
            "engine": "cuda (RTX 5070); latência GPU não vale p/ aparelho",
        },
        "models": models,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {args.out} com {len(models)} células: {', '.join(sorted(models))}")


if __name__ == "__main__":
    main()
