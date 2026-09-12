#!/usr/bin/env python3
"""
build_judge_input.py — Empacota as respostas das 8 células (topK×trecho) no formato do
bench/judge.py (models -> results -> classic_rag -> items). Cada célula = um "modelo".
O juiz computa os 4 eixos + gate por item; a decomposição por eixo/gargalo e a conversão
chunk-certo->aprovada são feitas em analyze.py.

Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CELLS = [f"k{k}_{t}" for k in (2, 3, 4, 5) for t in ("full", "trim")]

DESC = {c: f"v4 embarcado × 1.2B QAD × topK={c[1]} × trecho={c.split('_')[1]}" for c in CELLS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--out", default=str(_HERE / "data" / "harness_for_judge.json"))
    args = ap.parse_args()

    models = {}
    for cell in args.cells:
        path = _HERE / "data" / f"responses_{cell}.json"
        if not path.exists():
            print(f"AVISO: {path} ausente, pulando.")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items = []
        for it in data["items"]:
            items.append({
                "item_id": it["item_id"],
                "question": it["question"],
                "golden_answer": it["golden_answer"],
                "doc": it["doc"],
                "section": it["section"],
                "response": it["response"],
                "retrieval_hit": it["retrieval_hit"],
                "completion_tokens": it["completion_tokens"],
                "prompt_tokens": it["prompt_tokens"],
                "stage1_mode": it.get("stage1_decision", {}).get("mode", ""),
            })
        models[cell] = {"description": DESC.get(cell, cell), "results": {"classic_rag": items}}

    out = {
        "metadata": {
            "benchmark": "ctx_topk — JANELA×TRECHOS (151 reais), v4 embarcado × 1.2B QAD",
            "engine": "cuda (RTX 5070); latência GPU não vale p/ aparelho (medida no S24+)",
        },
        "models": models,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {args.out} com {len(models)} células.")


if __name__ == "__main__":
    main()
