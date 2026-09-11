#!/usr/bin/env python3
"""
build_judge_input.py — Empacota as respostas E2E das 3 configs no formato esperado por
bench/judge.py (models -> results -> modo -> items), para reusar o juiz vLLM 27B sem
modificar o judge original.

Cada config vira um "modelo"; o modo é fixo 'classic_rag' (RAG de um turno, síntese direta),
pois o judge só computa os 4 eixos + gate por item — a decomposição de gargalo é feita à parte
em analyze.py, que lê o judge_evaluations.json resultante.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CONFIG_DESC = {
    "lfm1.2b": "LFM2.5-1.2B-Instruct QAD-Q4_0 (embarcado, baseline)",
    "lfm2.6b": "LFM2.5-2.6B QAD-Q4_0 (reasoning ON)",
    "lfm2.6b_noth": "LFM2.5-2.6B QAD-Q4_0 (reasoning OFF / --reasoning-budget 0)",
    # Configs da PARTE 1 (Retrieval v3): fusão RRF 3-sinais + prompt v1_rigido de produção.
    "v3_lfm1.2b": "v3 (RRF 3-sinais) x LFM2.5-1.2B QAD-Q4_0 (embarcado) x v1_rigido",
    "v3_lfm2.6b_noth": "v3 (RRF 3-sinais) x LFM2.5-2.6B-Q4_0 thinking-OFF x v1_rigido",
    "old_lfm1.2b_v1": "retrieval antigo x LFM2.5-1.2B QAD-Q4_0 x v1_rigido",
    "lfm2.6b_noth_v1": "retrieval antigo x LFM2.5-2.6B-Q4_0 thinking-OFF x v1_rigido",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["lfm1.2b", "lfm2.6b", "lfm2.6b_noth"])
    ap.add_argument("--out", default=str(_HERE / "data" / "harness_for_judge.json"))
    args = ap.parse_args()

    models = {}
    for cfg in args.configs:
        path = _HERE / "data" / f"responses_{cfg}.json"
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
                # metadados extras preservados p/ analyze.py (o judge os ignora)
                "completion_tokens": it["completion_tokens"],
                "reasoning_tokens": it.get("reasoning_tokens", 0),
                "wall_s_gpu": it.get("wall_s_gpu", 0),
                "stage1_mode": it.get("stage1_decision", {}).get("mode", ""),
            })
        models[cfg] = {
            "description": CONFIG_DESC.get(cfg, cfg),
            "results": {"classic_rag": items},
        }

    out = {
        "metadata": {
            "benchmark": "judge151 — pipeline híbrido E2E (151 reais)",
            "engine": "cuda (RTX 5070); latência GPU não vale p/ aparelho",
            "note": "Cada modelo = 1 config de sintetizador; modo classic_rag (síntese de 1 turno).",
        },
        "models": models,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {args.out} com {len(models)} configs.")


if __name__ == "__main__":
    main()
