#!/usr/bin/env python3
"""
score_e2e.py — PASSO 4 (e): ponta a ponta aprovação/alucinação/recusa da arquitetura tool.

Roda o loop COMPLETO (decisão -> chamada -> busca v4 REAL -> síntese) do candidato sobre as
perguntas ANSWERABLE da suite de decisão (que carregam facts + gold_text) e julga a resposta
FINAL pela régua honesta (bench/regua) — a MESMA barra do v3/sft_1_2b. Reporta:
  - aprovação% (cobertura>=0.5, sem tautologia, sem alucinação);
  - alucinação% (juiz da régua);
  - conversão chunk-certo -> aprovada (isola a síntese do retrieval): entre os turnos em que a
    busca do PRÓPRIO modelo trouxe o chunk-ouro no top-K, quantos viraram resposta aprovada.

A alucinação usa o gold_text real como referência (o modelo pode ter buscado o chunk errado;
se afirmar dado que não está no que buscou nem no ouro, é alucinação).

Comentários PT-BR; identificadores em inglês.

Uso: ../classifier/.venv/bin/python score_e2e.py --url http://127.0.0.1:8500 --label tools_r32 \
        --topk 2 [--deterministic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import common_tools as C
from infer_tools import DET_SAMPLING, LFM_SAMPLING, run_turn

_HERE = Path(__file__).resolve().parent


def _answerable_items() -> List[Dict[str, Any]]:
    """Positivos da suite de decisão que têm facts+gold_text (dá p/ régua)."""
    suite = json.loads((_HERE / "data" / "eval_decision.json").read_text(encoding="utf-8"))
    return [it for it in suite["items"]
            if it.get("should_call") and it.get("facts") and it.get("gold_text")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING
    url = args.url.rstrip("/")
    items = _answerable_items()
    print(f"[{args.label}] e2e sobre {len(items)} answerable...", flush=True)

    def work(it):
        r = run_turn(url, [], it["question"], sampling, max_tokens=384, topk=args.topk)
        answer = r["answer"]
        # got_gold pelo check_hit oficial (doc+section OU id) — não só id (subconta vizinhos)
        from corpus.eval_retrieval import check_hit
        gold = {"doc": it.get("doc", ""), "section": it.get("section", ""),
                "chunk_id": it["gold_chunk_id"],
                "relevant_chunk_ids": [it["gold_chunk_id"]]}
        got_gold = any(check_hit({"id": c["id"], "doc": c["doc"], "section": c["section"],
                                  "text": ""}, gold) for c in r["retrieved"])
        # régua honesta
        approved = C.approve_answer(it["question"], answer, it["facts"], it["gold_text"],
                                    args.threshold)
        # alucinação isolada (juiz de desempate da régua)
        from ruler import deterministic_pass
        from judge_regua import judge_disambiguate
        rr = deterministic_pass(it["question"], answer, it["facts"], args.threshold)
        hallucination = False
        if not rr.empty:
            try:
                jr = judge_disambiguate(it["question"], answer, rr.facts_ambiguous, it["gold_text"])
                hallucination = bool(jr.get("alucinacao", False))
            except Exception:
                pass
        return {"id": it["id"], "called": r["called_tool"], "got_gold": got_gold,
                "approved": approved, "hallucination": hallucination,
                "answer": answer[:400], "consulta": r["consulta"], "nr": r["nr"]}

    results: List[Dict[str, Any]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(items)}", flush=True)
    results = [r for r in results if r]

    n = len(results)
    approved = sum(1 for r in results if r["approved"])
    halluc = sum(1 for r in results if r["hallucination"])
    got_gold = [r for r in results if r["got_gold"]]
    conv = sum(1 for r in got_gold if r["approved"])
    out = {
        "label": args.label, "n": n,
        "approval_pct": round(100 * approved / n, 1) if n else 0.0,
        "hallucination_pct": round(100 * halluc / n, 1) if n else 0.0,
        "got_gold_pct": round(100 * len(got_gold) / n, 1) if n else 0.0,
        "conv_goldcorrect_to_approved_pct":
            round(100 * conv / len(got_gold), 1) if got_gold else 0.0,
        "topk": args.topk, "deterministic": args.deterministic,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_item": results,
    }
    p = _HERE / "data" / f"e2e_{args.label}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.label}] aprov={out['approval_pct']}% aluc={out['hallucination_pct']}% "
          f"got_gold={out['got_gold_pct']}% conv={out['conv_goldcorrect_to_approved_pct']}% "
          f"-> {p}", flush=True)


if __name__ == "__main__":
    main()
