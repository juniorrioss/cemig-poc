#!/usr/bin/env python3
"""
score_e2e.py — régua honesta (porte do bench/lfm3_f0/score_f0.py, célula answerable) sobre
as gerações fim-a-fim da Frente 3, + métricas de compressão.

Por resposta:
  APROVADO   = cobertura >= 0.5 ∧ ¬tautologia ∧ ¬alucinação ∧ ¬recusa ∧ ¬truncada
               (gabarito 384 do F0, juiz 27B desempata fatos ambíguos + alucinação
               contra as EVIDÊNCIAS do benchmark — idêntico à série LFM3);
  citação    = cita a NR (citation_present, coluna informativa da régua) e
  cite_item  = cita um CÓDIGO DE ITEM que cobre a evidência (gold_items; só onde existe
               — mede a lição do trim.py: a compressão não pode apagar o id citável);
  cite_any_item = cita algum código de item.
Agregados por célula: aprovação, cobertura, alucinação, recusa, citação NR/item, tokens de
contexto e de prompt, TTFT estimado (prefill), e o recorte "contexto contém a evidência".

TTFT estimado = n_prompt_tokens / PREFILL_TPS. PREFILL_TPS = 164 t/s (LFM2.5-1.2B Q4_0 no
S24+, medido em bench/ctx_topk, aparelho frio) × 1.2/1.7 ≈ 116 t/s para o LFM3-1.7B —
ESTIMATIVA (o LFM3 ainda não foi medido no aparelho); a diferença RELATIVA entre células
não depende da constante.

Uso: classifier/.venv/bin/python bench/lfm3_f5/score_e2e.py --tag f4-r32-ep2 [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
sys.path.insert(0, str(REPO / "bench" / "regua"))

import common_f5 as C  # noqa: E402
from common import parallel_map  # noqa: E402  (bench/regua)
from common_f0 import looks_like_refusal  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402
from ruler import deterministic_pass  # noqa: E402

PREFILL_TPS = 164.0 * 1.2 / 1.7
ITEM_IN_ANSWER = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d{1,3}){1,5})(?![\w]|\.\d)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    gens = C.read_jsonl(C.DATA / "e2e" / f"gen_{args.tag}.jsonl")
    gab = {g["base_qa_id"]: g for g in C.read_jsonl(C.GAB384)}
    rows = {r["id"]: r for r in C.load_bench()}
    out_dir = C.DATA / "e2e"
    ckpt = C.CACHE / f"scored_{args.tag}.ckpt.json"
    existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    print(f"[score] n={len(gens)} (ckpt {len(existing)})")

    def worker(g):
        row = rows[g["id"]]
        facts = gab[g["base_qa_id"]]["facts"]
        ev_text = "\n\n".join(e["text"] for e in row["evidence"])
        answer = g["answer"].replace("\u2011", "-").replace("\u2013", "-")
        r = deterministic_pass(row["question"], answer, facts)
        truncated = (not g["stopped_eos"]) and not answer.strip()
        refused = False if truncated else looks_like_refusal(answer)
        judge = None
        if answer.strip() and not r.empty:
            judge = judge_disambiguate(row["question"], answer, r.facts_ambiguous, ev_text)
            conf = {f.lower() for f in judge["fatos_presentes"]}
            extra = [f for f in r.facts_ambiguous if f.lower() in conf]
            r.coverage = (len(r.facts_present) + len(extra)) / max(1, len(facts))
            r.hallucination = bool(judge["alucinacao"])
        approved = (r.coverage >= 0.5 and not r.tautology and not r.hallucination
                    and not refused and not truncated)
        items = set(ITEM_IN_ANSWER.findall(answer))
        gi = set(g["gold_items"])
        return {
            "key": g["key"], "id": g["id"], "cell": g["cell"],
            "vocabulary_level": g["vocabulary_level"], "gold_kind": g["gold_kind"],
            "ctx_ntok": g["ctx_ntok"], "n_prompt_tokens": g["n_prompt_tokens"],
            "n_gen_tokens": g["n_gen_tokens"], "ctx_has_gold_span": g["ctx_has_gold_span"],
            "answer": answer, "coverage": round(r.coverage, 4), "tautology": r.tautology,
            "hallucination": r.hallucination, "refused": refused, "truncated": truncated,
            "citation_present": r.citation_present, "cite_any_item": bool(items),
            "cite_item": (bool(items & gi) if gi else None), "has_gold_items": bool(gi),
            "approved": approved, "judge_note": (judge or {}).get("justificativa", ""),
        }

    res = parallel_map(worker, gens, max_workers=args.workers, checkpoint_path=ckpt,
                       key_fn=lambda g: g["key"], existing=existing)
    scored = [res[g["key"]] for g in gens if g["key"] in res and "__error__" not in res[g["key"]]]
    errs = len(gens) - len(scored)
    with (out_dir / f"scored_{args.tag}.jsonl").open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[ok] {len(scored)} pontuados, {errs} erros")
    C.write_json(out_dir / f"summary_{args.tag}.json", summarize(scored))
    print_table(summarize(scored))


def _agg(rs):
    n = len(rs)
    if not n:
        return {}
    pct = lambda k: round(100 * sum(1 for r in rs if r[k]) / n, 1)  # noqa: E731
    gi = [r for r in rs if r["has_gold_items"]]
    return {
        "n": n, "approved": pct("approved"), "coverage": round(sum(r["coverage"] for r in rs) / n, 3),
        "hallucination": pct("hallucination"), "refused": pct("refused"),
        "truncated": pct("truncated"), "cite_nr": pct("citation_present"),
        "cite_any_item": pct("cite_any_item"),
        "cite_gold_item": round(100 * sum(1 for r in gi if r["cite_item"]) / max(1, len(gi)), 1),
        "n_gold_items": len(gi),
        "ctx_ntok": round(sum(r["ctx_ntok"] for r in rs) / n, 1),
        "prompt_ntok": round(sum(r["n_prompt_tokens"] for r in rs) / n, 1),
        "gen_ntok": round(sum(r["n_gen_tokens"] for r in rs) / n, 1),
        "ttft_est_s": round(sum(r["n_prompt_tokens"] for r in rs) / n / PREFILL_TPS, 2),
        "ctx_has_gold": pct("ctx_has_gold_span"),
    }


def summarize(scored):
    by = defaultdict(list)
    for s in scored:
        by[s["cell"]].append(s)
    ids_all = defaultdict(set)
    for s in scored:
        ids_all[s["cell"]].add(s["id"])
    out = {"prefill_tps_assumed": round(PREFILL_TPS, 1), "cells": {}}
    for cell, rs in sorted(by.items()):
        out["cells"][cell] = {
            "all": _agg(rs),
            "norma": _agg([r for r in rs if r["gold_kind"] == "norma"]),
            "ctx_has_gold": _agg([r for r in rs if r["ctx_has_gold_span"]]),
            "by_vocab": {v: _agg([r for r in rs if r["vocabulary_level"] == v]) for v in C.VOCABS},
        }
    # pareado vs controle a_raw2 (mesma query): ganhos/perdas de aprovação
    ctrl = {r["id"]: r["approved"] for r in by.get("a_raw2", [])}
    for cell, rs in by.items():
        win = sum(1 for r in rs if r["approved"] and not ctrl.get(r["id"], False))
        loss = sum(1 for r in rs if not r["approved"] and ctrl.get(r["id"], False))
        out["cells"][cell]["paired_vs_a"] = {"win": win, "loss": loss}
    return out


def print_table(sm):
    print(f"{'célula':<11} {'aprov':>6} {'cov':>5} {'aluc':>5} {'recusa':>6} {'citeNR':>6} "
          f"{'citeItem':>8} {'ctxTok':>7} {'prompt':>7} {'TTFT~s':>6} {'w/l vs a':>9}")
    for cell, d in sm["cells"].items():
        a, p = d["all"], d["paired_vs_a"]
        print(f"{cell:<11} {a['approved']:6.1f} {a['coverage']:5.3f} {a['hallucination']:5.1f} "
              f"{a['refused']:6.1f} {a['cite_nr']:6.1f} {a['cite_gold_item']:8.1f} "
              f"{a['ctx_ntok']:7.1f} {a['prompt_ntok']:7.1f} {a['ttft_est_s']:6.2f} "
              f"{p['win']:>4}/{p['loss']:<4}")


if __name__ == "__main__":
    main()
