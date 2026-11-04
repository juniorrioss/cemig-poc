#!/usr/bin/env python3
"""
score_f6.py — régua honesta sobre as gerações fim-a-fim da F6 (recorte 344 e POC 151).

Por resposta (MESMA definição do score_e2e.py da F5, para comparar com os 50,6/54,7 e
55,6/57,5 da F5):
  APROVADO = cobertura >= 0,5 ∧ ¬tautologia ∧ ¬alucinação ∧ ¬recusa ∧ ¬truncada
  - recorte 344: gabarito 384 do F0 (bench/lfm3_f0/data/gabarito_384.jsonl); o juiz 27B
    desempata fatos ambíguos e checa alucinação contra as EVIDÊNCIAS do benchmark;
  - POC 151: gabarito 151 da régua (bench/regua/data/gabarito_151.jsonl); o juiz checa
    alucinação contra o TEXTO DO TRECHO-OURO (rejudge.load_gold_texts) — régua honesta.
  cite_nr / cite_item (códigos de item que cobrem a evidência; só no recorte) /
  tokens de contexto e de prompt / TTFT estimado (prefill 116 t/s, ESTIMATIVA da F5).

Uso: classifier/.venv/bin/python bench/lfm3_f6/score_f6.py --suite recorte --tag f6-recorte
     classifier/.venv/bin/python bench/lfm3_f6/score_f6.py --suite 151 --tag f6-151
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402
import common_f5 as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "bench" / "lfm3_f0"))
sys.path.insert(0, str(C.REPO / "bench" / "regua"))
from common import parallel_map  # noqa: E402  (bench/regua)
from common_f0 import looks_like_refusal  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402
from ruler import deterministic_pass  # noqa: E402
from score_e2e import ITEM_IN_ANSWER, PREFILL_TPS  # noqa: E402


def load_refs(suite: str):
    """(perguntas por id, fatos por id, texto de referência p/ alucinação por id)."""
    if suite == "recorte":
        gab = {g["base_qa_id"]: g["facts"] for g in C.read_jsonl(C.GAB384)}
        rows = {r["id"]: r for r in C.load_bench()}
        q = {i: r["question"] for i, r in rows.items()}
        facts = {i: gab[r["base_qa_id"]] for i, r in rows.items()}
        ref = {i: "\n\n".join(e["text"] for e in r["evidence"]) for i, r in rows.items()}
        return q, facts, ref
    from rejudge import load_gold_texts
    gab = {g["id"]: g["facts"] for g in C.read_jsonl(C.GAB151)}
    q = {it["id"]: it["question"] for it in C.load_151()}
    return q, gab, load_gold_texts()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["recorte", "151"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    gens = C.read_jsonl(F6.DATA / "e2e" / f"gen_{args.tag}.jsonl")
    Q, FACTS, REF = load_refs(args.suite)
    F6.CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = F6.CACHE / f"scored_{args.tag}.ckpt.json"
    existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    print(f"[score] {args.suite} n={len(gens)} (ckpt {len(existing)})", flush=True)

    def worker(g):
        facts = FACTS[g["id"]]
        answer = g["answer"].replace("\u2011", "-").replace("\u2013", "-")
        r = deterministic_pass(Q[g["id"]], answer, facts)
        truncated = (not g["stopped_eos"]) and not answer.strip()
        refused = False if truncated else looks_like_refusal(answer)
        judge = None
        if answer.strip() and not r.empty:
            judge = judge_disambiguate(Q[g["id"]], answer, r.facts_ambiguous, REF[g["id"]])
            conf = {f.lower() for f in judge["fatos_presentes"]}
            extra = [f for f in r.facts_ambiguous if f.lower() in conf]
            r.coverage = (len(r.facts_present) + len(extra)) / max(1, len(facts))
            r.hallucination = bool(judge["alucinacao"])
        approved = (r.coverage >= 0.5 and not r.tautology and not r.hallucination
                    and not refused and not truncated)
        items = set(ITEM_IN_ANSWER.findall(answer))
        gi = set(g.get("gold_items") or [])
        return {
            "key": g["key"], "id": g["id"], "cell": g["cell"], "variant": g["variant"],
            "suite": g["suite"], "vocabulary_level": g.get("vocabulary_level"),
            "gold_kind": g.get("gold_kind", "norma"), "context_type": g["context_type"],
            "ctx_ntok": g["ctx_ntok"], "n_prompt_tokens": g["n_prompt_tokens"],
            "n_gen_tokens": g["n_gen_tokens"], "stopped_eos": g["stopped_eos"],
            "n_manual_ctx": g["n_manual_ctx"],
            "ctx_has_gold_span": g.get("ctx_has_gold_span", g.get("retrieval_hit")),
            "answer": answer, "raw": g.get("raw", ""), "coverage": round(r.coverage, 4),
            "n_facts": len(facts), "tautology": r.tautology, "hallucination": r.hallucination,
            "refused": refused, "truncated": truncated, "citation_present": r.citation_present,
            "cite_any_item": bool(items), "cite_item": (bool(items & gi) if gi else None),
            "has_gold_items": bool(gi), "approved": approved,
            "judge_note": (judge or {}).get("justificativa", ""),
        }

    res = parallel_map(worker, gens, max_workers=args.workers, checkpoint_path=ckpt,
                       key_fn=lambda g: g["key"], existing=existing)
    scored = [res[g["key"]] for g in gens if g["key"] in res and "__error__" not in res[g["key"]]]
    print(f"[ok] {len(scored)} pontuados, {len(gens) - len(scored)} erros")
    with (F6.DATA / "e2e" / f"scored_{args.tag}.jsonl").open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps({k: v for k, v in s.items() if k != "raw"}, ensure_ascii=False) + "\n")
    sm = summarize(scored, "ctrl_d50" if args.suite == "recorte" else "a_raw2")
    C.write_json(F6.DATA / "e2e" / f"summary_{args.tag}.json", sm)
    print_table(sm)


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
        "ctx_with_manual": round(100 * sum(1 for r in rs if r["n_manual_ctx"]) / n, 1),
    }


def sign_test_p(win: int, loss: int) -> float:
    """Teste do sinal bilateral exato (pareado por query)."""
    from math import comb
    n = win + loss
    if n == 0:
        return 1.0
    k = min(win, loss)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def summarize(scored, ctrl_cell: str):
    by = defaultdict(list)
    for s in scored:
        by[s["cell"]].append(s)
    out = {"prefill_tps_assumed": round(PREFILL_TPS, 1), "control_cell": ctrl_cell, "cells": {}}
    for cell, rs in sorted(by.items()):
        d = {"variant": rs[0]["variant"], "all": _agg(rs),
             "ctx_has_gold": _agg([r for r in rs if r["ctx_has_gold_span"]]),
             "ctx_no_gold": _agg([r for r in rs if not r["ctx_has_gold_span"]])}
        if rs[0]["suite"] == "recorte344":
            d["by_kind"] = {k: _agg([r for r in rs if r["gold_kind"] == k]) for k in ("norma", "manual")}
            d["by_vocab"] = {v: _agg([r for r in rs if r["vocabulary_level"] == v]) for v in C.VOCABS}
            d["by_kind_vocab"] = {k: {v: _agg([r for r in rs if r["gold_kind"] == k
                                              and r["vocabulary_level"] == v]) for v in C.VOCABS}
                                  for k in ("norma", "manual")}
        out["cells"][cell] = d
    ctrl = {r["id"]: r["approved"] for r in by.get(ctrl_cell, [])}
    for cell, rs in by.items():
        pv = {}
        for k in (None, "norma", "manual"):
            sel = [r for r in rs if k is None or r["gold_kind"] == k]
            win = sum(1 for r in sel if r["approved"] and not ctrl.get(r["id"], False))
            loss = sum(1 for r in sel if not r["approved"] and ctrl.get(r["id"], False))
            pv[k or "all"] = {"win": win, "loss": loss, "sign_p": sign_test_p(win, loss)}
        out["cells"][cell]["paired_vs_ctrl"] = pv
    return out


def print_table(sm):
    print(f"{'célula':<10} {'aprov':>6} {'norma':>6} {'manual':>6} {'aluc':>5} {'recusa':>6} "
          f"{'citeNR':>6} {'citeIt':>6} {'ctxTok':>7} {'prompt':>7} {'gold%':>6} {'w/l':>8} {'p':>6}")
    for cell, d in sm["cells"].items():
        a, p = d["all"], d["paired_vs_ctrl"]["all"]
        bk = d.get("by_kind", {})
        nm = bk.get("norma", {}).get("approved", float("nan"))
        mm = bk.get("manual", {}).get("approved", float("nan"))
        print(f"{cell:<10} {a['approved']:6.1f} {nm:6.1f} {mm:6.1f} {a['hallucination']:5.1f} "
              f"{a['refused']:6.1f} {a['cite_nr']:6.1f} {a['cite_gold_item']:6.1f} "
              f"{a['ctx_ntok']:7.1f} {a['prompt_ntok']:7.1f} {a['ctx_has_gold']:6.1f} "
              f"{p['win']:>3}/{p['loss']:<4} {p['sign_p']:6.3f}")


if __name__ == "__main__":
    main()
