#!/usr/bin/env python3
"""
consolidate_p2.py — tabela-mestre P2: tags × métricas, com Δ pareado vs a base.

Métricas (recorte 344, mesmo protocolo da série LFM3):
  ranking   : aprovação, recusa-indevida
  distrator : ALUCINAÇÃO (=não recusou e o juiz marcou alucinação OU respondeu sem
              recusar — como no F3/F4: hallucination_pct = % alucinação do juiz entre
              todas), recusa-correta
  oraculo / closed_book : aprovação
  e2e op_d50 (app): aprovação, cita item-ouro, alucinação, recusa
  probe     : qualidade geral / comportamental
  tokens médios de resposta (todas as células) — concisão antes/depois
  teste do sinal pareado vs base por célula (aprovação)

Uso: classifier/.venv/bin/python bench/lfm3_p1p2/eval/consolidate_p2.py --tags p2-base p2-s400 ...
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ("ranking", "distrator", "oraculo", "closed_book")


def rj(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def sign_p(w: int, l: int) -> float:
    n = w + l
    if not n:
        return 1.0
    k = min(w, l)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def pct(rs, k):
    return round(100 * sum(1 for r in rs if r.get(k)) / max(1, len(rs)), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/d/cemig-scratch/p1p2/eval")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--base", default="p2-base")
    ap.add_argument("--out", default=str(HERE.parent / "data" / "consolidation.json"))
    args = ap.parse_args()
    w = Path(args.work)
    res = {}
    base_rows = {}
    for tag in [args.base] + [t for t in args.tags if t != args.base]:
        d = {}
        allgen = []
        for c in CELLS:
            rs = rj(w / "f2root" / "data" / "scored" / f"scored_{tag}_{c}.jsonl")
            if not rs:
                continue
            allgen += [r["n_gen_tokens"] for r in rs]
            m = {"n": len(rs), "approved": pct(rs, "approved"),
                 "gen_tok": round(sum(r["n_gen_tokens"] for r in rs) / len(rs), 1)}
            if c == "ranking":
                m["refusal_undue"] = pct(rs, "refusal_undue")
                m["cite_nr"] = pct(rs, "citation_present")
            if c == "distrator":
                m["hallucination"] = pct(rs, "hallucination")
                m["refusal_correct"] = pct(rs, "refusal_correct")
                m["answered_anyway"] = round(100 - m["refusal_correct"], 1)
            if tag == args.base:
                base_rows[c] = {r["id"]: r["approved"] for r in rs}
            elif c in base_rows:
                b = base_rows[c]
                win = sum(1 for r in rs if r["approved"] and not b.get(r["id"], False))
                loss = sum(1 for r in rs if not r["approved"] and b.get(r["id"], False))
                m["paired_vs_base"] = {"win": win, "loss": loss, "sign_p": sign_p(win, loss)}
            d[c] = m
        e2e = rj(w / "f6root" / "data" / "e2e" / f"scored_{tag}.jsonl")
        if e2e:
            gi = [r for r in e2e if r["has_gold_items"]]
            m = {"n": len(e2e), "approved": pct(e2e, "approved"),
                 "hallucination": pct(e2e, "hallucination"), "refused": pct(e2e, "refused"),
                 "cite_gold_item": round(100 * sum(1 for r in gi if r["cite_item"]) / max(1, len(gi)), 1),
                 "cite_nr": pct(e2e, "citation_present"),
                 "gen_tok": round(sum(r["n_gen_tokens"] for r in e2e) / len(e2e), 1)}
            allgen += [r["n_gen_tokens"] for r in e2e]
            if tag == args.base:
                base_rows["e2e"] = {r["id"]: r["approved"] for r in e2e}
            elif "e2e" in base_rows:
                b = base_rows["e2e"]
                win = sum(1 for r in e2e if r["approved"] and not b.get(r["id"], False))
                loss = sum(1 for r in e2e if not r["approved"] and b.get(r["id"], False))
                m["paired_vs_base"] = {"win": win, "loss": loss, "sign_p": sign_p(win, loss)}
            d["e2e_op_d50"] = m
        pr = rj(w / "probe" / f"probe_scored_{tag}.jsonl")
        if pr:
            g = [r for r in pr if r["suite"] == "general"]
            bh = [r for r in pr if r["suite"] == "behavioral"]
            d["probe"] = {"general": round(sum(r["qualidade"] for r in g) / max(1, len(g)), 2),
                          "behavioral": round(sum(r["qualidade"] for r in bh) / max(1, len(bh)), 2)}
        if allgen:
            d["gen_tok_mean_all"] = round(sum(allgen) / len(allgen), 1)
        res[tag] = d
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    hdr = (f"{'tag':<16}{'rank':>6}{'r-ind':>6}{'dist-h':>7}{'dist-rc':>8}{'orac':>6}{'closed':>7}"
           f"{'e2e':>6}{'e2eCit':>7}{'e2eH':>6}{'probe':>6}{'tok':>6}")
    print(hdr)
    for t, d in res.items():
        g = lambda c, k: d.get(c, {}).get(k, float("nan"))  # noqa: E731
        print(f"{t:<16}{g('ranking','approved'):6.1f}{g('ranking','refusal_undue'):6.1f}"
              f"{g('distrator','hallucination'):7.1f}{g('distrator','refusal_correct'):8.1f}"
              f"{g('oraculo','approved'):6.1f}{g('closed_book','approved'):7.1f}"
              f"{g('e2e_op_d50','approved'):6.1f}{g('e2e_op_d50','cite_gold_item'):7.1f}"
              f"{g('e2e_op_d50','hallucination'):6.1f}{d.get('probe',{}).get('general',float('nan')):6.2f}"
              f"{d.get('gen_tok_mean_all',float('nan')):6.1f}")


if __name__ == "__main__":
    main()
