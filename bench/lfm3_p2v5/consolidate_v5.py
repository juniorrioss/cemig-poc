#!/usr/bin/env python3
"""
consolidate_v5.py — bateria completa do v5 contra as DUAS referências do capitão
(f4-r32-ep2 base e v3s400 = modelo do app): 4 células ×1536 (inclui closed-book),
e2e op_d50 ×1536 (formato do app) e e2e ×151 coloquiais. Pareados (teste do sinal)
vs base E vs v3s400, recorte × fora do recorte, tokens médios (concisão) e o
CRITÉRIO DE PROMOÇÃO: v5 ≥ v3s400 em distrator, e2e e 151 (sem perda significativa
E sem ponto pior) E melhor em concisão ou em outra métrica relevante.

Lê os scored da avaliação (mesmos caminhos do bench/lfm3_p1p2/eval/score_p2.py):
  f2root/data/scored/scored_p2-{tag}1536_{cell}.jsonl
  f6root/data/e2e/scored_p2-{tag}-{1536,151}.jsonl

Uso: classifier/.venv/bin/python bench/lfm3_p2v5/consolidate_v5.py --tags base v3s400 v4s400 v5s400
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORTE = {10, 18, 34, 13, 31, 33}
CELLS = ("distrator", "ranking", "oraculo", "closed_book")


def rj(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def sign_p(w, l):
    n = w + l
    if not n:
        return 1.0
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4)


def pct(rs, k):
    return round(100 * sum(1 for r in rs if r.get(k)) / len(rs), 1) if rs else None


def mean_tok(rs):
    return round(sum(r["n_gen_tokens"] for r in rs) / len(rs), 1) if rs else None


def paired(a, b, key):
    """win = a tem key e b não (para alucinação: 'win' = a PIOR)."""
    bb = {r["id"]: bool(r.get(key)) for r in b}
    common = [r for r in a if r["id"] in bb]
    w = sum(1 for r in common if r.get(key) and not bb[r["id"]])
    l = sum(1 for r in common if not r.get(key) and bb[r["id"]])
    return {"a_yes_b_no": w, "a_no_b_yes": l, "sign_p": sign_p(w, l), "n": len(common)}


def nr_of(src):
    m = re.search(r"nr[-_ ]?(\d+)", (src or "").lower())
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/d/cemig-scratch/p1p2/eval")
    ap.add_argument("--tags", nargs="+", default=["base", "v3s400", "v4s400", "v5s400"])
    ap.add_argument("--candidate", default="v5s400")
    ap.add_argument("--refs", nargs="+", default=["base", "v3s400"])
    ap.add_argument("--out", default=str(HERE / "data" / "consolidation_full.json"))
    args = ap.parse_args()
    w = Path(args.work)
    par = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/"
                        "snapshots/*/normas_regulamentadoras/train-*.parquet"))[0]
    import pyarrow.parquet as pq
    src = {r["id"]: r["source_file"] for r in pq.read_table(par).to_pylist()}

    def load(tag):
        d = {c: rj(w / "f2root" / "data" / "scored" / f"scored_p2-{tag}1536_{c}.jsonl") for c in CELLS}
        d["e2e"] = rj(w / "f6root" / "data" / "e2e" / f"scored_p2-{tag}-1536.jsonl")
        d["poc151"] = rj(w / "f6root" / "data" / "e2e" / f"scored_p2-{tag}-151.jsonl")
        return d

    data = {t: load(t) for t in args.tags}
    res = {}
    for t, d in data.items():
        o = {}
        for c in CELLS:
            rs = d[c]
            if not rs:
                continue
            m = {"n": len(rs), "approved": pct(rs, "approved"), "gen_tok": mean_tok(rs),
                 "hallucination": pct(rs, "hallucination"), "refused": pct(rs, "refused")}
            if c == "distrator":
                m["refusal_correct"] = pct(rs, "refusal_correct")
                fora = [r for r in rs if nr_of(src[r["id"]]) not in RECORTE]
                m["fora_recorte"] = {"n": len(fora), "hallucination": pct(fora, "hallucination"),
                                     "refusal_correct": pct(fora, "refusal_correct")}
            if c in ("ranking", "oraculo"):
                m["refusal_undue"] = pct(rs, "refusal_undue")
            o[c] = m
        for k in ("e2e", "poc151"):
            rs = d[k]
            if not rs:
                continue
            gi = [r for r in rs if r.get("has_gold_items")]
            m = {"n": len(rs), "approved": pct(rs, "approved"), "hallucination": pct(rs, "hallucination"),
                 "refused": pct(rs, "refused"), "gen_tok": mean_tok(rs),
                 "cite_gold_item": round(100 * sum(1 for r in gi if r["cite_item"]) / len(gi), 1) if gi else None}
            if k == "e2e":
                fora = [r for r in rs if nr_of(src[r["id"]]) not in RECORTE]
                m["fora_recorte"] = {"n": len(fora), "approved": pct(fora, "approved"),
                                     "hallucination": pct(fora, "hallucination")}
            o[k] = m
        allrs = [r for c in CELLS for r in d[c]] + d["e2e"]
        o["gen_tok_mean_all"] = mean_tok(allrs)
        # pareados vs referências
        for ref in args.refs:
            if ref == t or ref not in data:
                continue
            pr = {}
            for c in list(CELLS) + ["e2e", "poc151"]:
                a, b = d.get(c), data[ref].get(c)
                if a and b:
                    pr[c] = {"approved": paired(a, b, "approved"),
                             "hallucination_worse_better": paired(a, b, "hallucination")}
            o[f"paired_vs_{ref}"] = pr
        res[t] = o

    # critério de promoção (candidato vs v3s400)
    cand, ref = res.get(args.candidate), res.get("v3s400")
    verdict = None
    if cand and ref and "paired_vs_v3s400" in cand:
        pv = cand["paired_vs_v3s400"]
        checks = {
            "distrator_halluc_le": cand["distrator"]["hallucination"] <= ref["distrator"]["hallucination"],
            "distrator_halluc_not_sig_worse": not (pv["distrator"]["hallucination_worse_better"]["sign_p"] < 0.05 and
                                                   pv["distrator"]["hallucination_worse_better"]["a_yes_b_no"] >
                                                   pv["distrator"]["hallucination_worse_better"]["a_no_b_yes"]),
            "e2e_approved_ge": cand["e2e"]["approved"] >= ref["e2e"]["approved"],
            "e2e_not_sig_worse": not (pv["e2e"]["approved"]["sign_p"] < 0.05 and
                                      pv["e2e"]["approved"]["a_no_b_yes"] > pv["e2e"]["approved"]["a_yes_b_no"]),
            "poc151_approved_ge": cand["poc151"]["approved"] >= ref["poc151"]["approved"],
            "poc151_not_sig_worse": not (pv["poc151"]["approved"]["sign_p"] < 0.05 and
                                         pv["poc151"]["approved"]["a_no_b_yes"] > pv["poc151"]["approved"]["a_yes_b_no"]),
            "concision_better": cand["e2e"]["gen_tok"] < ref["e2e"]["gen_tok"],
        }
        strict = all(v for k, v in checks.items() if k != "concision_better") and checks["concision_better"]
        verdict = {"checks": checks, "promote_strict": strict}
    res["_promotion"] = verdict
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    f = lambda x: f"{x:6.1f}" if isinstance(x, (int, float)) else "     -"  # noqa: E731
    print(f"{'tag':<8}{'dist-h':>7}{'d-rc':>6}{'rank':>6}{'orac':>6}{'closd':>6}{'e2e':>6}{'e2eH':>6}{'cite':>6}"
          f"{'151':>6}{'151H':>6}{'151R':>6}{'tokE2E':>7}{'tokAll':>7}")
    for t, o in res.items():
        if t.startswith("_"):
            continue
        g = lambda c, k: o.get(c, {}).get(k)  # noqa: E731
        print(f"{t:<8}{f(g('distrator','hallucination'))} {f(g('distrator','refusal_correct'))[1:]}"
              f"{f(g('ranking','approved'))}{f(g('oraculo','approved'))}{f(g('closed_book','approved'))}"
              f"{f(g('e2e','approved'))}{f(g('e2e','hallucination'))}{f(g('e2e','cite_gold_item'))}"
              f"{f(g('poc151','approved'))}{f(g('poc151','hallucination'))}{f(g('poc151','refused'))}"
              f"{f(g('e2e','gen_tok'))} {f(o.get('gen_tok_mean_all'))}")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
