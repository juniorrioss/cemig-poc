#!/usr/bin/env python3
"""
consolidate_f6.py — tabela-mestra da F6 -> data/consolidation.json + stdout.

(1) Pareto de retrieval (data/retrieval/*.json): todas as variantes com manuais, Δ vs o
    controle SEM manuais da mesma configuração de gate (v4app ou thr0.7_f2) e vs v4app (app).
    Trava do brief: norma hit@5 >= −0,5 p.p. e 151 R@2 intacto (Δ=0).
(2) Fim-a-fim (data/e2e/summary_*.json): rodada 1 + réplica, média por célula/recorte.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402
import common_f5 as C  # noqa: E402

OP = "thr0.7_f2+man1.75+margin"


def load(name):
    p = F6.DATA / "retrieval" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def retrieval_rows():
    app = load("v4app")
    rows = []
    for p in sorted((F6.DATA / "retrieval").glob("*.json")):
        a = json.loads(p.read_text())
        name = a["variant"]
        base = name.split("+")[0]
        ctrl = load(base)
        P, N, M, A = a["poc151"], a["bench_norma"], a["bench_manual"], a["bench_all"]
        cp, cn = ctrl["poc151"], ctrl["bench_norma"]
        d_norm5 = N["hit5"] - cn["hit5"]
        d_r2 = P["r2"] - cp["r2"]
        rows.append({
            "variant": name, "base": base, "manual_w": a["cfg"]["manual_w"],
            "manual_gate": a["cfg"]["manual_gate"], "manual_margin": a["cfg"]["manual_margin"],
            "151_r2": P["r2"], "151_r5": P["r5"], "151_mrr": P["mrr"],
            "norma_hit2": N["hit2"], "norma_hit5": N["hit5"], "norma_evrec2": N["evrec2"],
            "manual_hit2": M["hit2"], "manual_hit5": M["hit5"], "manual_chit5": M["chit5"],
            "all_hit5": A["hit5"],
            "d_norma_hit5_vs_base": round(d_norm5, 4), "d_151_r2_vs_base": round(d_r2, 4),
            "d_norma_hit5_vs_v4app": round(N["hit5"] - app["bench_norma"]["hit5"], 4),
            "d_151_r2_vs_v4app": round(P["r2"] - app["poc151"]["r2"], 4),
            "trava_ok": (d_norm5 >= -F6.TOL - 1e-9) and abs(d_r2) < 1e-9,
            "chosen": name == OP,
        })
    return rows


KEYS = ("approved", "hallucination", "refused", "cite_nr", "cite_gold_item", "coverage",
        "ctx_ntok", "prompt_ntok", "ttft_est_s", "ctx_has_gold")


def _mean(ds):
    ds = [d for d in ds if d]
    return {k: round(sum(d[k] for d in ds) / len(ds), 3) for k in KEYS if k in ds[0]} | {
        "n": ds[0]["n"], "n_runs": len(ds)}


def e2e():
    """Média das rodadas disponíveis por célula (rep3 só tem ctrl_d50/op_d50)."""
    out = {}
    for suite in ("recorte", "151"):
        runs = [F6.DATA / "e2e" / f"summary_f6-{suite}{sfx}.json" for sfx in ("", "-rep2", "-rep3")]
        runs = [json.loads(p.read_text()) for p in runs if p.exists()]
        if not runs:
            continue
        cells = {}
        for cell in runs[0]["cells"]:
            ds = [r["cells"][cell] for r in runs if cell in r["cells"]]
            c = {"runs": ds, "n_runs": len(ds), "mean_all": _mean([d["all"] for d in ds])}
            if "by_kind" in ds[0]:
                c["mean_by_kind"] = {k: _mean([d["by_kind"][k] for d in ds]) for k in ds[0]["by_kind"]}
                c["mean_by_vocab"] = {v: _mean([d["by_vocab"][v] for d in ds]) for v in ds[0]["by_vocab"]}
                c["mean_by_kind_vocab"] = {
                    k: {v: _mean([d["by_kind_vocab"][k][v] for d in ds]) for v in ds[0]["by_kind_vocab"][k]}
                    for k in ds[0]["by_kind_vocab"]}
            cells[cell] = c
        out[suite] = {"n_runs_max": len(runs), "cells": cells}
    return out


def causal_norma():
    """Δ de aprovação em NORMA atribuível ao contexto: só as queries cujo turno tool MUDOU
    vs o controle ctrl_d50 (nas de contexto idêntico a diferença é só amostragem; lá o
    efeito real é zero por construção). Agregado sobre as rodadas com as duas células."""
    sys.path.insert(0, str(F6.HERE))
    from score_f6 import sign_test_p
    ctx = {p["key"]: p["context_given"] for p in C.read_jsonl(F6.DATA / "e2e" / "prompts_recorte.jsonl")}
    out = {}
    runs = [C.read_jsonl(F6.DATA / "e2e" / f"scored_f6-recorte{sfx}.jsonl") for sfx in ("", "-rep2", "-rep3")]
    runs = [{(s["id"], s["cell"]): s for s in r} for r in runs if r]
    for cell in ("op_d50", "mid_d50", "hi_d50"):
        acc = {"ident": [0, 0, 0], "diff": [0, 0, 0]}
        nr = 0
        for R in runs:
            if not any(c == cell for _, c in R):
                continue
            nr += 1
            for (i, c), s in R.items():
                if c != "ctrl_d50" or s["gold_kind"] != "norma":
                    continue
                o = R[(i, cell)]
                g = "ident" if ctx[f"{i}::ctrl_d50"] == ctx[f"{i}::{cell}"] else "diff"
                acc[g][0] += 1
                acc[g][1] += o["approved"] and not s["approved"]
                acc[g][2] += s["approved"] and not o["approved"]
        n_norma = (acc["ident"][0] + acc["diff"][0]) or 1
        out[cell] = {"n_runs": nr, "n_norma_per_run": n_norma // max(1, nr),
                     "n_ctx_changed_per_run": acc["diff"][0] // max(1, nr),
                     "ctx_changed": {"win": acc["diff"][1], "loss": acc["diff"][2],
                                     "sign_p": sign_test_p(acc["diff"][1], acc["diff"][2])},
                     "ctx_identical_noise": {"win": acc["ident"][1], "loss": acc["ident"][2],
                                             "sign_p": sign_test_p(acc["ident"][1], acc["ident"][2])},
                     "d_norma_causal_pp": round(100 * (acc["diff"][1] - acc["diff"][2]) / n_norma, 2),
                     "d_norma_observed_pp": round(100 * (acc["diff"][1] + acc["ident"][1]
                                                         - acc["diff"][2] - acc["ident"][2]) / n_norma, 2)}
    return out


def main() -> None:
    rows = retrieval_rows()
    out = {"chosen_operating_point": OP, "tolerance": F6.TOL, "retrieval": rows, "e2e": e2e(),
           "e2e_norma_causal": causal_norma()}
    C.write_json(F6.DATA / "consolidation.json", out)
    print("== retrieval (trava: norma hit@5 >= -0,5pp vs base sem manuais; 151 R@2 =)")
    for r in sorted(rows, key=lambda r: (r["base"], r["manual_hit5"])):
        print(f"  {r['variant']:<34} 151 R@2 {r['151_r2']*100:5.1f} R@5 {r['151_r5']*100:5.1f} | "
              f"norma h@5 {r['norma_hit5']:.4f} (Δ {r['d_norma_hit5_vs_base']*100:+.2f}) | "
              f"man h@5 {r['manual_hit5']:.3f} | all h@5 {r['all_hit5']:.3f} | "
              f"{'OK' if r['trava_ok'] else '--'}{' <== escolhido' if r['chosen'] else ''}")
    for suite, s in out["e2e"].items():
        print(f"== e2e {suite} (média das rodadas disponíveis)")
        for cell, c in s["cells"].items():
            m = c["mean_all"]
            kinds = c.get("mean_by_kind", {})
            ks = " ".join(f"{k} {v['approved']:.1f}" for k, v in kinds.items())
            print(f"  {cell:<10} [{c['n_runs']}x] aprov {m['approved']:.1f} {ks} | aluc {m['hallucination']:.1f} "
                  f"recusa {m['refused']:.1f} citeIt {m['cite_gold_item']:.1f} tok {m['ctx_ntok']:.0f}/"
                  f"{m['prompt_ntok']:.0f} gold {m['ctx_has_gold']:.1f}")
    print("== Δ norma causal (só queries com contexto alterado) vs observado")
    for cell, d in out["e2e_norma_causal"].items():
        print(f"  {cell:<8} [{d['n_runs']}x] ctx alterado {d['n_ctx_changed_per_run']}/{d['n_norma_per_run']} "
              f"w/l {d['ctx_changed']['win']}/{d['ctx_changed']['loss']} -> causal {d['d_norma_causal_pp']:+.2f} p.p. "
              f"| observado {d['d_norma_observed_pp']:+.2f} (ruído em ctx idêntico w/l "
              f"{d['ctx_identical_noise']['win']}/{d['ctx_identical_noise']['loss']} p={d['ctx_identical_noise']['sign_p']})")
    print(f"[ok] {F6.DATA / 'consolidation.json'}")


if __name__ == "__main__":
    main()
