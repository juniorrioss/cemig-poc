#!/usr/bin/env python3
"""
score_smoke.py — régua honesta (= bench/lfm3_f7/score_smoke.py, juiz 27B) no smoke de 20
operacionais: Q4_K_M novo (data/smoke/smoke20_q4km_gen.jsonl, gerado na Spark com o protocolo
da F7) + rejulgamento, no MESMO lote, das respostas Q4_0 e bf16 da F7
(data/smoke/f7_smoke20_gen.jsonl, cópia de bench/lfm3_f7/data/smoke/) como controle do juiz.

Uso: classifier/.venv/bin/python bench/lfm3_q4km/score_smoke.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "bench" / "lfm3_f6", REPO / "bench" / "lfm3_f5", REPO / "bench" / "lfm3_f0",
          REPO / "bench" / "regua"):
    sys.path.insert(0, str(p))

import common_f5 as C  # noqa: E402
from common import parallel_map  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402
from ruler import deterministic_pass  # noqa: E402
from score_f6 import load_refs  # noqa: E402

D = HERE / "data" / "smoke"
VARIANTS = ("q4_k_m", "q4_0", "bf16")


def clean(raw: str) -> str:
    m = re.match(r"\s*<think>(.*?)</think>", raw or "", re.DOTALL)
    t = raw[m.end():] if m else raw
    return t.replace("<|im_end|>", "").strip()


def main() -> None:
    Q, FACTS, REF = load_refs("recorte")
    new = {g["id"]: g for g in map(json.loads, (D / "smoke20_q4km_gen.jsonl").read_text().splitlines())}
    old = {g["id"]: g for g in map(json.loads, (D / "f7_smoke20_gen.jsonl").read_text().splitlines())}
    assert set(new) == set(old) and len(new) == 20
    gens = {i: {**old[i], "q4_k_m": new[i]["q4_k_m"]} for i in new}
    jobs = [(gens[i], v) for i in sorted(gens) for v in VARIANTS]

    def worker(job):
        g, v = job
        answer = clean(g[v]["answer_raw"]).replace("\u2011", "-").replace("\u2013", "-")
        facts = FACTS[g["id"]]
        r = deterministic_pass(Q[g["id"]], answer, facts)
        refused = looks_like_refusal(answer)
        tool_echo = "<tool_call>" in answer
        if answer.strip() and not r.empty:
            j = judge_disambiguate(Q[g["id"]], answer, r.facts_ambiguous, REF[g["id"]])
            conf = {f.lower() for f in j["fatos_presentes"]}
            extra = [f for f in r.facts_ambiguous if f.lower() in conf]
            r.coverage = (len(r.facts_present) + len(extra)) / max(1, len(facts))
            r.hallucination = bool(j["alucinacao"])
        appr = (r.coverage >= 0.5 and not r.tautology and not r.hallucination and not refused
                and not tool_echo)
        return {"key": f"{g['id']}::{v}", "id": g["id"], "variant": v, "gold_kind": g["gold_kind"],
                "answer": answer, "coverage": round(r.coverage, 3), "hallucination": r.hallucination,
                "refused": refused, "tool_echo": tool_echo, "approved": appr,
                "tok_parity": g[v]["tok_parity"], "n_gen": g[v]["n_gen"]}

    res = parallel_map(worker, jobs, max_workers=8, checkpoint_path=D / "scored.ckpt.json",
                       key_fn=lambda j: f"{j[0]['id']}::{j[1]}", existing={})
    rows = list(res.values())
    hist = {d["key"]: d for d in map(json.loads, (D / "f7_scored_smoke20_ref.jsonl").read_text().splitlines())}
    summ = {}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        n = len(rs)
        summ[v] = {"n": n, "approved": sum(r["approved"] for r in rs),
                   "approved_norma": sum(r["approved"] for r in rs if r["gold_kind"] != "manual"),
                   "approved_manual": sum(r["approved"] for r in rs if r["gold_kind"] == "manual"),
                   "coverage": round(sum(r["coverage"] for r in rs) / n, 3),
                   "hallucination": sum(r["hallucination"] for r in rs),
                   "refused": sum(r["refused"] for r in rs), "tool_echo": sum(r["tool_echo"] for r in rs),
                   "tok_parity": sum(r["tok_parity"] for r in rs),
                   "mean_n_gen": round(sum(r["n_gen"] or 0 for r in rs) / n, 1)}
    for v in ("q4_0", "bf16"):
        summ[v]["approved_f7_historico"] = sum(hist[f"{i}::{v}"]["approved"] for i in gens)
        summ[v]["juiz_concorda_f7"] = sum(hist[f"{i}::{v}"]["approved"] == res[f"{i}::{v}"]["approved"] for i in gens)
    by = {r["key"]: r for r in rows}
    for ref in ("q4_0", "bf16"):
        w = sum(by[f"{i}::q4_k_m"]["approved"] and not by[f"{i}::{ref}"]["approved"] for i in gens)
        l = sum(by[f"{i}::{ref}"]["approved"] and not by[f"{i}::q4_k_m"]["approved"] for i in gens)
        summ[f"paired_q4_k_m_vs_{ref}"] = {"q4km_win": w, "q4km_loss": l}
    C.write_json(D / "summary_smoke20.json", summ)
    with (D / "scored_smoke20.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["key"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
