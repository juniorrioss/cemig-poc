#!/usr/bin/env python3
"""
score_smoke.py — régua honesta (= score_f6.py, juiz 27B) no smoke de 20 perguntas
operacionais do recorte (célula op_d50, prompts IDÊNTICOS aos da F6) gerado nos GGUF
bf16 e Q4_0 do f4-r32-ep2 (llama.cpp v0.9.0, mesmo seed 42). Referência: as 3 rodadas
HF/transformers da F6 para os mesmos ids.

Uso: classifier/.venv/bin/python bench/lfm3_f7/score_smoke.py
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


def clean(raw: str) -> str:
    m = re.match(r"\s*<think>(.*?)</think>", raw or "", re.DOTALL)
    t = raw[m.end():] if m else raw
    return t.replace("<|im_end|>", "").strip()


def main() -> None:
    Q, FACTS, REF = load_refs("recorte")
    gens = [json.loads(l) for l in (D / "smoke20_gen.jsonl").read_text().splitlines() if l.strip()]
    jobs = [(g, v) for g in gens for v in ("bf16", "q4_0")]

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
    # referência HF (F6, 3 rodadas) dos mesmos ids
    ids = {g["id"] for g in gens}
    hf = {}
    for f in ("scored_f6-recorte.jsonl", "scored_f6-recorte-rep2.jsonl", "scored_f6-recorte-rep3.jsonl"):
        for l in (REPO / "bench/lfm3_f6/data/e2e" / f).read_text().splitlines():
            d = json.loads(l)
            if d["cell"] == "op_d50" and d["id"] in ids:
                hf.setdefault(d["id"], []).append(d["approved"])
    summ = {}
    for v in ("bf16", "q4_0"):
        rs = [r for r in rows if r["variant"] == v]
        n = len(rs)
        summ[v] = {"n": n, "approved": sum(r["approved"] for r in rs),
                   "coverage": round(sum(r["coverage"] for r in rs) / n, 3),
                   "hallucination": sum(r["hallucination"] for r in rs),
                   "refused": sum(r["refused"] for r in rs), "tool_echo": sum(r["tool_echo"] for r in rs),
                   "tok_parity": sum(r["tok_parity"] for r in rs)}
    hf_mean = sum(sum(v) / len(v) for v in hf.values())
    summ["hf_f6_mean_of_3"] = {"n": len(hf), "approved_expected": round(hf_mean, 2)}
    by = {r["key"]: r for r in rows}
    pair = {"q4_win": 0, "q4_loss": 0}
    for i in ids:
        a, b = by[f"{i}::bf16"]["approved"], by[f"{i}::q4_0"]["approved"]
        pair["q4_win"] += (b and not a)
        pair["q4_loss"] += (a and not b)
    summ["paired_q4_vs_bf16"] = pair
    C.write_json(D / "summary_smoke20.json", summ)
    with (D / "scored_smoke20.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["key"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
