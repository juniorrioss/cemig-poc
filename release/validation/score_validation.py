#!/usr/bin/env python3
"""
score_validation.py — confere a replicação do zero (run_from_scratch.sh) contra o pipeline
do repo, nas 10 perguntas operacionais:

  1. identidade textual: respostas llama (release, seed 42) × smoke Q4_0 da F7
     (llama-server, mesmo GGUF, mesmo prompt, seed 42); rodada 1 × rodada 2 (determinismo);
  2. régua honesta (= bench/lfm3_f7/score_smoke.py / score_f6.py, juiz 27B) nas respostas
     llama e hf da release × as 3 rodadas HF/transformers da F6 (op_d50) para os mesmos ids.

Ferramenta de VALIDAÇÃO do repo (importa bench/; o inference.py não).
Uso: classifier/.venv/bin/python release/validation/score_validation.py --out-dir <scratch/out>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "bench" / "lfm3_f6", REPO / "bench" / "lfm3_f5", REPO / "bench" / "lfm3_f0",
          REPO / "bench" / "regua"):
    sys.path.insert(0, str(p))

from common import parallel_map  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402
from ruler import deterministic_pass  # noqa: E402
from score_f6 import load_refs  # noqa: E402


def clean(raw: str) -> str:
    m = re.match(r"\s*<think>(.*?)</think>", raw or "", re.DOTALL)
    t = raw[m.end():] if m else (raw or "")
    return t.replace("<|im_end|>", "").strip()


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke-ref", default=None, help="smoke20_gen.jsonl da F7 (default: git)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    qs = read_jsonl(HERE / "questions10.jsonl")
    ids = [q["id"] for q in qs]
    kind = {q["id"]: q["gold_kind"] for q in qs}

    runs = {}
    for name in ("llama_seed42_r1", "llama_seed42_r2", "hf_seed42_r1", "hf_seed42_r2"):
        p = out / f"{name}.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            qid = {q["question"]: q["id"] for q in qs}
            runs[name] = {qid[r["question"]]: clean(r["raw"]) for r in rows}
    # referências do repo: F6 HF (3 rodadas) e smoke Q4 da F7
    f6 = {}
    for tag in ("f6-recorte", "f6-recorte-rep2", "f6-recorte-rep3"):
        for r in read_jsonl(REPO / "bench/lfm3_f6/data/e2e" / f"gen_{tag}.jsonl"):
            if r["cell"] == "op_d50" and r["id"] in ids:
                f6.setdefault(tag, {})[r["id"]] = clean(r["raw"])
    if a.smoke_ref:
        smoke_txt = Path(a.smoke_ref).read_text(encoding="utf-8")
    else:
        smoke_txt = subprocess.check_output(
            ["git", "-C", str(REPO), "show",
             "fm/lfm3-f7-embarque:bench/lfm3_f7/data/smoke/smoke20_gen.jsonl"], text=True)
    smoke = {json.loads(l)["id"]: clean(json.loads(l)["q4_0"]["answer_raw"])
             for l in smoke_txt.splitlines() if l.strip()}

    ident = {}
    if "llama_seed42_r1" in runs:
        r1 = runs["llama_seed42_r1"]
        common = [i for i in ids if i in smoke]
        ident["llama_r1_vs_f7_smoke_q4"] = {"identical": sum(r1[i] == smoke[i] for i in common),
                                            "n": len(common)}
        for other in ("llama_seed42_r2",):
            if other in runs:
                ident[f"llama_r1_vs_{other}"] = {
                    "identical": sum(r1[i] == runs[other][i] for i in ids), "n": len(ids)}
    if "hf_seed42_r1" in runs and "hf_seed42_r2" in runs:
        ident["hf_r1_vs_hf_r2"] = {"identical": sum(runs["hf_seed42_r1"][i] == runs["hf_seed42_r2"][i]
                                                    for i in ids), "n": len(ids)}

    Q, FACTS, REF = load_refs("recorte")
    cells = {**{k: v for k, v in runs.items()}, **{f"repo_{k}": v for k, v in f6.items()},
             "repo_f7_smoke_q4": {i: smoke[i] for i in ids if i in smoke}}
    jobs = [(c, i, ans) for c, d in cells.items() for i, ans in d.items()]

    def worker(job):
        c, i, answer = job
        answer = answer.replace("\u2011", "-").replace("\u2013", "-")
        facts = FACTS[i]
        r = deterministic_pass(Q[i], answer, facts)
        refused = looks_like_refusal(answer)
        tool_echo = "<tool_call>" in answer
        if answer.strip() and not r.empty:
            j = judge_disambiguate(Q[i], answer, r.facts_ambiguous, REF[i])
            conf = {f.lower() for f in j["fatos_presentes"]}
            extra = [f for f in r.facts_ambiguous if f.lower() in conf]
            r.coverage = (len(r.facts_present) + len(extra)) / max(1, len(facts))
            r.hallucination = bool(j["alucinacao"])
        appr = (r.coverage >= 0.5 and not r.tautology and not r.hallucination and not refused
                and not tool_echo)
        return {"cell": c, "id": i, "gold_kind": kind[i], "approved": appr,
                "coverage": round(r.coverage, 3), "hallucination": r.hallucination,
                "refusal": refused, "answer": answer}

    res = parallel_map(worker, jobs, max_workers=8, key_fn=lambda j: f"{j[0]}::{j[1]}")
    errs = [v for v in res.values() if "__error__" in v]
    assert not errs, errs[:2]
    scored = list(res.values())
    summary = {}
    for c in cells:
        rs = [s for s in scored if s["cell"] == c]
        summary[c] = {"n": len(rs), "approved": sum(s["approved"] for s in rs),
                      "hallucination": sum(bool(s["hallucination"]) for s in rs),
                      "refusal": sum(s["refusal"] for s in rs),
                      "coverage_mean": round(sum(s["coverage"] for s in rs) / max(1, len(rs)), 3)}
    rep = {"identity": ident, "regua": summary,
           "per_question": {i: {c: next((s["approved"] for s in scored if s["cell"] == c and s["id"] == i), None)
                                for c in cells} for i in ids}}
    (HERE / "validation_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    (HERE / "scored_validation.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scored) + "\n", encoding="utf-8")
    print(json.dumps({"identity": ident, "regua": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
