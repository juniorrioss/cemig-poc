#!/usr/bin/env python3
"""score_device.py — régua honesta (= score_f6, juiz 27B, gabarito 384) nas respostas geradas NO
APARELHO pelas perguntas-exemplo (texto e voz). Uso: classifier/.venv/bin/python
bench/lfm3_f7/score_device.py data/device/f7_e2e_text_earlystop.json [...]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in ("bench/lfm3_f6", "bench/lfm3_f5", "bench/lfm3_f0", "bench/regua"):
    sys.path.insert(0, str(REPO / p))
from common_f0 import looks_like_refusal  # noqa
from judge_regua import judge_disambiguate  # noqa
from ruler import deterministic_pass  # noqa
from score_f6 import load_refs  # noqa
Q, FACTS, REF = load_refs("recorte")
summary = {}
for f in sys.argv[1:]:
    d = json.load(open(f))
    rows = []
    for c in d["cases"]:
        if c["id"] not in FACTS:
            continue
        a = (c.get("final_answer") or "").replace("\u2011", "-").replace("\u2013", "-")
        r = deterministic_pass(Q[c["id"]], a, FACTS[c["id"]])
        if a.strip() and not r.empty:
            j = judge_disambiguate(Q[c["id"]], a, r.facts_ambiguous, REF[c["id"]])
            conf = {x.lower() for x in j["fatos_presentes"]}
            r.coverage = (len(r.facts_present) + sum(1 for x in r.facts_ambiguous if x.lower() in conf)) / max(1, len(FACTS[c["id"]]))
            r.hallucination = bool(j["alucinacao"])
        ref = looks_like_refusal(a)
        ok = r.coverage >= 0.5 and not r.tautology and not r.hallucination and not ref
        rows.append({"id": c["id"], "kind": c["kind"], "approved": ok, "coverage": round(r.coverage, 3),
                     "hallucination": r.hallucination, "refused": ref, "cites_nr": r.citation_present})
    summary[Path(f).name] = {"n": len(rows), "approved": sum(x["approved"] for x in rows),
                             "cites_nr": sum(x["cites_nr"] for x in rows), "rows": rows}
    print(Path(f).name, summary[Path(f).name]["approved"], "/", len(rows), "cita NR", summary[Path(f).name]["cites_nr"])
    for x in rows: print("  ", x)
(HERE / "data/device/scored_device.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
