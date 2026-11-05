#!/usr/bin/env python3
"""device_parity.py — compara top-5 e CONTEXTO COMPRIMIDO entregues no aparelho (f7_e2e_results*.json)
com o espelho Python da F6 (prompts_recorte.jsonl, célula op_d50) para as perguntas-exemplo.
Uso: classifier/.venv/bin/python bench/lfm3_f7/device_parity.py <f7_e2e_results.json>"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
P = {}
for l in (REPO / "bench/lfm3_f6/data/e2e/prompts_recorte.jsonl").open():
    d = json.loads(l)
    if d["cell"] == "op_d50":
        P[d["id"]] = d
dev = json.load(open(sys.argv[1]))
ok_ids = ok_ctx = n = 0
rows = []
for c in dev["cases"]:
    if c["id"] not in P:
        continue
    n += 1
    ref = P[c["id"]]
    dids = [int(x.split()[0]) for x in c.get("chunks", [])]
    same_ids = dids == ref["ctx_ids"]
    same_ctx = c.get("tool_context") == ref["messages"][-1]["content"]
    ok_ids += same_ids; ok_ctx += same_ctx
    rows.append({"id": c["id"], "device_top5": dids, "mirror_top5": ref["ctx_ids"], "ids_equal": same_ids, "context_equal": same_ctx})
    print(c["id"], same_ids, same_ctx, dids, ref["ctx_ids"])
out = {"n": n, "top5_equal": ok_ids, "context_equal": ok_ctx, "cases": rows}
Path(sys.argv[1]).with_suffix(".parity.json").write_text(json.dumps(out, indent=1))
print(f"top5 {ok_ids}/{n}  contexto {ok_ctx}/{n}")
