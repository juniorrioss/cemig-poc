#!/usr/bin/env python3
"""consolidate.py — tabela Q4_0 × Q4_K_M: llama-bench S24+ (3 rodadas), E2E 8 perguntas (mesma
sessão), smoke 20 e gate de decode (regressão > 10% → para). Saída: data/consolidation.json.
Uso: python3 bench/lfm3_q4km/consolidate.py"""
import json, statistics as st
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
out = {"bench": {}, "e2e": {}, "smoke20": {}, "device8": {}}
for q in ("Q4_0", "Q4_K_M"):
    pp, tg = [], []
    for r in (1, 2, 3):
        for row in json.loads((D / f"device/bench_{q}_r{r}.json").read_text()):
            (pp if row["n_prompt"] else tg).append(row["avg_ts"])
    out["bench"][q] = {"prefill_tps": [round(x, 1) for x in pp], "decode_tps": [round(x, 1) for x in tg],
                       "prefill_mean": round(st.mean(pp), 1), "prefill_sd": round(st.stdev(pp), 1),
                       "decode_mean": round(st.mean(tg), 2), "decode_sd": round(st.stdev(tg), 2)}
b = out["bench"]
dec = b["Q4_K_M"]["decode_mean"] / b["Q4_0"]["decode_mean"] - 1
out["gate"] = {"decode_delta_pct": round(100 * dec, 1),
               "prefill_delta_pct": round(100 * (b["Q4_K_M"]["prefill_mean"] / b["Q4_0"]["prefill_mean"] - 1), 1),
               "limite_pct": -10.0, "passou": dec >= -0.10}
for tag, q in (("q40", "Q4_0"), ("q4km", "Q4_K_M")):
    d = json.loads((D / f"device/e2e/f7_e2e_results_{tag}.json").read_text())
    tool = [c for c in d["cases"] if c.get("called_tool")]
    m = lambda k: round(st.mean(c[k] for c in tool))
    out["e2e"][q] = {"model": d["model"], "peak_pss_mb": round(d["peak_pss_mb"]), "n": len(tool),
                     "decision_ms": m("decision_ms"), "search_ms": m("search_ms"), "ttft_synth_ms": m("ttft_ms"),
                     "first_visible_token_ms": m("first_visible_token_ms"), "decode_ms": m("decode_ms"),
                     "total_ms": m("total_ms"), "tok_per_s": round(st.mean(c["tok_per_s"] for c in tool), 1)}
sm = json.loads((D / "smoke/summary_smoke20.json").read_text())
out["smoke20"] = {"Q4_0": sm["q4_0"]["approved"], "Q4_K_M": sm["q4_k_m"]["approved"], "bf16": sm["bf16"]["approved"],
                  "paired_q4km_vs_q40": sm["paired_q4_k_m_vs_q4_0"]}
dv = json.loads((D / "device/e2e/scored_device.json").read_text())
out["device8"] = {"Q4_0": dv["q40"]["approved"], "Q4_K_M": dv["q4km"]["approved"]}
(D / "consolidation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
