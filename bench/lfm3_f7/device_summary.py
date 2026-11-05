#!/usr/bin/env python3
"""device_summary.py — tabela por caso + médias de um f7_e2e_results*.json do aparelho."""
import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
print(d["model"], "pico PSS %.0f MB" % d["peak_pss_mb"])
print(f"{'id':10s} {'tipo':8s} {'tool':4s} {'gate':6s} {'man':3s} {'decis':>6s} {'busca':>5s} {'ttft':>5s} {'decode':>6s} {'tok':>4s} {'t/s':>5s} {'total':>6s} {'prompt':>6s}")
tool = []
for c in d["cases"]:
    print(f"{c['id'][:10]:10s} {c['kind'][:8]:8s} {str(c.get('called_tool'))[:4]:4s} {c.get('gate','')[:6]:6s} {('S' if c.get('manual_on') else '-'):3s} "
          f"{c.get('decision_ms',0):6d} {c.get('search_ms',0):5d} {c.get('ttft_ms',0):5d} {c.get('decode_ms',0):6d} {c.get('completion_tokens',0):4d} {c.get('tok_per_s',0):5.1f} {c.get('total_ms',0):6d} {c.get('prompt_tokens',0):6d}")
    if c.get("called_tool"):
        tool.append(c)
if tool:
    m = lambda k: st.mean(c[k] for c in tool)
    print("médias (casos com busca, n=%d): decisão %.0f · busca %.0f · TTFT síntese %.0f · decisão+busca+TTFT %.0f · decode %.0f · total %.0f ms · %.1f t/s · prompt %.0f tok" % (
        len(tool), m("decision_ms"), m("search_ms"), m("ttft_ms"), m("decision_ms") + m("search_ms") + m("ttft_ms"), m("decode_ms"), m("total_ms"), m("tok_per_s"), m("prompt_tokens")))
