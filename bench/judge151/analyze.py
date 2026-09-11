#!/usr/bin/env python3
"""
analyze.py — Responde as 3 perguntas do capitão a partir do judge_evaluations.json:

1. Tabela modelo × 4 eixos × global × gate-pass% (antes v2 pré-fix vs agora híbrido no 1.2b;
   1.2b vs 2.6b agora). O 'antes' vem de bench/results/v2 (classic_rag do lfm2.5-1.2b-instruct).
2. DECOMPOSIÇÃO DO GARGALO: nas perguntas com retrieval OK (chunk-ouro no top-2), qual %
   cada modelo converte em resposta correta (pass_gate e acerto>=4)? -> 'quanto é o LM'.
3. Nas de retrieval RUIM: o 2.6b recusa melhor ('não sei' correto) ou alucina mais?
4. Projeção de latência mobile (tokens medidos × tok/s do device-bench) + RAM + caber no S24+/S21,
   com o novo eixo do capitão: TTFT e decode tok/s vs leitura humana (~4-7 tok/s).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

REFUSAL_PAT = re.compile(r"não sei|nao sei|não encontrei|nao encontrei|não consta|nao consta|"
                         r"não é possível|nao e possivel|não foi possível|informações suficientes",
                         re.IGNORECASE)


def load_evals() -> Dict[str, List[Dict[str, Any]]]:
    d = json.loads((_HERE / "data" / "judge_evaluations.json").read_text(encoding="utf-8"))
    out = {}
    for mk, mv in d["models"].items():
        out[mk] = mv["modes"]["classic_rag"]["items"]
    return out


def load_responses(cfg: str) -> Dict[str, Dict[str, Any]]:
    d = json.loads((_HERE / "data" / f"responses_{cfg}.json").read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in d["items"]}


def axis_table(evals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for cfg, items in evals.items():
        n = len(items)
        fac = sum(i["metrics"]["acerto_factual"] for i in items) / n
        fid = sum(i["metrics"]["fidelidade_contexto"] for i in items) / n
        pt = sum(i["metrics"]["qualidade_pt"] for i in items) / n
        cit = sum(i["metrics"]["citacao_fonte"] for i in items) / n
        glob = fac * 0.35 + fid * 0.30 + cit * 0.20 + pt * 0.15
        gate = 100 * sum(1 for i in items if i["pass_gate"]) / n
        rows.append({"cfg": cfg, "n": n, "fac": fac, "fid": fid, "pt": pt, "cit": cit,
                     "global": glob, "gate": gate})
    return rows


def v2_baseline() -> Dict[str, Any]:
    """'Antes' (v2, pré-fix FTS5 + pré-classificador): 1.2b Instruct classic_rag em 101 perguntas."""
    import csv
    path = _ROOT / "bench" / "results" / "v2" / "bench_summary_v2.csv"
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["model"] == "lfm2.5-1.2b-instruct" and r["mode"] == "classic_rag":
            return {
                "recall": float(r["retrieval_recall_pct"]), "fac": float(r["acerto_factual"]),
                "fid": float(r["fidelidade_contexto"]), "pt": float(r["qualidade_pt"]),
                "cit": float(r["citacao_fonte"]), "global": float(r["score_global"]),
                "gate": float(r["pass_gate_pct"]), "n": int(r["total_items"]),
            }
    return {}


def bottleneck(evals, responses) -> Dict[str, Any]:
    """Decompõe: retrieval OK vs RUIM, e conversão do LM."""
    out = {}
    for cfg, items in evals.items():
        resp = responses[cfg]
        ok = [i for i in items if resp[i["item_id"]]["retrieval_hit"]]
        bad = [i for i in items if not resp[i["item_id"]]["retrieval_hit"]]

        def conv(sub):
            n = len(sub)
            if n == 0:
                return {"n": 0}
            passg = sum(1 for i in sub if i["pass_gate"])
            fac4 = sum(1 for i in sub if i["metrics"]["acerto_factual"] >= 4)
            fac3 = sum(1 for i in sub if i["metrics"]["acerto_factual"] >= 3)
            return {"n": n, "pass_pct": 100 * passg / n, "fac4_pct": 100 * fac4 / n,
                    "fac3_pct": 100 * fac3 / n}

        # Recusa nas de retrieval RUIM (idealmente deveria dizer 'não sei')
        refusals = 0
        halluc_fac1 = 0  # afirmou algo e o juiz deu acerto=1 (grave) -> alucinação
        for i in bad:
            r = resp[i["item_id"]]["response"]
            if REFUSAL_PAT.search(r):
                refusals += 1
            elif i["metrics"]["acerto_factual"] <= 1:
                halluc_fac1 += 1
        nb = len(bad)
        out[cfg] = {
            "retrieval_ok": conv(ok),
            "retrieval_bad": conv(bad),
            "bad_refusal_pct": 100 * refusals / nb if nb else 0,
            "bad_refusals": refusals,
            "bad_halluc_fac1_pct": 100 * halluc_fac1 / nb if nb else 0,
            "bad_halluc_fac1": halluc_fac1,
            "bad_n": nb,
        }
    return out


def token_stats(responses) -> Dict[str, Any]:
    out = {}
    for cfg, resp in responses.items():
        items = list(resp.values())
        ct = [it["completion_tokens"] for it in items]
        rt = [it.get("reasoning_tokens", 0) for it in items]
        out[cfg] = {
            "avg_completion": sum(ct) / len(ct),
            "avg_reasoning": sum(rt) / len(rt),
            "avg_total_gen": (sum(ct)) / len(ct),  # completion inclui thinking no llama-server? não: reasoning separado
            "max_completion": max(ct),
        }
    return out


# Projeção de latência mobile pela tabela do device-bench (bench/device-liquid/README.md).
# QAD-Q4_0 1.2B no S24+: prefill pp800=219 tok/s, pp1500=207; decode(real)=39.9; TTFT pp800=3.65s.
# S21 ~ 2.25x mais lento (device README). O 2.6B não foi medido no device -> projetamos por
# escala de parâmetros (2.6B/1.2B = 2.17x mais FLOPs/token) sobre o 1.2B QAD: prefill ~1/2.17,
# decode ~1/2.17. Prompt RAG topk2 ~ 820-900 tokens (contexto lean).
DEV = {
    "prefill_800_1p2b": 219.0, "decode_1p2b": 39.9, "ttft_800_1p2b": 3.65,
    "s21_slowdown": 2.25, "ram_1p2b_mib": 1420.8,
    "param_ratio_2p6_over_1p2": 2.6 / 1.2,
}


def project_mobile(token_stats: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tok = 850  # RAG topk2 lean médio
    out = {}
    ratio = DEV["param_ratio_2p6_over_1p2"]
    for cfg, ts in token_stats.items():
        is26 = cfg.startswith("lfm2.6b")
        pf = DEV["prefill_800_1p2b"] / (ratio if is26 else 1.0)
        dec = DEV["decode_1p2b"] / (ratio if is26 else 1.0)
        # geração mobile: completion_tokens do llama-server JÁ inclui os tokens de thinking
        # (reasoning_tokens é apenas contagem de palavras do texto de raciocínio, p/ referência).
        gen_tok = ts["avg_completion"]
        ttft_s24 = prompt_tok / pf
        decode_s24 = gen_tok / dec
        total_s24 = ttft_s24 + decode_s24
        # S21
        ttft_s21 = ttft_s24 * DEV["s21_slowdown"]
        total_s21 = total_s24 * DEV["s21_slowdown"]
        ram = DEV["ram_1p2b_mib"] * (ratio if is26 else 1.0)  # aproximação por escala de pesos
        out[cfg] = {
            "prefill_toks": round(pf, 1), "decode_toks": round(dec, 1),
            "gen_tok": round(gen_tok), "ttft_s24_s": round(ttft_s24, 2),
            "decode_s24_s": round(decode_s24, 2), "total_s24_s": round(total_s24, 2),
            "ttft_s21_s": round(ttft_s21, 2), "total_s21_s": round(total_s21, 2),
            "ram_mib_est": round(ram),
            "decode_above_reading": dec >= 7.0,  # leitura adulta PT ~4-7 tok/s
        }
    return out


def main() -> None:
    evals = load_evals()
    responses = {cfg: load_responses(cfg) for cfg in evals}

    tab = axis_table(evals)
    v2 = v2_baseline()
    bn = bottleneck(evals, responses)
    ts = token_stats(responses)
    proj = project_mobile(ts)

    report = {"axis_table": tab, "v2_baseline_1p2b": v2, "bottleneck": bn,
              "token_stats": ts, "mobile_projection": proj}
    (_HERE / "data" / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Impressão legível ---
    print("=" * 78)
    print("TABELA 1 — 4 EIXOS × GLOBAL × GATE-PASS (agora, híbrido, 151 reais)")
    print("=" * 78)
    print(f"{'config':14} {'Fac':>5} {'Fid':>5} {'PT':>5} {'Cit':>5} {'Global':>7} {'Gate%':>6}")
    for r in tab:
        print(f"{r['cfg']:14} {r['fac']:5.2f} {r['fid']:5.2f} {r['pt']:5.2f} {r['cit']:5.2f} "
              f"{r['global']:7.2f} {r['gate']:6.1f}")
    print(f"\nANTES (v2 pré-fix, 1.2b Instruct classic_rag, n={v2.get('n')}): "
          f"Fac {v2.get('fac')} Fid {v2.get('fid')} PT {v2.get('pt')} Cit {v2.get('cit')} "
          f"Global {v2.get('global')} Gate {v2.get('gate')}% | Recall {v2.get('recall')}%")

    print("\n" + "=" * 78)
    print("TABELA 2 — DECOMPOSIÇÃO DO GARGALO (retrieval OK vs RUIM)")
    print("=" * 78)
    for cfg, b in bn.items():
        ok, bad = b["retrieval_ok"], b["retrieval_bad"]
        print(f"\n[{cfg}]")
        print(f"  Retrieval OK  (n={ok['n']:>3}): pass_gate {ok.get('pass_pct',0):5.1f}% | "
              f"acerto>=4 {ok.get('fac4_pct',0):5.1f}% | acerto>=3 {ok.get('fac3_pct',0):5.1f}%")
        print(f"  Retrieval RUIM(n={bad['n']:>3}): pass_gate {bad.get('pass_pct',0):5.1f}% | "
              f"acerto>=4 {bad.get('fac4_pct',0):5.1f}% | acerto>=3 {bad.get('fac3_pct',0):5.1f}%")
        print(f"    Recusa correta ('não sei'): {b['bad_refusals']}/{b['bad_n']} "
              f"({b['bad_refusal_pct']:.1f}%) | Alucinação grave (fac<=1): {b['bad_halluc_fac1']}/{b['bad_n']} "
              f"({b['bad_halluc_fac1_pct']:.1f}%)")

    print("\n" + "=" * 78)
    print("TABELA 3 — PROJEÇÃO MOBILE (device-bench; engine GPU não vale p/ latência)")
    print("=" * 78)
    print(f"{'config':14} {'gen_tok':>7} {'pf t/s':>7} {'dec t/s':>7} {'TTFT S24':>9} "
          f"{'tot S24':>8} {'tot S21':>8} {'RAM MiB':>8} {'>=leitura':>9}")
    for cfg, p in proj.items():
        print(f"{cfg:14} {p['gen_tok']:7d} {p['prefill_toks']:7.1f} {p['decode_toks']:7.1f} "
              f"{p['ttft_s24_s']:8.2f}s {p['total_s24_s']:7.2f}s {p['total_s21_s']:7.2f}s "
              f"{p['ram_mib_est']:8d} {str(p['decode_above_reading']):>9}")
    print("\nLeitura adulta PT ~ 4-7 tok/s: se decode > 7 tok/s, a resposta 'acompanha a leitura'.")
    print(f"Análise salva em {(_HERE/'data'/'analysis.json')}")


if __name__ == "__main__":
    main()
