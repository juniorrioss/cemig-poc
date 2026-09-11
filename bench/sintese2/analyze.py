#!/usr/bin/env python3
"""
analyze.py — Métricas do shootout de síntese v2 (ordem do capitão: além do gate binário,
MÉDIA POR EIXO + DISTRIBUIÇÕES + identificação do EIXO-GARGALO).

Lê data/judge_evaluations.json (juiz vLLM 27B, 4 eixos por item) + data/responses_*.json
(tokens, retrieval_hit) e produz data/analysis.json com, por modelo × condição:
  - por eixo: média, desvio, mediana, histograma (1-5) de acerto/fidelidade/pt/citação;
  - EIXO-GARGALO por célula (menor média) e global;
  - gate-pass% (>=3.5 global ∧ fac,fid,cit>=3) e gates alternativos:
      * pass isolado por eixo (>=3 em cada eixo separadamente);
      * 'quase-pass' (reprovou por 1 eixo só) com o eixo culpado;
  - conversão chunk-certo -> aprovada E chunk-certo -> acerto>=4;
  - tokens médios + projeção de latência mobile (device-bench) + RAM estimada.

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent

AXES = ["acerto_factual", "fidelidade_contexto", "qualidade_pt", "citacao_fonte"]
AXIS_SHORT = {"acerto_factual": "fac", "fidelidade_contexto": "fid",
              "qualidade_pt": "pt", "citacao_fonte": "cit"}

# Projeção mobile (device-bench, bench/device-liquid/README.md):
# QAD-Q4_0 1.2B no S24+: prefill pp800=219 tok/s, decode(real)=39.9, TTFT pp800=3.65s, RAM 1420.8 MiB.
# S21 ~2.25x mais lento. Modelos maiores: escala por nº de parâmetros (FLOPs/token).
DEV = {"prefill_1p2b": 219.0, "decode_1p2b": 39.9, "s21_slowdown": 2.25, "ram_1p2b_mib": 1420.8}
# Parâmetros efetivos por família (bilhões) p/ escalar prefill/decode/RAM vs 1.2B baseline.
PARAMS_B = {"lfm1.2b": 1.2, "lfm2.6b_noth": 2.6, "minicpm2b": 2.52, "minicpm1b": 1.0}
PROMPT_TOK = 900  # RAG topk2 + few-shot no completo ~ maior; usamos 900 como média conservadora
READING_TOK_S = 7.0  # teto de leitura adulta PT-BR (~4-7 tok/s)


def load_evals(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {mk: mv["modes"]["classic_rag"]["items"] for mk, mv in d["models"].items()}


def load_responses(cfg: str) -> Dict[str, Dict[str, Any]]:
    p = _HERE / "data" / f"responses_{cfg}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in d["items"]}


def hist(values: List[int]) -> Dict[str, int]:
    h = {str(k): 0 for k in range(1, 6)}
    for v in values:
        h[str(int(v))] += 1
    return h


def axis_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for ax in AXES:
        vals = [i["metrics"][ax] for i in items]
        out[AXIS_SHORT[ax]] = {
            "mean": round(st.mean(vals), 3),
            "std": round(st.pstdev(vals), 3),
            "median": st.median(vals),
            "hist": hist(vals),
        }
    return out


def gate_metrics(items: List[Dict[str, Any]], resp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    means = {AXIS_SHORT[ax]: st.mean([i["metrics"][ax] for i in items]) for ax in AXES}
    glob = means["fac"] * 0.35 + means["fid"] * 0.30 + means["cit"] * 0.20 + means["pt"] * 0.15

    # Gate oficial: fac>=3 ∧ fid>=3 ∧ cit>=3 ∧ global(item)>=3.5.
    def item_global(m):
        return m["acerto_factual"] * 0.35 + m["fidelidade_contexto"] * 0.30 + \
               m["citacao_fonte"] * 0.20 + m["qualidade_pt"] * 0.15

    gate_pass = 0
    quase_by_axis = {"fac": 0, "fid": 0, "pt": 0, "cit": 0, "global": 0}
    quase_total = 0
    # gates isolados por eixo (>=3 em cada eixo, independente dos demais)
    iso = {"fac": 0, "fid": 0, "pt": 0, "cit": 0}
    for i in items:
        m = i["metrics"]
        ig = item_global(m)
        conds = {
            "fac": m["acerto_factual"] >= 3,
            "fid": m["fidelidade_contexto"] >= 3,
            "cit": m["citacao_fonte"] >= 3,
            "global": ig >= 3.5,
        }
        for k in ("fac", "fid", "cit"):
            if m[{"fac": "acerto_factual", "fid": "fidelidade_contexto",
                  "cit": "citacao_fonte"}[k]] >= 3:
                iso[k] += 1
        if m["qualidade_pt"] >= 3:
            iso["pt"] += 1
        passed = all(conds.values())
        if passed:
            gate_pass += 1
        else:
            failed = [k for k, ok in conds.items() if not ok]
            if len(failed) == 1:  # 'quase-pass': reprovou por 1 eixo só
                quase_total += 1
                quase_by_axis[failed[0]] += 1

    # Conversão chunk-certo (retrieval_hit) -> aprovada e -> acerto>=4.
    ok = [i for i in items if resp.get(i["item_id"], {}).get("retrieval_hit")]
    n_ok = len(ok)
    conv_gate = 100 * sum(1 for i in ok if i["pass_gate"]) / n_ok if n_ok else 0.0
    conv_fac4 = 100 * sum(1 for i in ok if i["metrics"]["acerto_factual"] >= 4) / n_ok if n_ok else 0.0

    bottleneck_axis = min(means, key=means.get)
    return {
        "n": n, "means": {k: round(v, 3) for k, v in means.items()}, "global": round(glob, 3),
        "gate_pass_pct": round(100 * gate_pass / n, 1), "gate_pass_n": gate_pass,
        "isolated_axis_pass_pct": {k: round(100 * v / n, 1) for k, v in iso.items()},
        "quase_pass_total": quase_total,
        "quase_pass_by_axis": quase_by_axis,
        "bottleneck_axis": bottleneck_axis,
        "recall_top2_pct": round(100 * n_ok / n, 1), "n_retrieval_ok": n_ok,
        "conv_chunkcerto_aprovada_pct": round(conv_gate, 1),
        "conv_chunkcerto_acerto4_pct": round(conv_fac4, 1),
    }


def token_and_mobile(cfg: str, resp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    items = list(resp.values())
    if not items:
        return {}
    ct = [it["completion_tokens"] for it in items]
    avg_ct = st.mean(ct)
    params = PARAMS_B.get(cfg.split("__")[0], 1.2)
    scale = params / 1.2
    pf = DEV["prefill_1p2b"] / scale
    dec = DEV["decode_1p2b"] / scale
    ttft_s24 = PROMPT_TOK / pf
    decode_s24 = avg_ct / dec
    total_s24 = ttft_s24 + decode_s24
    ram = DEV["ram_1p2b_mib"] * scale
    return {
        "avg_completion_tok": round(avg_ct, 1),
        "median_completion_tok": st.median(ct),
        "max_completion_tok": max(ct),
        "prefill_toks_s24": round(pf, 1), "decode_toks_s24": round(dec, 1),
        "ttft_s24_s": round(ttft_s24, 2), "decode_s24_s": round(decode_s24, 2),
        "total_s24_s": round(total_s24, 2), "total_s21_s": round(total_s24 * DEV["s21_slowdown"], 2),
        "ram_mib_est": round(ram), "decode_above_reading": dec >= READING_TOK_S,
    }


def main() -> None:
    evals = load_evals(_HERE / "data" / "judge_evaluations.json")
    cells: Dict[str, Any] = {}
    for cell_key, items in evals.items():
        # cell_key formato: '<cand>__<condition>'
        cand = cell_key.split("__")[0]
        resp = load_responses(cell_key) or load_responses(f"{cand}_{cell_key.split('__')[-1]}")
        cells[cell_key] = {
            "axis_stats": axis_stats(items),
            "gate": gate_metrics(items, resp),
            "mobile": token_and_mobile(cand, resp),
        }

    # Eixo-gargalo GLOBAL: média de cada eixo agregada em todas as células.
    global_axis_means = {ax: [] for ax in ("fac", "fid", "pt", "cit")}
    for c in cells.values():
        for ax in global_axis_means:
            global_axis_means[ax].append(c["gate"]["means"][ax])
    global_means = {ax: round(st.mean(v), 3) for ax, v in global_axis_means.items()}
    global_bottleneck = min(global_means, key=global_means.get)

    report = {
        "cells": cells,
        "global_axis_means": global_means,
        "global_bottleneck_axis": global_bottleneck,
        "notes": {
            "gate": "fac>=3 & fid>=3 & cit>=3 & global_item>=3.5",
            "quase_pass": "reprovou por 1 eixo só; by_axis mostra o eixo culpado",
            "conv": "% das perguntas com chunk-ouro no top-2 (retrieval_hit) que passaram",
            "mobile": "engine GPU não vale; projeção device-bench (S24+ 1.2B QAD-Q4_0), "
                      "escala por nº de parâmetros; leitura adulta ~4-7 tok/s",
        },
    }
    (_HERE / "data" / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Impressão legível: tabela mestre ---
    print("=" * 108)
    print("TABELA MESTRE — modelo × condição × eixos (média) | Global | Gate% | conv(ok->ok) | gargalo")
    print("=" * 108)
    hdr = (f"{'célula':26} {'n':>3} {'Fac':>5} {'Fid':>5} {'PT':>5} {'Cit':>5} "
           f"{'Glob':>5} {'Gate%':>6} {'R@2%':>5} {'conv%':>6} {'fac4%':>6} {'gargalo':>8}")
    print(hdr)
    print("-" * 108)
    for key in sorted(cells):
        g = cells[key]["gate"]
        m = g["means"]
        print(f"{key:26} {g['n']:3d} {m['fac']:5.2f} {m['fid']:5.2f} {m['pt']:5.2f} {m['cit']:5.2f} "
              f"{g['global']:5.2f} {g['gate_pass_pct']:6.1f} {g['recall_top2_pct']:5.1f} "
              f"{g['conv_chunkcerto_aprovada_pct']:6.1f} {g['conv_chunkcerto_acerto4_pct']:6.1f} "
              f"{g['bottleneck_axis']:>8}")

    print("\n" + "=" * 78)
    print(f"EIXO-GARGALO GLOBAL: {global_bottleneck.upper()}  (médias: "
          + ", ".join(f"{k}={v}" for k, v in global_means.items()) + ")")
    print("=" * 78)

    print("\nDISTRIBUIÇÕES (histograma 1-5) e QUASE-PASS por célula:")
    for key in sorted(cells):
        g = cells[key]["gate"]
        ax = cells[key]["axis_stats"]
        print(f"\n[{key}]  gargalo={g['bottleneck_axis']}  quase-pass={g['quase_pass_total']} "
              f"(por eixo: {g['quase_pass_by_axis']})")
        for a in ("fac", "fid", "pt", "cit"):
            print(f"   {a}: μ={ax[a]['mean']:.2f} σ={ax[a]['std']:.2f} med={ax[a]['median']} "
                  f"hist={ax[a]['hist']}")

    print("\n" + "=" * 100)
    print("PROJEÇÃO MOBILE (device-bench; latência GPU não vale)")
    print("=" * 100)
    print(f"{'célula':26} {'tok μ':>6} {'tok med':>7} {'dec t/s':>7} {'TTFT S24':>9} "
          f"{'tot S24':>8} {'tot S21':>8} {'RAM MiB':>8} {'>=leit':>7}")
    for key in sorted(cells):
        mo = cells[key]["mobile"]
        if not mo:
            continue
        print(f"{key:26} {mo['avg_completion_tok']:6.0f} {mo['median_completion_tok']:7.0f} "
              f"{mo['decode_toks_s24']:7.1f} {mo['ttft_s24_s']:8.2f}s {mo['total_s24_s']:7.2f}s "
              f"{mo['total_s21_s']:7.2f}s {mo['ram_mib_est']:8d} {str(mo['decode_above_reading']):>7}")
    print(f"\nAnálise salva em {(_HERE/'data'/'analysis.json')}")


if __name__ == "__main__":
    main()
