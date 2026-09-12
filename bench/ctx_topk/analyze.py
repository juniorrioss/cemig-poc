#!/usr/bin/env python3
"""
analyze.py — PAINEL DE EIXOS da v4 (ordem 2 do capitão) + trade-off JANELA×TRECHOS (ordem 1).

Lê data/judge_evaluations.json (juiz vLLM 27B, 4 eixos/item) + data/responses_*.json
(tokens/prompt/retrieval_hit) e produz data/analysis.json + tabelas legíveis:

ORDEM 2 (painel por eixo, além do gate binário) — por célula (topK×trecho) E global:
  - por eixo (acerto_factual, fidelidade_contexto, qualidade_pt, citacao_fonte):
    média, desvio (pstdev), mediana, HISTOGRAMA 1-5;
  - EIXO-GARGALO explícito (menor média) por célula e global.

ORDEM 1 (janela×trechos) — a pergunta que decide "adiantou enxergar mais?":
  - retrieval_hit% (chunk-ouro visível) por topK;
  - conversão chunk-certo -> resposta aprovada (e -> acerto>=4);
  - QUANTO do ganho teórico de R@5 (visão) vira resposta aprovada de fato:
    Δaprovadas / Δchunk-visível vs o baseline k2;
  - tokens de prompt médios por célula (custo do prompt = proxy do TTFT no aparelho).

Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
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

CELLS = [f"k{k}_{t}" for k in (2, 3, 4, 5) for t in ("full", "trim")]

# Projeção mobile (device-bench, bench/device-liquid/README.md) p/ o 1.2B QAD-Q4_0 no S24+.
# TTFT depende do PREFILL (tamanho do prompt); decode do nº de tokens gerados.
DEV = {"prefill_s24": 219.0, "decode_s24": 39.9, "s21_slowdown": 2.25}
READING_TOK_S = 7.0  # teto de leitura adulta PT-BR (~4-7 tok/s)


def load_evals(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {mk: mv["modes"]["classic_rag"]["items"] for mk, mv in d["models"].items()}


def load_responses(cell: str) -> Dict[str, Dict[str, Any]]:
    p = _HERE / "data" / f"responses_{cell}.json"
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


def item_global(m: Dict[str, int]) -> float:
    return (m["acerto_factual"] * 0.35 + m["fidelidade_contexto"] * 0.30 +
            m["citacao_fonte"] * 0.20 + m["qualidade_pt"] * 0.15)


def gate_metrics(items: List[Dict[str, Any]], resp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    means = {AXIS_SHORT[ax]: st.mean([i["metrics"][ax] for i in items]) for ax in AXES}
    glob = means["fac"] * 0.35 + means["fid"] * 0.30 + means["cit"] * 0.20 + means["pt"] * 0.15

    gate_pass = 0
    quase_by_axis = {"fac": 0, "fid": 0, "pt": 0, "cit": 0, "global": 0}
    quase_total = 0
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
            key = {"fac": "acerto_factual", "fid": "fidelidade_contexto", "cit": "citacao_fonte"}[k]
            if m[key] >= 3:
                iso[k] += 1
        if m["qualidade_pt"] >= 3:
            iso["pt"] += 1
        if all(conds.values()):
            gate_pass += 1
        else:
            failed = [k for k, ok in conds.items() if not ok]
            if len(failed) == 1:
                quase_total += 1
                quase_by_axis[failed[0]] += 1

    ok = [i for i in items if resp.get(i["item_id"], {}).get("retrieval_hit")]
    n_ok = len(ok)
    conv_gate = 100 * sum(1 for i in ok if i["pass_gate"]) / n_ok if n_ok else 0.0
    conv_fac4 = 100 * sum(1 for i in ok if i["metrics"]["acerto_factual"] >= 4) / n_ok if n_ok else 0.0
    bottleneck_axis = min(means, key=means.get)
    return {
        "n": n, "means": {k: round(v, 3) for k, v in means.items()}, "global": round(glob, 3),
        "gate_pass_pct": round(100 * gate_pass / n, 1), "gate_pass_n": gate_pass,
        "isolated_axis_pass_pct": {k: round(100 * v / n, 1) for k, v in iso.items()},
        "quase_pass_total": quase_total, "quase_pass_by_axis": quase_by_axis,
        "bottleneck_axis": bottleneck_axis,
        "recall_hit_pct": round(100 * n_ok / n, 1), "n_retrieval_ok": n_ok,
        "conv_chunkcerto_aprovada_pct": round(conv_gate, 1),
        "conv_chunkcerto_acerto4_pct": round(conv_fac4, 1),
    }


def token_and_mobile(resp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    items = list(resp.values())
    if not items:
        return {}
    pt = [it["prompt_tokens"] for it in items]
    ct = [it["completion_tokens"] for it in items]
    avg_pt, avg_ct = st.mean(pt), st.mean(ct)
    ttft_s24 = avg_pt / DEV["prefill_s24"]
    decode_s24 = avg_ct / DEV["decode_s24"]
    total_s24 = ttft_s24 + decode_s24
    return {
        "avg_prompt_tok": round(avg_pt, 1), "median_prompt_tok": st.median(pt), "max_prompt_tok": max(pt),
        "avg_completion_tok": round(avg_ct, 1), "median_completion_tok": st.median(ct),
        "ttft_s24_s": round(ttft_s24, 2), "decode_s24_s": round(decode_s24, 2),
        "total_s24_s": round(total_s24, 2), "total_s21_s": round(total_s24 * DEV["s21_slowdown"], 2),
        "decode_above_reading": DEV["decode_s24"] >= READING_TOK_S,
    }


def main() -> None:
    evals = load_evals(_HERE / "data" / "judge_evaluations.json")
    cells: Dict[str, Any] = {}
    for cell in CELLS:
        if cell not in evals:
            continue
        resp = load_responses(cell)
        cells[cell] = {
            "axis_stats": axis_stats(evals[cell]),
            "gate": gate_metrics(evals[cell], resp),
            "mobile": token_and_mobile(resp),
        }

    # Eixo-gargalo GLOBAL (média dos eixos em todas as células).
    gax = {ax: [] for ax in ("fac", "fid", "pt", "cit")}
    for c in cells.values():
        for ax in gax:
            gax[ax].append(c["gate"]["means"][ax])
    global_means = {ax: round(st.mean(v), 3) for ax, v in gax.items()}
    global_bottleneck = min(global_means, key=global_means.get)

    # ORDEM 1: quanto do ganho de VISÃO (chunk-ouro visível) vira resposta aprovada?
    # Baseline = k2_full (topK atual). Δ vs baseline por célula.
    tradeoff = {}
    base = "k2_full"
    if base in cells:
        b_hit = cells[base]["gate"]["n_retrieval_ok"]
        b_pass = cells[base]["gate"]["gate_pass_n"]
        b_fac4 = round(cells[base]["gate"]["conv_chunkcerto_acerto4_pct"] * cells[base]["gate"]["n_retrieval_ok"] / 100)
        for cell, c in cells.items():
            d_hit = c["gate"]["n_retrieval_ok"] - b_hit
            d_pass = c["gate"]["gate_pass_n"] - b_pass
            # acerto>=4 absolutos p/ um sinal menos rígido que o gate
            fac4_abs = round(c["gate"]["conv_chunkcerto_acerto4_pct"] * c["gate"]["n_retrieval_ok"] / 100)
            d_fac4 = fac4_abs - b_fac4
            tradeoff[cell] = {
                "d_chunk_visible": d_hit,
                "d_gate_pass": d_pass,
                "d_acerto4_abs": d_fac4,
                "convert_ratio_gate": round(d_pass / d_hit, 3) if d_hit else None,
                "convert_ratio_acerto4": round(d_fac4 / d_hit, 3) if d_hit else None,
            }

    report = {
        "cells": cells,
        "global_axis_means": global_means,
        "global_bottleneck_axis": global_bottleneck,
        "tradeoff_vs_k2full": tradeoff,
        "notes": {
            "gate": "fac>=3 & fid>=3 & cit>=3 & global_item>=3.5",
            "conv": "% das perguntas com chunk-ouro visível (retrieval_hit) que passaram o gate",
            "tradeoff": "Δ vs k2_full (topK atual). convert_ratio = Δrespostas_aprovadas / Δchunks_visíveis. "
                        "Responde 'adiantou enxergar mais?': se ~0, a visão extra NÃO vira resposta aprovada.",
            "mobile": "engine GPU não vale; projeção device-bench (S24+ 1.2B QAD-Q4_0). "
                      "TTFT = prompt_tok/219; decode = gen_tok/39.9; leitura adulta ~4-7 tok/s.",
        },
    }
    (_HERE / "data" / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Tabela mestre ----
    print("=" * 118)
    print("TABELA MESTRE — célula (topK×trecho) × eixos | Global | Gate% | visão(R_hit) | conv | gargalo | prompt_tok | TTFT")
    print("=" * 118)
    hdr = (f"{'célula':10} {'Fac':>5} {'Fid':>5} {'PT':>5} {'Cit':>5} {'Glob':>5} {'Gate%':>6} "
           f"{'Rhit%':>6} {'conv%':>6} {'fac4%':>6} {'gargalo':>8} {'ptokμ':>6} {'TTFTs':>6} {'totS24':>7}")
    print(hdr)
    print("-" * 118)
    for cell in CELLS:
        if cell not in cells:
            continue
        g = cells[cell]["gate"]
        m = g["means"]
        mo = cells[cell]["mobile"]
        print(f"{cell:10} {m['fac']:5.2f} {m['fid']:5.2f} {m['pt']:5.2f} {m['cit']:5.2f} "
              f"{g['global']:5.2f} {g['gate_pass_pct']:6.1f} {g['recall_hit_pct']:6.1f} "
              f"{g['conv_chunkcerto_aprovada_pct']:6.1f} {g['conv_chunkcerto_acerto4_pct']:6.1f} "
              f"{g['bottleneck_axis']:>8} {mo['avg_prompt_tok']:6.0f} {mo['ttft_s24_s']:5.2f}s {mo['total_s24_s']:6.2f}s")

    print("\n" + "=" * 78)
    print(f"EIXO-GARGALO GLOBAL: {global_bottleneck.upper()}  (médias: "
          + ", ".join(f"{k}={v}" for k, v in global_means.items()) + ")")
    print("=" * 78)

    print("\nORDEM 1 — 'ADIANTOU ENXERGAR MAIS?' (Δ vs k2_full = topK atual):")
    print(f"{'célula':10} {'Δvisível':>9} {'Δgate_pass':>11} {'Δacerto4':>9} "
          f"{'conv_gate':>10} {'conv_ac4':>9}")
    for cell in CELLS:
        if cell not in tradeoff:
            continue
        t = tradeoff[cell]
        cg = "-" if t["convert_ratio_gate"] is None else f"{t['convert_ratio_gate']:.2f}"
        ca = "-" if t["convert_ratio_acerto4"] is None else f"{t['convert_ratio_acerto4']:.2f}"
        print(f"{cell:10} {t['d_chunk_visible']:9d} {t['d_gate_pass']:11d} {t['d_acerto4_abs']:9d} "
              f"{cg:>10} {ca:>9}")

    print("\nPAINEL POR EIXO (histograma 1-5) + QUASE-PASS por célula:")
    for cell in CELLS:
        if cell not in cells:
            continue
        g = cells[cell]["gate"]
        ax = cells[cell]["axis_stats"]
        print(f"\n[{cell}]  gargalo={g['bottleneck_axis']}  gate={g['gate_pass_pct']}%  "
              f"quase-pass={g['quase_pass_total']} (por eixo: {g['quase_pass_by_axis']})")
        for a in ("fac", "fid", "pt", "cit"):
            print(f"   {a}: μ={ax[a]['mean']:.2f} σ={ax[a]['std']:.2f} med={ax[a]['median']} hist={ax[a]['hist']}")

    print(f"\nAnálise salva em {(_HERE / 'data' / 'analysis.json')}")


if __name__ == "__main__":
    main()
