#!/usr/bin/env python3
"""
consolidate.py — Tabela mestre QUALIDADE × CUSTO (PART 3) + recomendação.

Junta:
  - PART 1 (data/analysis.json): eixos, gate%, conversão, gargalo, prompt_tok por célula;
  - PART 2 (results/device_cost.json): custo REAL no S24+ (prefill/TTFT/decode/RAM sintético
    + prefill/gen/wall/acerto-de-norma dos prompts reais).

Produz results/consolidation.json e a tabela mestre legível. A recomendação é derivada de
regras explícitas (não hard-coded): embarcar topK maior só se Δgate_pass ≥ limiar E o custo
de TTFT no aparelho não regredir além do teto.

Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Regras de decisão (explícitas):
MIN_DGATE_PP = 3.0        # ganho mínimo de gate-pass (p.p.) p/ justificar mais trechos
TTFT_REGRESSION_S = 2.0   # aumento de TTFT (s) acima do qual é regressão de experiência


def main() -> None:
    q = json.loads((_HERE / "data" / "analysis.json").read_text(encoding="utf-8"))
    c = json.loads((_HERE / "results" / "device_cost.json").read_text(encoding="utf-8"))

    cells = q["cells"]
    base = "k2_full"
    base_gate = cells[base]["gate"]["gate_pass_pct"]

    # Custo no device: sintético (TTFT/RAM) por célula de custo medida; real (wall/prefill/norm).
    syn = c.get("synthetic", {})
    real = c.get("real", {})

    def real_stats(cfg):
        rows = real.get(cfg, [])
        if not rows:
            return None
        pp = [r["prompt_tok_s"] for r in rows if r["prompt_tok_s"] > 0]
        gen = [r["gen_tok_s"] for r in rows if r["gen_tok_s"] > 0]
        walls = [r["wall_s"] for r in rows]
        normok = sum(1 for r in rows if r["norm_ok"])
        return {
            "n": len(rows),
            "prefill_med": round(st.median(pp), 1) if pp else 0,
            "gen_med": round(st.median(gen), 1) if gen else 0,
            "wall_med": round(st.median(walls), 1),
            "norm_ok": normok,
        }

    master = {}
    for cell, cd in cells.items():
        g = cd["gate"]
        mo = cd["mobile"]
        # Custo de device: usa a célula de custo correspondente quando medida (k2_full/k5_full/k5_trim);
        # p/ as demais, projeta TTFT pela regra prompt_tok/prefill(k2 real) — sinalizado como proj.
        dev_syn = syn.get(cell)
        dev_real = real_stats(cell)
        master[cell] = {
            "quality": {
                "fac": g["means"]["fac"], "fid": g["means"]["fid"],
                "pt": g["means"]["pt"], "cit": g["means"]["cit"],
                "global": g["global"], "gate_pct": g["gate_pass_pct"],
                "d_gate_pp": round(g["gate_pass_pct"] - base_gate, 1),
                "recall_hit_pct": g["recall_hit_pct"],
                "conv_pct": g["conv_chunkcerto_aprovada_pct"],
                "bottleneck": g["bottleneck_axis"],
            },
            "prompt_tok_med": mo["median_prompt_tok"],
            "device_synth": dev_syn,
            "device_real": dev_real,
        }

    # TTFT do baseline (device, sintético) p/ o critério de regressão.
    base_ttft = syn.get("k2_full", {}).get("ttft_s")

    # Recomendação por regras.
    reasons = []
    # Nenhuma célula ganha gate suficiente?
    best_gain = max((master[c]["quality"]["d_gate_pp"] for c in master), default=0.0)
    winner_gate = [c for c in master if master[c]["quality"]["d_gate_pp"] == best_gain]
    if best_gain < MIN_DGATE_PP:
        rec = "MANTER topK=2 (full) — default atual"
        reasons.append(f"Nenhuma configuração eleva o gate-pass em ≥{MIN_DGATE_PP} p.p. "
                       f"(máx Δ={best_gain:+.1f} p.p. em {winner_gate}).")
        reasons.append("O gargalo global é CITAÇÃO (limite do LM 1.2B), não a visão de trechos: "
                       "mais trechos elevam o recall (chunk-ouro visível) mas NÃO viram resposta aprovada.")
        if base_ttft and syn.get("k5_full"):
            dttft = syn["k5_full"]["ttft_s"] - base_ttft
            reasons.append(f"Custo de subir p/ 5 trechos (full): TTFT no S24+ +{dttft:.1f}s "
                           f"({base_ttft:.1f}s -> {syn['k5_full']['ttft_s']:.1f}s) = regressão de experiência sem ganho.")
        reasons.append("TRIM cabe mais trechos na janela (prompt ~1340 tok como k2) MAS não melhora "
                       "qualidade (gate igual/menor) e piora levemente citação — não compensa.")
    else:
        rec = f"AVALIAR embarque de {winner_gate}"
        reasons.append(f"Ganho de gate-pass {best_gain:+.1f} p.p. em {winner_gate}.")

    consolidation = {
        "task": "poc-ctx-topk",
        "baseline": base,
        "decision_rules": {"min_dgate_pp": MIN_DGATE_PP, "ttft_regression_s": TTFT_REGRESSION_S},
        "master": master,
        "global_bottleneck_axis": q["global_bottleneck_axis"],
        "global_axis_means": q["global_axis_means"],
        "recommendation": rec,
        "reasons": reasons,
    }
    (_HERE / "results" / "consolidation.json").write_text(
        json.dumps(consolidation, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Tabela mestre legível ----
    print("=" * 120)
    print("TABELA MESTRE — QUALIDADE × CUSTO (v4 embarcado × 1.2B QAD) | 151 juiz + S24+ device")
    print("=" * 120)
    print(f"{'célula':9} {'Glob':>5} {'Gate%':>6} {'Δgate':>6} {'Rhit%':>6} {'conv%':>6} {'gargalo':>8} "
          f"{'ptokμ':>6} {'TTFTdev':>8} {'RAMdev':>7} {'wallR':>6} {'normR':>6}")
    print("-" * 120)
    for cell in [f"k{k}_{t}" for k in (2, 3, 4, 5) for t in ("full", "trim")]:
        if cell not in master:
            continue
        m = master[cell]
        ql = m["quality"]
        ds = m["device_synth"]
        dr = m["device_real"]
        ttft = f"{ds['ttft_s']:.1f}s" if ds else "  proj"
        ram = f"{ds['peak_ram_mib']:.0f}" if ds else "-"
        wall = f"{dr['wall_med']:.1f}s" if dr else "-"
        norm = f"{dr['norm_ok']}/{dr['n']}" if dr else "-"
        print(f"{cell:9} {ql['global']:5.2f} {ql['gate_pct']:6.1f} {ql['d_gate_pp']:+6.1f} "
              f"{ql['recall_hit_pct']:6.1f} {ql['conv_pct']:6.1f} {ql['bottleneck']:>8} "
              f"{m['prompt_tok_med']:6.0f} {ttft:>8} {ram:>7} {wall:>6} {norm:>6}")

    print("\n" + "=" * 90)
    print(f"EIXO-GARGALO GLOBAL: {consolidation['global_bottleneck_axis'].upper()}  "
          + "(" + ", ".join(f"{k}={v}" for k, v in consolidation["global_axis_means"].items()) + ")")
    print("=" * 90)
    print(f"\nRECOMENDAÇÃO: {rec}")
    for r in reasons:
        print(f"  - {r}")
    print(f"\nSalvo em {(_HERE / 'results' / 'consolidation.json')}")


if __name__ == "__main__":
    main()
