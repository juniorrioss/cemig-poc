#!/usr/bin/env python3
"""
consolidate.py — Junta todos os score_*.json em uma tabela mestre + análise do teto.

Produz data/consolidation.json com:
  - tabela de variantes (Parte 1): 2.6B x 6 prompts, retrieved.
  - teto do 27B (Parte 2): baseline/numeros x retrieved/oracle.
  - oráculo do 2.6B e 1.2B (separa retrieval de geração).
  - respostas literais das 2 perguntas do capitão por variante.
  - veredito automático sobre severidade da régua.

Comentários PT-BR; código em inglês. Procedência: task poc-prompt-teto.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"


def load_metrics(name: str):
    p = DATA / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def captain_from(resp_name: str):
    p = DATA / resp_name
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for it in d["items"]:
        if it.get("captain_case"):
            out[it["item_id"]] = {"response": it.get("response", ""),
                                  "chunks_docs": it.get("chunks_docs", [])}
    return out


def main() -> None:
    variants = ["baseline", "anti_evasao", "acao_primeiro", "numeros",
                "combinado_6f", "combinado_4f"]

    part1 = []
    captain_by_variant = {}
    for v in variants:
        m = load_metrics(f"score_lfm2.6b_{v}_retrieved.json")
        if m:
            part1.append(m)
        captain_by_variant[v] = captain_from(f"resp_lfm2.6b_{v}_retrieved.json")

    part2_ceiling = {
        "judge27b_baseline_retrieved": load_metrics("score_judge27b_baseline_retrieved.json"),
        "judge27b_numeros_retrieved": load_metrics("score_judge27b_numeros_retrieved.json"),
        "judge27b_baseline_oracle": load_metrics("score_judge27b_baseline_oracle.json"),
        "judge27b_numeros_oracle": load_metrics("score_judge27b_numeros_oracle.json"),
    }

    oracle_gen = {
        "lfm2.6b_baseline_oracle": load_metrics("score_lfm2.6b_baseline_oracle.json"),
        "lfm2.6b_numeros_oracle": load_metrics("score_lfm2.6b_numeros_oracle.json"),
        "lfm1.2b_baseline_retrieved": load_metrics("score_lfm1.2b_baseline_retrieved.json"),
        "lfm1.2b_numeros_retrieved": load_metrics("score_lfm1.2b_numeros_retrieved.json"),
    }

    # Veredito automático de severidade da régua (Parte 2).
    ceil_oracle = part2_ceiling["judge27b_numeros_oracle"]["approval_pct"]
    if ceil_oracle >= 60:
        verdict = ("REGUA OK. O teto do 27B com chunk-ouro é %.1f%% (>60%%): a régua não "
                   "está severa; o gargalo é o SLM+retrieval." % ceil_oracle)
    elif ceil_oracle < 40:
        verdict = ("REGUA SEVERA. Nem o 27B com chunk-ouro passa (%.1f%% <40%%): recalibrar."
                   % ceil_oracle)
    else:
        verdict = ("REGUA NA FRONTEIRA (%.1f%%). O 27B com chunk-ouro fica entre 40-60%%; "
                   "a régua é exigente mas mede capacidade real." % ceil_oracle)

    best_p1 = max(part1, key=lambda m: (m["approval_pct"], m["mean_coverage"]))

    out = {
        "task": "poc-prompt-teto",
        "n": 151,
        "regua": "bench/regua (ruler.py + judge_regua 27B), limiar 0.5, sem citação no gate",
        "retriever": "v4 embarcado (RRF k=10 w=2,1,2), R@2=51.0%",
        "part1_prompt_variants_2.6b_retrieved": sorted(
            part1, key=lambda m: -m["approval_pct"]),
        "part1_best": {"variant": best_p1["variant"],
                       "approval_pct": best_p1["approval_pct"]},
        "part2_ceiling_27b": part2_ceiling,
        "oracle_and_1.2b": oracle_gen,
        "captain_cases_by_variant_2.6b": captain_by_variant,
        "captain_cases_1.2b_baseline": captain_from("resp_lfm1.2b_baseline_retrieved.json"),
        "captain_cases_27b_numeros": captain_from("resp_judge27b_numeros_retrieved.json"),
        "verdict_regua_severity": verdict,
    }
    (DATA / "consolidation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=== PARTE 1: variantes de prompt no 2.6B (retrieved, n=151) ===")
    print(f"{'variant':<16}{'aprov%':>8}{'cov':>7}{'halluc%':>9}{'conv_hit%':>10}{'tok':>7}")
    for m in sorted(part1, key=lambda m: -m["approval_pct"]):
        print(f"{m['variant']:<16}{m['approval_pct']:>8}{m['mean_coverage']:>7}"
              f"{m['hallucination_pct']:>9}{m['conv_chunk_hit_pct']:>10}"
              f"{m['avg_completion_tokens']:>7.0f}")
    print("\n=== PARTE 2: teto do 27B ===")
    for k, m in part2_ceiling.items():
        if m:
            print(f"{k:<34} aprov={m['approval_pct']:>5}% cov={m['mean_coverage']} "
                  f"conv_hit={m['conv_chunk_hit_pct']}% conv_miss={m['conv_chunk_miss_pct']}%")
    print("\n=== ORÁCULO 2.6B + 1.2B embarcado ===")
    for k, m in oracle_gen.items():
        if m:
            print(f"{k:<34} aprov={m['approval_pct']:>5}% cov={m['mean_coverage']} "
                  f"conv_hit={m['conv_chunk_hit_pct']}%")
    print("\nVEREDITO:", verdict)


if __name__ == "__main__":
    main()
