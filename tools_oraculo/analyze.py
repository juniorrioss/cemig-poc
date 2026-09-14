#!/usr/bin/env python3
"""
analyze.py — Consolida as 4 medições da task poc-tools-oraculo.

Junta os arquivos de score (sft_v2/score_v2.py + sft_v3/refusal_metrics.py) que este harness
produziu com os arquivos de GERAÇÃO (que carregam got_gold/ctx_source/called_tool) e emite:

  MEDIÇÃO 1 — oráculo: aprovação/cobertura/alucinação (média±desvio dos 3 runs), bf16 E Q4,
              lado a lado com base QAD / sft_1_2b sem tool / 2.6B v3 / 27B.
  MEDIÇÃO 2 — recusa: matriz TP/FN/FP/TN + recall/precision/F1 (bf16 E Q4).
  MEDIÇÃO 3 — híbrido: (A) híbrido, (B) tools puro, (C) pipeline fixo sem tool — aprovação,
              alucinação, e (A/B) decisão/reuso.
  MEDIÇÃO 4 — de onde vem a alucinação: decomposição por condição (contexto trouxe o
              chunk-ouro? / contexto errado / não chamou a tool), sobre os runs de A e B.

Referências (números já medidos na POC, de sft_1_2b/data/consolidation.json — reuso, não
recomputamos): base QAD, sft_1_2b, 2.6B v3, 27B. Fonte anotada em cada linha.

Higiene: idempotente, não-interativo, procedência. Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DATA = _HERE / "data"
REF = json.loads((_ROOT / "sft_1_2b" / "data" / "consolidation.json").read_text(encoding="utf-8"))


def _score(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"score_{label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _gen(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"gen_{label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mean_std(xs: List[float]) -> Dict[str, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.mean(xs), 1),
            "std": round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0}


def agg_runs(prefix: str, runs=(1, 2, 3)) -> Dict[str, Any]:
    """Média±desvio de aprovação/cobertura/alucinação sobre os runs de um score prefixo."""
    appr, hall, cov, tok = [], [], [], []
    for r in runs:
        s = _score(f"{prefix}_run{r}")
        if not s:
            continue
        m = s["metrics"]
        appr.append(m.get("approval_pct"))
        hall.append(m.get("hallucination_pct"))
        cov.append(m.get("mean_coverage"))
        tok.append(m.get("avg_completion_tokens"))
    return {"approval": _mean_std(appr), "hallucination": _mean_std(hall),
            "coverage": _mean_std([c for c in cov if c is not None]),
            "avg_tok": _mean_std(tok), "n_runs": len(appr)}


def decision_reuse(prefix: str, runs=(1, 2, 3)) -> Dict[str, Any]:
    """Fração de called_tool e got_gold (médias) a partir dos arquivos de GERAÇÃO."""
    called, got_gold = [], []
    for r in runs:
        g = _gen(f"{prefix}_run{r}")
        if not g:
            continue
        items = [it for it in g["items"] if not it.get("__error__")]
        n = len(items)
        called.append(100 * sum(1 for it in items if it.get("called_tool")) / max(1, n))
        # got_gold só faz sentido quando chamou
        cg = [it for it in items if it.get("called_tool")]
        got_gold.append(100 * sum(1 for it in cg if it.get("got_gold")) / max(1, len(cg)))
    return {"called_pct": _mean_std(called), "got_gold_when_called_pct": _mean_std(got_gold)}


def hallucination_decomposition(prefixes: Dict[str, str], runs=(1, 2, 3)) -> Dict[str, Any]:
    """MEDIÇÃO 4: alucinação por condição (got_gold / contexto errado / não chamou).

    Junta o score (hallucination por item) com a geração (got_gold/called_tool por item).
    Reporta, por config: distribuição das alucinações entre as 3 condições e a taxa de
    alucinação DENTRO de cada condição.
    """
    out: Dict[str, Any] = {}
    for cfg, prefix in prefixes.items():
        # acumula por condição ao longo dos runs
        cond = {"gold_ok": {"n": 0, "hall": 0}, "gold_wrong": {"n": 0, "hall": 0},
                "no_call": {"n": 0, "hall": 0}}
        for r in runs:
            g = _gen(f"{prefix}_run{r}")
            s = _score(f"{prefix}_run{r}")
            if not g or not s:
                continue
            hall_by_id = {it["item_id"]: bool(it.get("hallucination"))
                          for it in s["items"]}
            for it in g["items"]:
                if it.get("__error__"):
                    continue
                iid = it["item_id"]
                if iid not in hall_by_id:
                    continue
                h = hall_by_id[iid]
                if not it.get("called_tool"):
                    c = "no_call"
                elif it.get("got_gold"):
                    c = "gold_ok"
                else:
                    c = "gold_wrong"
                cond[c]["n"] += 1
                cond[c]["hall"] += int(h)
        total_hall = sum(cond[c]["hall"] for c in cond)
        total_n = sum(cond[c]["n"] for c in cond)
        detail = {}
        for c, v in cond.items():
            detail[c] = {
                "n": v["n"],
                "hall_count": v["hall"],
                "hall_rate_pct": round(100 * v["hall"] / v["n"], 1) if v["n"] else None,
                "share_of_all_hall_pct": round(100 * v["hall"] / total_hall, 1) if total_hall else None,
            }
        out[cfg] = {"total_items": total_n, "total_hall": total_hall, "by_condition": detail}
    return out


def build() -> Dict[str, Any]:
    ref_sv = REF["summary_by_variant"]

    # ---- MEDIÇÃO 1: oráculo ----
    oracle = {
        "tools_r128_q4": agg_runs("tools_r128_oracle_q4"),
        "tools_r128_bf16": agg_runs("tools_r128_oracle_bf16"),
        "refs": {
            "base_qad_q4": {"approval": ref_sv["base_qad_q4"]["oracle151"]["approval"],
                            "hallucination": ref_sv["base_qad_q4"]["oracle151"]["hallucination"]},
            "sft_1_2b_r64_q4": {"approval": ref_sv["sft_r64_q4"]["oracle151"]["approval"],
                                "hallucination": ref_sv["sft_r64_q4"]["oracle151"]["hallucination"]},
            "sft_1_2b_r64_bf16": {"approval": ref_sv["sft_r64_bf16"]["oracle151"]["approval"],
                                  "hallucination": ref_sv["sft_r64_bf16"]["oracle151"]["hallucination"]},
            "lfm2.6b_v3_r64_q4": {"approval": ref_sv["v3_r64_q4"]["oracle151"]["approval"],
                                  "hallucination": ref_sv["v3_r64_q4"]["oracle151"]["hallucination"]},
            "lfm2.6b_v3_r64_bf16": {"approval": ref_sv["v3_r64_bf16"]["oracle151"]["approval"],
                                    "hallucination": ref_sv["v3_r64_bf16"]["oracle151"]["hallucination"]},
            "qwen27b": {"approval": ref_sv["27b"]["oracle151"]["approval"],
                        "hallucination": ref_sv["27b"]["oracle151"]["hallucination"]},
        },
    }

    # ---- MEDIÇÃO 2: recusa ----
    refusal = {}
    for q in ("bf16", "q4"):
        p = DATA / f"refusal_confusion_{q}.json"
        if p.exists():
            refusal[f"tools_r128_{q}"] = list(json.loads(p.read_text(encoding="utf-8")).values())[0]
    refusal["refs"] = {
        "sft_1_2b_r64_q4": ref_sv["sft_r64_q4"]["refusal_classifier"],
        "lfm2.6b_v3_r64_q4": ref_sv["v3_r64_q4"]["refusal_classifier"],
        "qwen27b": ref_sv["27b"]["refusal_classifier"],
    }

    # ---- MEDIÇÃO 3: híbrido ----
    def cfgC(variant_q4="sft_r64_q4", variant_bf16="sft_r64_bf16"):
        return {
            "q4": {"approval": ref_sv[variant_q4]["retrieved151"]["approval"],
                   "hallucination": ref_sv[variant_q4]["retrieved151"]["hallucination"]},
            "bf16": {"approval": ref_sv[variant_bf16]["retrieved151"]["approval"],
                     "hallucination": ref_sv[variant_bf16]["retrieved151"]["hallucination"]},
        }
    hybrid = {
        "A_hibrido": {
            "q4": {**agg_runs("tools_r128_hibrido_q4"), **decision_reuse("tools_r128_hibrido_q4")},
            "bf16": {**agg_runs("tools_r128_hibrido_bf16"), **decision_reuse("tools_r128_hibrido_bf16")},
        },
        "B_tools_puro": {
            "q4": {**agg_runs("tools_r128_pure_q4"), **decision_reuse("tools_r128_pure_q4")},
            "bf16": {**agg_runs("tools_r128_pure_bf16"), **decision_reuse("tools_r128_pure_bf16")},
        },
        "C_pipeline_fixo_sem_tool": cfgC(),
        "note": "C = sft_1_2b_r64 + busca v4 SEMPRE (retrieved_pack_151, R@2 51%); reuso de sft_1_2b/consolidation.json.",
    }

    # ---- MEDIÇÃO 4: decomposição da alucinação ----
    halluc = hallucination_decomposition({
        "A_hibrido_q4": "tools_r128_hibrido_q4",
        "A_hibrido_bf16": "tools_r128_hibrido_bf16",
        "B_tools_puro_q4": "tools_r128_pure_q4",
        "B_tools_puro_bf16": "tools_r128_pure_bf16",
    })

    return {
        "task": "poc-tools-oraculo",
        "checkpoint": "tools_1_2b_r128 (melhor do tools_v1; ranks quase indiferentes na barra de erro)",
        "regua": "bench/regua (ruler + judge_regua 27B) + juiz de recusa, limiar 0.5",
        "judge": "vLLM 27B 10.100.0.111:8005 (roteável deste host)",
        "medicao1_oraculo": oracle,
        "medicao2_recusa": refusal,
        "medicao3_hibrido": hybrid,
        "medicao4_alucinacao": halluc,
    }


def main() -> None:
    out = build()
    (DATA / "consolidation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
