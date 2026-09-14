#!/usr/bin/env python3
"""
consolidate_2_6b.py — TABELA MESTRE + as medições do 2.6B tool-calling.

Junta:
  - a bateria NOVA do 2.6B (decisão/args/reuso de run_eval_2_6b; e2e de score_e2e_2_6b;
    oráculo/híbrido/puro/recusa de harness_oraculo_2_6b + régua sft_v2/score_v2), média±desvio
    dos 3 runs onde houver;
  - as REFERÊNCIAS já medidas na POC (reuso, não recomputa): 1.2B base/sft/tool
    (sft_1_2b + tools_oraculo), 2.6B sft_v3 (sft_1_2b/consolidation), 27B (teto).

Emite:
  1. tabela mestre única (1.2B base | 1.2B sft | 1.2B tool | 2.6B sft_v3 | 2.6B tool | 27B),
     todas as métricas, bf16 E Q4 lado a lado (preferência do capitão);
  2. curva de saturação do 2.6B (degraus 750/1500/3000/4236) vs a do 1.2B — responde se o 2.6B
     satura no mesmo ponto;
  3. medições 1-4 (oráculo/recusa/híbrido/decomposição da alucinação) do 2.6B tool.

Higiene: idempotente, não-interativo, procedência. Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DATA = _HERE / "data"
LABEL = "tools_2_6b_r64"  # candidato final default (ajustável via --label)

REF_12B = json.loads((_ROOT / "sft_1_2b" / "data" / "consolidation.json").read_text("utf-8"))
REF_TOOLS_12B = json.loads((_ROOT / "tools_oraculo" / "data" / "consolidation.json")
                           .read_text("utf-8"))


def _mean_std(xs: List[Optional[float]]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(statistics.mean(xs), 1),
            "std": round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0, "n": len(xs)}


def _score(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"score_{label}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def _gen(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"gen_{label}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def _eval(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"eval_{label}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def _e2e(label: str) -> Optional[Dict[str, Any]]:
    p = DATA / f"e2e_{label}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def agg_runs(prefix: str, runs=(1, 2, 3)) -> Dict[str, Any]:
    """Média±desvio de aprovação/cobertura/alucinação de um score prefixo (régua sft_v2)."""
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
            "coverage": _mean_std(cov), "avg_tok": _mean_std(tok), "n_runs": len(appr)}


def decision_reuse_from_gen(prefix: str, runs=(1, 2, 3)) -> Dict[str, Any]:
    called, got_gold = [], []
    for r in runs:
        g = _gen(f"{prefix}_run{r}")
        if not g:
            continue
        items = [it for it in g["items"] if not it.get("__error__")]
        n = len(items)
        called.append(100 * sum(1 for it in items if it.get("called_tool")) / max(1, n))
        cg = [it for it in items if it.get("called_tool")]
        got_gold.append(100 * sum(1 for it in cg if it.get("got_gold")) / max(1, len(cg)))
    return {"called_pct": _mean_std(called), "got_gold_when_called_pct": _mean_std(got_gold)}


def hallucination_decomposition(prefixes: Dict[str, str], runs=(1, 2, 3)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cfg, prefix in prefixes.items():
        cond = {"gold_ok": {"n": 0, "hall": 0}, "gold_wrong": {"n": 0, "hall": 0},
                "no_call": {"n": 0, "hall": 0}}
        for r in runs:
            g = _gen(f"{prefix}_run{r}")
            s = _score(f"{prefix}_run{r}")
            if not g or not s:
                continue
            hall_by_id = {it["item_id"]: bool(it.get("hallucination")) for it in s["items"]}
            for it in g["items"]:
                if it.get("__error__") or it["item_id"] not in hall_by_id:
                    continue
                h = hall_by_id[it["item_id"]]
                c = "no_call" if not it.get("called_tool") else (
                    "gold_ok" if it.get("got_gold") else "gold_wrong")
                cond[c]["n"] += 1
                cond[c]["hall"] += int(h)
        total_hall = sum(cond[c]["hall"] for c in cond)
        total_n = sum(cond[c]["n"] for c in cond)
        detail = {c: {"n": v["n"], "hall_count": v["hall"],
                      "hall_rate_pct": round(100 * v["hall"] / v["n"], 1) if v["n"] else None,
                      "share_pct": round(100 * v["hall"] / total_hall, 1) if total_hall else None}
                  for c, v in cond.items()}
        out[cfg] = {"total_items": total_n, "total_hall": total_hall, "by_condition": detail}
    return out


def saturation_curve(rank=32, steps=(750, 1500, 3000)) -> Dict[str, Any]:
    """Curva de saturação do 2.6B (decisão/argumento/reuso por degrau) vs a do 1.2B."""
    curve = {}
    for s in list(steps) + ["full"]:
        lbl = f"tools_2_6b_step{s}_r{rank}" if s != "full" else f"tools_2_6b_r{rank}"
        ev = _eval(lbl)
        if not ev:
            continue
        dec = ev.get("decision", {}).get("confusion", {})
        args = ev.get("args", {})
        model_r2 = args.get("model", {}).get("recall_at_2") if args.get("model") else None
        reuse = ev.get("reuse", {}).get("metrics", {})
        curve[str(s)] = {
            "dec_f1": dec.get("f1"), "model_r2": model_r2,
            "reuse_correct_pct": reuse.get("reuse_correct_pct"),
            "newtopic_correct_pct": reuse.get("newtopic_correct_pct"),
            "syntactic_pct": ev.get("decision", {}).get("syntactic", {}).get("valid_pct"),
        }
    return curve


def arg_quality(label: str) -> Dict[str, Any]:
    """Qualidade do argumento (recall@k): fala crua vs consulta do 2.6B vs 27B."""
    ev = _eval(label)
    if not ev or "args" not in ev:
        return {}
    t = ev["args"].get("table", {})
    return t


def build(label: str) -> Dict[str, Any]:
    ref = REF_12B["summary_by_variant"]
    tref = REF_TOOLS_12B  # tools_oraculo (1.2B tool r128)

    # ---- MEDIÇÃO 1: oráculo (2.6B tool) ----
    oracle = {
        "tools_2_6b_q4": agg_runs(f"{label}_oracle_q4"),
        "tools_2_6b_bf16": agg_runs(f"{label}_oracle_bf16"),
    }

    # ---- MEDIÇÃO 2: recusa ----
    refusal = {}
    for q in ("bf16", "q4"):
        p = DATA / f"refusal_confusion_{q}.json"
        if p.exists():
            refusal[f"tools_2_6b_{q}"] = list(json.loads(p.read_text("utf-8")).values())[0]

    # ---- MEDIÇÃO 3: híbrido ----
    hybrid = {}
    for q in ("q4", "bf16"):
        hybrid[q] = {
            "A_hibrido": {**agg_runs(f"{label}_hibrido_{q}"),
                          **decision_reuse_from_gen(f"{label}_hibrido_{q}")},
            "B_tools_puro": {**agg_runs(f"{label}_pure_{q}"),
                             **decision_reuse_from_gen(f"{label}_pure_{q}")},
        }

    # ---- MEDIÇÃO 4: decomposição da alucinação ----
    halluc = hallucination_decomposition({
        f"A_hibrido_q4": f"{label}_hibrido_q4",
        f"A_hibrido_bf16": f"{label}_hibrido_bf16",
        f"B_tools_puro_q4": f"{label}_pure_q4",
        f"B_tools_puro_bf16": f"{label}_pure_bf16",
    })

    # ---- decisão / argumento / reuso (candidato final, Q4) ----
    ev_q4 = _eval(f"{label}") or {}
    decision = ev_q4.get("decision", {}).get("confusion", {})
    syntactic = ev_q4.get("decision", {}).get("syntactic", {})
    reuse = ev_q4.get("reuse", {}).get("metrics", {})
    args_table = arg_quality(f"{label}")
    e2e = _e2e(f"{label}") or {}

    # ---- TABELA MESTRE ----
    def o(v):  # oráculo (approval/hall) de uma variante REF do 1.2B/2.6B/27b
        return {"approval": ref[v]["oracle151"]["approval"],
                "hallucination": ref[v]["oracle151"]["hallucination"]}

    def rt(v):
        return {"approval": ref[v]["retrieved151"]["approval"],
                "hallucination": ref[v]["retrieved151"]["hallucination"]}

    def f1(v):
        return ref[v]["refusal_classifier"]["f1_pct"]

    master = {
        # 1.2B base (QAD embarcado)
        "1.2B_base_qad_q4": {"oracle": o("base_qad_q4"), "retrieved": rt("base_qad_q4"),
                             "refusal_f1": f1("base_qad_q4"), "src": "sft_1_2b"},
        # 1.2B sft (sem tool)
        "1.2B_sft_r64_q4": {"oracle": o("sft_r64_q4"), "retrieved": rt("sft_r64_q4"),
                            "refusal_f1": f1("sft_r64_q4"), "src": "sft_1_2b"},
        "1.2B_sft_r64_bf16": {"oracle": o("sft_r64_bf16"), "retrieved": rt("sft_r64_bf16"),
                              "refusal_f1": f1("sft_r64_bf16"), "src": "sft_1_2b"},
        # 1.2B tool (r128) — do tools_oraculo
        "1.2B_tool_r128_q4": {
            "oracle": {"approval": tref["medicao1_oraculo"]["tools_r128_q4"]["approval"]["mean"],
                       "hallucination": tref["medicao1_oraculo"]["tools_r128_q4"]["hallucination"]["mean"]},
            "retrieved_hibrido": {"approval": tref["medicao3_hibrido"]["A_hibrido"]["q4"]["approval"]["mean"],
                                  "hallucination": tref["medicao3_hibrido"]["A_hibrido"]["q4"]["hallucination"]["mean"]},
            "refusal_f1": tref["medicao2_recusa"].get("tools_r128_q4", {}).get("f1_pct"),
            "src": "tools_oraculo"},
        "1.2B_tool_r128_bf16": {
            "oracle": {"approval": tref["medicao1_oraculo"]["tools_r128_bf16"]["approval"]["mean"],
                       "hallucination": tref["medicao1_oraculo"]["tools_r128_bf16"]["hallucination"]["mean"]},
            "retrieved_hibrido": {"approval": tref["medicao3_hibrido"]["A_hibrido"]["bf16"]["approval"]["mean"],
                                  "hallucination": tref["medicao3_hibrido"]["A_hibrido"]["bf16"]["hallucination"]["mean"]},
            "refusal_f1": tref["medicao2_recusa"].get("tools_r128_bf16", {}).get("f1_pct"),
            "src": "tools_oraculo"},
        # 2.6B sft_v3 (sem tool)
        "2.6B_sft_v3_r64_q4": {"oracle": o("v3_r64_q4"), "retrieved": rt("v3_r64_q4"),
                               "refusal_f1": f1("v3_r64_q4"), "src": "sft_v3"},
        "2.6B_sft_v3_r64_bf16": {"oracle": o("v3_r64_bf16"), "retrieved": rt("v3_r64_bf16"),
                                 "refusal_f1": f1("v3_r64_bf16"), "src": "sft_v3"},
        # 2.6B tool (ESTE) — preenchido pelas medições
        "2.6B_tool_q4": {
            "oracle": {"approval": oracle["tools_2_6b_q4"]["approval"]["mean"],
                       "hallucination": oracle["tools_2_6b_q4"]["hallucination"]["mean"]},
            "retrieved_hibrido": {"approval": hybrid.get("q4", {}).get("A_hibrido", {}).get("approval", {}).get("mean"),
                                  "hallucination": hybrid.get("q4", {}).get("A_hibrido", {}).get("hallucination", {}).get("mean")},
            "refusal_f1": refusal.get("tools_2_6b_q4", {}).get("f1_pct"),
            "src": "ESTE"},
        "2.6B_tool_bf16": {
            "oracle": {"approval": oracle["tools_2_6b_bf16"]["approval"]["mean"],
                       "hallucination": oracle["tools_2_6b_bf16"]["hallucination"]["mean"]},
            "retrieved_hibrido": {"approval": hybrid.get("bf16", {}).get("A_hibrido", {}).get("approval", {}).get("mean"),
                                  "hallucination": hybrid.get("bf16", {}).get("A_hibrido", {}).get("hallucination", {}).get("mean")},
            "refusal_f1": refusal.get("tools_2_6b_bf16", {}).get("f1_pct"),
            "src": "ESTE"},
        # 27B (teto)
        "27B_ceiling": {"oracle": o("27b"), "retrieved": rt("27b"), "refusal_f1": f1("27b"),
                        "src": "slm_oraculo"},
    }

    return {
        "task": "poc-tools-2.6b",
        "candidate_label": label,
        "regua": "bench/regua (ruler + judge_regua 27B), limiar 0.5, citação FORA do gate",
        "judge": "vLLM 27B 10.100.0.111:8005",
        "note_thinking": "2.6B é reasoning; toda a bateria roda thinking-OFF (render_jinja_2_6b), "
                         "corpo dos turnos byte-a-byte idêntico ao 1.2B (verify_2_6b.py).",
        "master_table": master,
        "medicao1_oraculo": oracle,
        "medicao2_recusa": refusal,
        "medicao3_hibrido": hybrid,
        "medicao4_alucinacao": halluc,
        "decisao": {"confusion": decision, "syntactic": syntactic, "reuse": reuse},
        "argumento_recall": args_table,
        "e2e_retrieval_real": {k: e2e.get(k) for k in
                               ("approval_pct", "hallucination_pct", "got_gold_pct",
                                "conv_goldcorrect_to_approved_pct") if k in e2e},
        "saturation_curve_2_6b": saturation_curve(),
        "saturation_curve_1_2b_ref": "ver tools_v1/README (dec satura ~1500, sintaxe ~750, "
                                     "argumento nunca melhora)",
    }


def main() -> None:
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else LABEL
    out = build(label)
    (DATA / "consolidation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(out["master_table"], ensure_ascii=False, indent=1))
    print(f"\n[ok] -> {DATA / 'consolidation.json'}")


if __name__ == "__main__":
    main()
