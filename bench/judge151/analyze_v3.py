#!/usr/bin/env python3
"""
analyze_v3.py — Tabela 4-células [retrieval antigo × v3] × [1.2B × 2.6B] + taxa de conversão
chunk-certo -> resposta-aprovada, para a PARTE 1 do brief.

As 4 células:
  (A) antigo × 1.2B  -> do judge-151 anterior (data/judge_evaluations.json, config 'lfm1.2b')
                        OU da rodada de paridade de prompt 'old_lfm1.2b_v1' (v1_rigido) desta task.
  (B) antigo × 2.6B  -> do judge-151 anterior (config 'lfm2.6b_noth', thinking-OFF v1_rigido é
                        'lfm2.6b_noth_v1' se julgado; senão 'lfm2.6b_noth').
  (C) v3     × 1.2B  -> desta task (config 'v3_lfm1.2b').
  (D) v3     × 2.6B  -> desta task (config 'v3_lfm2.6b_noth').

Para paridade honesta de PROMPT (produção usa v1_rigido), preferimos as células que usaram
v1_rigido; caímos para o base do judge-151 quando a variante v1 não foi julgada, e anotamos.

Métricas por célula: global (0.35Fac+0.30Fid+0.20Cit+0.15PT), gate-pass%, e conversão
chunk-certo (retrieval_hit) -> aprovada (pass_gate).

Lê:
  data/judge_evaluations_v3.json  (as 3 configs novas desta task)
  data/judge_evaluations.json     (o judge-151 anterior; opcional, p/ as células antigas)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent


def _load_eval_models(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for mk, mv in d["models"].items():
        out[mk] = mv["modes"]["classic_rag"]["items"]
    return out


def _load_responses(cfg: str) -> Dict[str, Dict[str, Any]]:
    path = _HERE / "data" / f"responses_{cfg}.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in d["items"]}


def cell_metrics(items: List[Dict[str, Any]], resp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    n = len(items)
    fac = sum(i["metrics"]["acerto_factual"] for i in items) / n
    fid = sum(i["metrics"]["fidelidade_contexto"] for i in items) / n
    pt = sum(i["metrics"]["qualidade_pt"] for i in items) / n
    cit = sum(i["metrics"]["citacao_fonte"] for i in items) / n
    glob = fac * 0.35 + fid * 0.30 + cit * 0.20 + pt * 0.15
    gate = 100 * sum(1 for i in items if i["pass_gate"]) / n
    # Conversão chunk-certo -> aprovada (usa retrieval_hit das respostas).
    ok = [i for i in items if resp.get(i["item_id"], {}).get("retrieval_hit")]
    n_ok = len(ok)
    conv = 100 * sum(1 for i in ok if i["pass_gate"]) / n_ok if n_ok else 0.0
    recall = 100 * n_ok / n
    return {"n": n, "fac": round(fac, 3), "fid": round(fid, 3), "pt": round(pt, 3),
            "cit": round(cit, 3), "global": round(glob, 3), "gate_pct": round(gate, 1),
            "recall_top2_pct": round(recall, 1), "n_retrieval_ok": n_ok,
            "conv_chunkcerto_aprovada_pct": round(conv, 1)}


def pick(evals: Dict, resp_cache: Dict, prefer: List[str]) -> Optional[Dict[str, Any]]:
    """Escolhe a 1ª config disponível em `prefer`; devolve métricas + qual config usou."""
    for cfg in prefer:
        if cfg in evals:
            resp = resp_cache.get(cfg) or _load_responses(cfg)
            m = cell_metrics(evals[cfg], resp)
            if m:
                m["config_used"] = cfg
                return m
    return None


def main() -> None:
    ev_v3 = _load_eval_models(_HERE / "data" / "judge_evaluations_v3.json")
    ev_old = _load_eval_models(_HERE / "data" / "judge_evaluations.json")
    all_ev = {**ev_old, **ev_v3}
    resp_cache = {cfg: _load_responses(cfg) for cfg in all_ev}

    # As 4 células (preferindo prompt v1_rigido p/ paridade honesta).
    cells = {
        "old_x_1.2b": pick(all_ev, resp_cache, ["old_lfm1.2b_v1", "lfm1.2b"]),
        "old_x_2.6b": pick(all_ev, resp_cache, ["lfm2.6b_noth_v1", "lfm2.6b_noth"]),
        "v3_x_1.2b": pick(all_ev, resp_cache, ["v3_lfm1.2b"]),
        "v3_x_2.6b": pick(all_ev, resp_cache, ["v3_lfm2.6b_noth"]),
    }

    report = {"four_cells": cells,
              "note": "gate-pass = fac>=3 & fid>=3 & cit>=3 & global>=3.5; "
                      "conv = % das perguntas com chunk-ouro no top-2 que viraram resposta aprovada."}
    (_HERE / "data" / "analysis_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 92)
    print("TABELA 4-CÉLULAS — [retrieval antigo × v3] × [1.2B × 2.6B]")
    print("=" * 92)
    hdr = f"{'célula':12} {'config':22} {'n':>4} {'R@2%':>6} {'Global':>7} {'Gate%':>6} {'conv%(ok->ok)':>14}"
    print(hdr)
    print("-" * 92)
    for name, m in cells.items():
        if not m:
            print(f"{name:12} {'(faltando — rode o juiz)':22}")
            continue
        print(f"{name:12} {m['config_used']:22} {m['n']:4d} {m['recall_top2_pct']:6.1f} "
              f"{m['global']:7.2f} {m['gate_pct']:6.1f} {m['conv_chunkcerto_aprovada_pct']:13.1f}%")

    # 2x2 gate-pass e a pergunta central: o 2.6b mantém >=2x o gate do 1.2b sobre a v3?
    if cells["v3_x_1.2b"] and cells["v3_x_2.6b"]:
        g1 = cells["v3_x_1.2b"]["gate_pct"]; g2 = cells["v3_x_2.6b"]["gate_pct"]
        ratio = g2 / g1 if g1 else float("inf")
        print("\n" + "=" * 92)
        print(f"DECISÃO PARTE 1: gate-pass v3×2.6B / v3×1.2B = {g2:.1f}% / {g1:.1f}% = {ratio:.2f}×")
        print(f"  -> 2.6B mantém >= 2× o gate do 1.2B sobre a v3? {'SIM' if ratio >= 2.0 else 'NÃO'}")
        print("  (SIM => avaliar destravar thinking-OFF no engine nativo; NÃO => fica no 1.2B.)")
    print(f"\nSalvo em {(_HERE/'data'/'analysis_v3.json')}")


if __name__ == "__main__":
    main()
