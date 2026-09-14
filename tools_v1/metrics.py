#!/usr/bin/env python3
"""
metrics.py — PASSO 4: as métricas NOVAS do tool-calling (as antigas não medem isto).

Funções puras que consomem os artefatos de inferência (infer_tools) e as suites
(build_eval_sets) e computam:

  a) DECISÃO DE CHAMAR: matriz de confusão chamou-vs-devia-chamar + precision/recall/F1.
  b) VALIDADE SINTÁTICA: % das chamadas que parseiam na sintaxe do template.
  c) QUALIDADE DOS ARGUMENTOS (central): recall@k no índice v4 de (i) pergunta crua,
     (ii) consulta do modelo, (iii) consulta do 27B. O delta (ii)-(i) é o valor do treino.
  d) REUSO no multiturno: % de turnos que deviam reusar e reusaram; e o inverso (chamou à toa).

As de aprovação/alucinação/recusa reusam a régua honesta (bench/regua) e os juízes de recusa
(sft_v2) — implementadas em score_e2e.py.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import retrieval_check as RC


# ---------------------------------------------------------------------------
# a) DECISÃO DE CHAMAR
# ---------------------------------------------------------------------------
def confusion_decision(preds: List[bool], golds: List[bool]) -> Dict[str, Any]:
    """Classe positiva = 'DEVERIA chamar'. Retorna TP/FP/FN/TN + precision/recall/F1."""
    tp = fp = fn = tn = 0
    for p, g in zip(preds, golds):
        if g and p:
            tp += 1
        elif g and not p:
            fn += 1
        elif not g and p:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
            "precision": round(100 * prec, 1), "recall": round(100 * rec, 1),
            "f1": round(100 * f1, 1), "accuracy": round(100 * acc, 1),
            "call_when_should_not_pct": round(100 * fp / (fp + tn), 1) if (fp + tn) else 0.0,
            "miss_when_should_pct": round(100 * fn / (tp + fn), 1) if (tp + fn) else 0.0}


# ---------------------------------------------------------------------------
# b) VALIDADE SINTÁTICA
# ---------------------------------------------------------------------------
def syntactic_validity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """% das chamadas emitidas que parseiam corretamente."""
    called = [r for r in results if r.get("called_tool")]
    ok = sum(1 for r in called if r.get("syntactic_ok"))
    return {"n_called": len(called), "n_valid": ok,
            "valid_pct": round(100 * ok / len(called), 1) if called else 100.0}


# ---------------------------------------------------------------------------
# c) QUALIDADE DOS ARGUMENTOS (recall@k no índice v4)
# ---------------------------------------------------------------------------
def recall_of_queries(items: List[Dict[str, Any]], query_key: str, topks=(1, 2, 5)
                      ) -> Dict[str, Any]:
    """Recall@k no índice v4 usando items[i][query_key] como consulta.

    Cada item precisa de gold_chunk_id (+ relevant_ids). Itens sem consulta (None/vazio,
    ex.: o modelo não chamou) contam como MISS (não recuperou nada) — honesto: se não buscou,
    não achou.
    """
    hits = {k: 0 for k in topks}
    n = len(items)
    detail = []
    for it in items:
        q = it.get(query_key)
        gid = it.get("gold_chunk_id", -1)
        rel = it.get("relevant_ids") or ([gid] if gid >= 0 else [])
        rank = None
        if q:
            rank = RC.gold_rank(q, gid, limit=max(topks), relevant_ids=rel)
        for k in topks:
            if rank is not None and rank <= k:
                hits[k] += 1
        detail.append({"id": it.get("id"), "rank": rank, "query": q})
    return {"n": n, **{f"recall_at_{k}": round(100 * hits[k] / n, 1) for k in topks},
            "detail": detail}


def arg_quality_table(raw_res: Dict[str, Any], model_res: Dict[str, Any],
                      llm27_res: Dict[str, Any]) -> Dict[str, Any]:
    """Junta os três recalls e computa o delta modelo-crua (o valor que o treino agrega)."""
    def d(a, b, k):
        return round(a[f"recall_at_{k}"] - b[f"recall_at_{k}"], 1)
    return {
        "raw_question": {k: raw_res[k] for k in raw_res if k != "detail"},
        "model_rewrite": {k: model_res[k] for k in model_res if k != "detail"},
        "llm27b_rewrite": {k: llm27_res[k] for k in llm27_res if k != "detail"},
        "delta_model_minus_raw": {f"recall_at_{k}": d(model_res, raw_res, k) for k in (1, 2, 5)},
        "delta_model_minus_27b": {f"recall_at_{k}": d(model_res, llm27_res, k) for k in (1, 2, 5)},
    }


# ---------------------------------------------------------------------------
# d) REUSO no multiturno
# ---------------------------------------------------------------------------
def reuse_metrics(dialog_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sobre diálogos de 2 turnos (eval_reuse): mede acerto de reuso e de nova-busca no t2.

    dialog_results[i] = {"item": <suite item>, "turns": [turn1_result, turn2_result]}.
    - reuse cases (should_call_turn2=False): CORRETO se o t2 NÃO chamou (reusou o contexto).
    - newtopic cases (should_call_turn2=True): CORRETO se o t2 CHAMOU de novo.
    """
    reuse_total = reuse_ok = 0        # deveria reusar (não chamar) e reusou
    newtopic_total = newtopic_ok = 0  # deveria chamar de novo e chamou
    called_when_reuse = 0             # chamou à toa (deveria reusar)
    for dr in dialog_results:
        item = dr["item"]
        t2 = dr["turns"][1]
        called2 = bool(t2.get("called_tool"))
        if item.get("should_call_turn2") is False:
            reuse_total += 1
            if not called2:
                reuse_ok += 1
            else:
                called_when_reuse += 1
        elif item.get("should_call_turn2") is True:
            newtopic_total += 1
            if called2:
                newtopic_ok += 1
    return {
        "reuse_cases": reuse_total,
        "reuse_correct_pct": round(100 * reuse_ok / reuse_total, 1) if reuse_total else 0.0,
        "called_when_should_reuse_pct":
            round(100 * called_when_reuse / reuse_total, 1) if reuse_total else 0.0,
        "newtopic_cases": newtopic_total,
        "newtopic_correct_pct":
            round(100 * newtopic_ok / newtopic_total, 1) if newtopic_total else 0.0,
    }
