#!/usr/bin/env python3
"""
score_tools.py — métricas de tool-calling do F4 a partir dos traces do
run_dialog_harness (receita tools_v1 adaptada ao multihop).

Métricas (por tag):
  decision : F1 de "chamou vs devia chamar" (positivos = perguntas normativas;
             negativos = saudação/fora de escopo) + % sintaxe válida.
  reuse    : % que respondeu SEM nova chamada (correto) + qualidade da resposta
             (juiz 27B: cobre o gabarito do follow-up?).
  switch   : % que fez NOVA chamada no 2º turno (correto) e a busca recuperou
             chunk da NR nova.
  refusal  : % recusa correta (looks_like_refusal) após busca que erra;
             alucinação = respondeu confiante sem recusa.
  multihop : % de cadeias com: 2ª chamada feita E argumento-2 != argumento-1
             E hop2 recuperou o chunk-ouro (containment real) E a síntese
             final usa as DUAS evidências (juiz 27B).
  (global) : % sintaxe válida sobre todas as chamadas emitidas em todas as suites.

Saída: data/scored/tools_{tag}.json

Uso: classifier/.venv/bin/python bench/lfm3_f4/eval/score_tools.py --tag f4-r32-ep2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
F4 = HERE.parent
REPO = F4.parents[1]
sys.path.insert(0, str(F4))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))

import common_f4 as C  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402

EVAL = F4 / "data" / "eval"
SCORED = F4 / "data" / "scored"

JUDGE_SYS = ("Você é um avaliador rigoroso de respostas sobre Normas "
             "Regulamentadoras. Responda apenas com o JSON pedido.")

PROMPT_COVER = """PERGUNTA:
{question}

RESPOSTA DE REFERÊNCIA (gabarito):
{gold}

RESPOSTA DO MODELO:
{answer}

A resposta do modelo contém a informação central do gabarito (mesmo dado
normativo, mesmo número/condição quando houver)? Pense e emita APENAS:
{{"contem": true/false}}"""

PROMPT_TWO_EVIDENCE = """TRECHO A ({src_a}):
{text_a}

TRECHO B ({src_b}):
{text_b}

PERGUNTA:
{question}

RESPOSTA DO MODELO:
{answer}

Avalie: (1) a resposta usa informação que SÓ está no trecho A? (2) usa
informação que SÓ está no trecho B? (3) contém a resposta correta ao que foi
perguntado, fiel aos trechos? Pense e emita APENAS:
{{"usa_a": true/false, "usa_b": true/false, "correta": true/false}}"""


def read_trace(tag: str, suite: str) -> List[Dict[str, Any]]:
    return C.read_jsonl(EVAL / f"trace_{tag}_{suite}.jsonl")


def pct(x: float) -> float:
    return round(100 * x, 1)


def score_decision(tag: str) -> Dict[str, Any]:
    suite = {s["id"]: s for s in C.read_jsonl(EVAL / "suite_decision.jsonl")}
    tp = fp = fn = tn = 0
    for r in read_trace(tag, "decision"):
        should = suite[r["id"]]["should_call"]
        called = r["n_calls"] > 0
        if should and called:
            tp += 1
        elif should:
            fn += 1
        elif called:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": pct(prec), "recall": pct(rec), "f1": pct(f1)}


def score_reuse(tag: str, workers: int) -> Dict[str, Any]:
    suite = {s["id"]: s for s in C.read_jsonl(EVAL / "suite_reuse.jsonl")}
    rows = read_trace(tag, "reuse")
    no_call = [r for r in rows if r["n_calls"] == 0]
    # qualidade: cobre o gabarito do follow-up? (nos que não chamaram)
    def worker(r):
        s = suite[r["id"]]
        v = C.gen_json(PROMPT_COVER.format(question=s["followup"],
                                           gold=s["gold_answer"],
                                           answer=r["final"]),
                       system=JUDGE_SYS, temperature=0.1, max_tokens=1200)
        return {"contem": bool(v.get("contem")) if isinstance(v, dict) else False}

    res = C.parallel_map(worker, rows, max_workers=workers,
                         checkpoint_path=SCORED / f"reuse_judge_{tag}.ckpt.json",
                         key_fn=lambda r: r["id"])
    ok_ans = sum(1 for r in rows
                 if isinstance(res.get(r["id"]), dict) and res[r["id"]].get("contem"))
    return {"n": len(rows),
            "reuse_correct_pct": pct(len(no_call) / len(rows)) if rows else 0,
            "answer_covers_gold_pct": pct(ok_ans / len(rows)) if rows else 0}


def score_switch(tag: str) -> Dict[str, Any]:
    suite = {s["id"]: s for s in C.read_jsonl(EVAL / "suite_switch.jsonl")}
    chunks_db = C.load_chunks()
    rows = read_trace(tag, "switch")
    called = [r for r in rows if r["n_calls"] > 0]
    nr_hit = 0
    for r in called:
        nr2 = suite[r["id"]]["nr2"]
        ids = [i for c in r["calls"] for i in c["chunk_ids"]]
        docs = {str(chunks_db[i]["doc"]).lower() for i in ids if i in chunks_db}
        if f"nr-{nr2:02d}" in docs or f"nr-{nr2}" in docs:
            nr_hit += 1
    return {"n": len(rows),
            "new_call_pct": pct(len(called) / len(rows)) if rows else 0,
            "new_call_nr_hit_pct": pct(nr_hit / len(rows)) if rows else 0}


def score_refusal(tag: str) -> Dict[str, Any]:
    rows = read_trace(tag, "refusal")
    called = [r for r in rows if r["n_calls"] > 0]
    refused = [r for r in rows if looks_like_refusal(r["final"])]
    halluc = [r for r in rows if r["n_calls"] > 0
              and not looks_like_refusal(r["final"])]
    return {"n": len(rows), "called_pct": pct(len(called) / len(rows)) if rows else 0,
            "refusal_correct_pct": pct(len(refused) / len(rows)) if rows else 0,
            "answered_anyway_pct": pct(len(halluc) / len(rows)) if rows else 0}


def score_multihop(tag: str, workers: int) -> Dict[str, Any]:
    suite = {s["id"]: s for s in C.read_jsonl(EVAL / "suite_multihop.jsonl")}
    assigns = {a["qa_id"]: a for a in C.load_assignments()}
    units = C.load_units()
    chunks_db = C.load_chunks()
    rows = read_trace(tag, "multihop")

    def norm(s: str) -> str:
        return " ".join(C.norm_tokens(s or ""))

    def chain(r) -> Dict[str, Any]:
        s = suite[r["id"]]
        a = assigns[s["qa_id"]]
        u = units[s["unit_id"]]
        fato_tokens = set(C.norm_tokens(u["fato"]))
        # a 1ª chamada recebeu o hop1 congelado; conta-se a partir da 2ª
        second = r["calls"][1] if len(r["calls"]) >= 2 else None
        second_made = second is not None
        arg_diff = bool(second and norm(second["consulta"]) != norm(r["calls"][0]["consulta"]))
        gold_hop2 = False
        gold_chunk = None
        if second:
            for cid in second["chunk_ids"]:
                c = chunks_db.get(cid)
                if not c:
                    continue
                doc = str(c["doc"]).lower()
                if doc in (f"nr-{a['nr']:02d}", f"nr-{a['nr']}"):
                    ft = fato_tokens
                    ct = set(C.norm_tokens(c["text"]))
                    if ft and len(ft & ct) / len(ft) >= 0.70:
                        gold_hop2 = True
                        gold_chunk = c
                        break
        return {"second_made": second_made, "arg_diff": arg_diff,
                "gold_hop2": gold_hop2,
                "gold_chunk_id": gold_chunk["id"] if gold_chunk else None}

    mech = {r["id"]: chain(r) for r in rows}

    # juiz: síntese usa as duas evidências (só nas cadeias mecânicas completas)
    def judge(r):
        m = mech[r["id"]]
        if not (m["second_made"] and m["arg_diff"] and m["gold_hop2"]):
            return {"skip": True}
        s = suite[r["id"]]
        a = assigns[s["qa_id"]]
        anchor = a["anchor_passage"]
        gold = C.load_chunks()[m["gold_chunk_id"]]
        v = C.gen_json(PROMPT_TWO_EVIDENCE.format(
            src_a=f"{anchor['doc']} {anchor['section']}", text_a=anchor["text"][:1500],
            src_b=f"{gold['doc']} {gold['section']}", text_b=gold["text"][:1500],
            question=s["question"], answer=r["final"]),
            system=JUDGE_SYS, temperature=0.1, max_tokens=1500)
        if not isinstance(v, dict):
            return {"usa_a": False, "usa_b": False, "correta": False}
        return {"usa_a": bool(v.get("usa_a")), "usa_b": bool(v.get("usa_b")),
                "correta": bool(v.get("correta"))}

    res = C.parallel_map(judge, rows, max_workers=workers,
                         checkpoint_path=SCORED / f"mh_judge_{tag}.ckpt.json",
                         key_fn=lambda r: r["id"])

    n = len(rows)
    n2 = sum(1 for m in mech.values() if m["second_made"])
    nd = sum(1 for m in mech.values() if m["second_made"] and m["arg_diff"])
    ng = sum(1 for m in mech.values()
             if m["second_made"] and m["arg_diff"] and m["gold_hop2"])
    full = 0
    for r in rows:
        j = res.get(r["id"])
        m = mech[r["id"]]
        if (m["second_made"] and m["arg_diff"] and m["gold_hop2"]
                and isinstance(j, dict) and j.get("usa_b") and j.get("correta")):
            full += 1
    return {"n": n, "second_call_pct": pct(n2 / n) if n else 0,
            "arg_diff_pct": pct(nd / n) if n else 0,
            "gold_hop2_pct": pct(ng / n) if n else 0,
            "full_chain_pct": pct(full / n) if n else 0}


def score_syntax(tag: str, suites: List[str]) -> Dict[str, Any]:
    total = fails = 0
    for s in suites:
        p = EVAL / f"trace_{tag}_{s}.jsonl"
        if not p.exists():
            continue
        for r in C.read_jsonl(p):
            total += r["n_calls"]
            if r.get("syntactic_fail"):
                fails += 1
    return {"calls": total, "syntax_fails": fails,
            "syntax_valid_pct": pct((total - fails) / total) if total else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    SCORED.mkdir(parents=True, exist_ok=True)

    out = {"tag": args.tag}
    out["decision"] = score_decision(args.tag)
    out["reuse"] = score_reuse(args.tag, args.workers)
    out["switch"] = score_switch(args.tag)
    out["refusal"] = score_refusal(args.tag)
    out["multihop"] = score_multihop(args.tag, args.workers)
    out["syntax"] = score_syntax(args.tag,
                                 ["decision", "reuse", "switch", "refusal", "multihop"])
    path = SCORED / f"tools_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
