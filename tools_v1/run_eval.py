#!/usr/bin/env python3
"""
run_eval.py — roda UM candidato (GGUF via llama-server) nas suites e computa o PASSO 4.

Produz data/eval_<label>.json com:
  a) decisão de chamar (matriz + P/R/F1)   — suite eval_decision
  b) validade sintática                      — das chamadas emitidas na eval_decision
  c) qualidade dos argumentos (recall@k)     — suite eval_args (151), 3 fontes de consulta
  d) reuso no multiturno                      — suite eval_reuse

As de aprovação/alucinação/recusa (e2e) ficam em score_e2e.py (reusa a régua/juízes).

Fontes de consulta da métrica (c):
  - raw     : a pergunta crua (baseline);
  - model   : a consulta emitida pelo MODELO no 1º turno (via infer_tools.run_turn);
  - llm27b  : a consulta do 27B (gen_dialogs.gen_query) — cache reutilizável entre candidatos.

Higiene: paralelo, checkpoint por suite, procedência. Comentários PT-BR; código em inglês.

Uso (classifier/.venv; modelo via llama-server tunelado):
  ../classifier/.venv/bin/python run_eval.py --url http://127.0.0.1:8500 --label tools_r32 \
      --topk 2 [--deterministic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import common_tools as C
import metrics as M
from gen_dialogs import gen_query
from infer_tools import DET_SAMPLING, LFM_SAMPLING, run_turn

_HERE = Path(__file__).resolve().parent


def _load(rel: str) -> Dict[str, Any]:
    return json.loads((_HERE / rel).read_text(encoding="utf-8"))


def eval_decision(url: str, sampling, topk: int, workers: int) -> Dict[str, Any]:
    """Roda a suite de decisão: cada item é 1 turno; captura chamou/sintaxe."""
    suite = _load("data/eval_decision.json")["items"]
    results: List[Dict[str, Any]] = [None] * len(suite)

    def work(i_item):
        i, it = i_item
        r = run_turn(url, [], it["question"], sampling, max_tokens=256, topk=topk)
        return i, {"id": it["id"], "should_call": it["should_call"],
                   "called_tool": r["called_tool"], "syntactic_ok": r["syntactic_ok"],
                   "consulta": r["consulta"], "nr": r["nr"],
                   "captain_case": it.get("captain_case", False),
                   "categoria": it.get("categoria")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, (i, it)) for i, it in enumerate(suite)]
        for fut in as_completed(futs):
            i, r = fut.result()
            results[i] = r
    results = [r for r in results if r]
    preds = [r["called_tool"] for r in results]
    golds = [r["should_call"] for r in results]
    conf = M.confusion_decision(preds, golds)
    syn = M.syntactic_validity(results)
    return {"confusion": conf, "syntactic": syn, "per_item": results}


def eval_args(url: str, sampling, topk: int, workers: int,
              cache_27b: Optional[Path], model_rewrite: bool = True) -> Dict[str, Any]:
    """Qualidade dos argumentos nas 151 (só medição). 3 fontes de consulta -> recall@k v4."""
    items = _load("data/eval_args.json")["items"]

    # consulta do MODELO (1º turno) — captura o campo consulta que ele emitiria
    model_items = []
    if model_rewrite:
        def work(it):
            r = run_turn(url, [], it["question"], sampling, max_tokens=192, topk=topk)
            return {**it, "model_query": r["consulta"] if r["called_tool"] else None,
                    "model_called": r["called_tool"]}
        model_items = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                model_items[futs[fut]] = fut.result()

    # consulta do 27B (cacheável entre candidatos)
    cache: Dict[str, Any] = {}
    if cache_27b and cache_27b.exists():
        cache = json.loads(cache_27b.read_text(encoding="utf-8"))
    llm_items = []
    missing = [it for it in items if it["id"] not in cache]
    if missing:
        def work27(it):
            qr = gen_query(it["question"], C.DEFAULT_VLLM_URL, C.DEFAULT_VLLM_MODEL)
            return it["id"], (qr[0] if qr else None)
        with ThreadPoolExecutor(max_workers=min(24, workers)) as ex:
            futs = [ex.submit(work27, it) for it in missing]
            for fut in as_completed(futs):
                cid, q = fut.result()
                cache[cid] = q
        if cache_27b:
            cache_27b.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    for it in items:
        llm_items.append({**it, "llm27b_query": cache.get(it["id"])})

    raw_res = M.recall_of_queries(items, "question")
    llm_res = M.recall_of_queries(llm_items, "llm27b_query")
    out = {"raw": raw_res, "llm27b": llm_res}
    if model_rewrite:
        model_res = M.recall_of_queries(model_items, "model_query")
        out["model"] = model_res
        out["table"] = M.arg_quality_table(raw_res, model_res, llm_res)
    else:
        out["table"] = {"raw_question": {k: raw_res[k] for k in raw_res if k != "detail"},
                        "llm27b_rewrite": {k: llm_res[k] for k in llm_res if k != "detail"}}
    return out


def eval_reuse(url: str, sampling, topk: int, workers: int) -> Dict[str, Any]:
    """Roda os diálogos de 2 turnos da suite de reuso e mede reuso/nova-busca."""
    suite = _load("data/eval_reuse.json")["items"]
    dialog_results: List[Dict[str, Any]] = [None] * len(suite)

    def work(i_item):
        i, it = i_item
        history: List[Dict[str, Any]] = []
        turns_out = []
        # roda os 2 turnos mantendo o histórico
        from infer_tools import run_turn as rt
        for ut in it["turns"]:
            r = rt(url, history, ut, sampling, max_tokens=256, topk=topk)
            history = r.pop("history")
            turns_out.append({"called_tool": r["called_tool"], "consulta": r["consulta"],
                              "nr": r["nr"], "answer": r["answer"][:300]})
        return i, {"item": it, "turns": turns_out}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, (i, it)) for i, it in enumerate(suite)]
        for fut in as_completed(futs):
            i, r = fut.result()
            dialog_results[i] = r
    dialog_results = [r for r in dialog_results if r]
    rm = M.reuse_metrics(dialog_results)
    return {"metrics": rm, "per_dialog": [
        {"id": d["item"]["id"], "should_call_turn2": d["item"].get("should_call_turn2"),
         "t2_called": d["turns"][1]["called_tool"]} for d in dialog_results]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="llama-server do candidato (/completion)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-model-rewrite", action="store_true",
                    help="pula a consulta do modelo (p/ candidatos que não são o aluno)")
    ap.add_argument("--suites", default="decision,args,reuse")
    args = ap.parse_args()

    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING
    url = args.url.rstrip("/")
    suites = args.suites.split(",")
    out: Dict[str, Any] = {"label": args.label, "url": url, "topk": args.topk,
                           "deterministic": args.deterministic,
                           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if "decision" in suites:
        print(f"[{args.label}] decisão...", flush=True)
        out["decision"] = eval_decision(url, sampling, args.topk, args.workers)
        print(f"  {out['decision']['confusion']}", flush=True)
        print(f"  sintaxe {out['decision']['syntactic']}", flush=True)
    if "args" in suites:
        print(f"[{args.label}] argumentos (151)...", flush=True)
        out["args"] = eval_args(url, sampling, args.topk, args.workers,
                                cache_27b=_HERE / "data" / "cache_27b_queries.json",
                                model_rewrite=not args.no_model_rewrite)
        print(f"  {out['args']['table']}", flush=True)
    if "reuse" in suites:
        print(f"[{args.label}] reuso...", flush=True)
        out["reuse"] = eval_reuse(url, sampling, args.topk, args.workers)
        print(f"  {out['reuse']['metrics']}", flush=True)

    p = _HERE / "data" / f"eval_{args.label}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"salvo {p}", flush=True)


if __name__ == "__main__":
    main()
