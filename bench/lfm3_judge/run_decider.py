#!/usr/bin/env python3
"""
run_decider.py — roda a rubrica de decisão (rubric_j) num decision model servido por
serve/decider_server.py, em LOTE entre itens, com checkpoint por linha.

Cada linha = (item, eixo, paráfrase) ou (item, fato). O servidor ordena por comprimento
e roda lotes fixos → determinístico. `--rep 2` grava numa saída separada para medir
bit-identidade entre execuções.

Uso:
  python run_decider.py --model pplx --sets V R B --rep 1
  python run_decider.py --model pplx --sets V --rep 2      # determinismo
Saída: data/raw/<model>_rep<k>_<set>.jsonl  ({key, answer})  + timing em data/raw/timing.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_j as C  # noqa: E402
from rubric_j import ANSWER_AXES, CONTEXT_AXES, axis_rows, fact_rows, span_rows  # noqa: E402


LEAN_AXES = ["verdict", "hallucination", "refusal"]  # só o que entra no gate


def rows_for(item: dict, with_facts: bool, with_span: bool = False, lean: bool = False) -> list[dict]:
    rows = []
    if lean:  # perfil de produção: 5 decisões/item, sem fatos/span/contexto
        return [r for ax in LEAN_AXES for r in axis_rows(ax, item)] if item.get("answer") else []
    if with_span and item["set"] in ("V", "R"):
        rows += span_rows(item)
    if item.get("answer"):
        for ax in ANSWER_AXES:
            rows += axis_rows(ax, item)
        if with_facts:
            rows += fact_rows(item)
    if item["set"] == "B":  # avaliação do retriever (contexto), inclusive o 5º (ours)
        for ax in CONTEXT_AXES:
            rows += axis_rows(ax, item)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pplx")
    ap.add_argument("--url", default=C.DECIDER_URL)
    ap.add_argument("--sets", nargs="+", default=["V", "R", "B"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256, help="linhas por requisição")
    ap.add_argument("--batch", type=int, default=8, help="lote do forward no servidor")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-facts", action="store_true")
    ap.add_argument("--lean", action="store_true", help="só os eixos do gate (velocidade de produção)")
    ap.add_argument("--backend", choices=["pplx", "vela"], default="pplx",
                    help="vela: agrupa por state e usa /system_one (inclui o eixo extra span)")
    args = ap.parse_args()
    raw_dir = C.DATA / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        items = C.read_jsonl(C.DATA / "sets" / f"set_{s}.jsonl")
        if args.limit:
            items = items[: args.limit]
        rows = [r for it in items for r in rows_for(it, not args.no_facts, args.backend == "vela", args.lean)]
        out = raw_dir / f"{args.model}_rep{args.rep}_{s}.jsonl"
        done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
        todo = [r for r in rows if r["key"] not in done]
        print(f"[{args.model} rep{args.rep} {s}] itens={len(items)} linhas={len(rows)} faltam={len(todo)}", flush=True)
        t_all, n_all = 0.0, 0
        with out.open("a", encoding="utf-8") as fh:
            for k in range(0, len(todo), args.chunk):
                part = todo[k:k + args.chunk]
                if args.backend == "pplx":
                    path = "/decide"
                    body = {"rows": [{"state": r["state"], "question": r["question"]} for r in part],
                            "batch_size": args.batch}
                else:  # vela: 1 system_one por state distinto (perguntas do mesmo state juntas)
                    path = "/system_one"
                    groups: dict[str, list[dict]] = {}
                    for r in part:
                        groups.setdefault(json.dumps(r["state"], ensure_ascii=False, sort_keys=True), []).append(r)
                    glist = list(groups.values())
                    body = {"items": [{"state": g[0]["state"],
                                       "questions": {r["key"]: r["question"] for r in g}} for g in glist]}
                for attempt in range(5):
                    try:
                        res = requests.post(f"{args.url}{path}", json=body, timeout=3600)
                        res.raise_for_status()
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"  retry {attempt}: {e}", flush=True)
                        time.sleep(10 * (attempt + 1))
                else:
                    raise SystemExit("decider indisponível")
                js = res.json()
                if args.backend == "vela":
                    amap = {}
                    for g, rr in zip(glist, js["results"]):
                        for r in g:
                            amap[r["key"]] = (rr.get("answers") or {}).get(r["key"], {"error": rr.get("error", "missing")})
                    js["answers"] = [amap[r["key"]] for r in part]
                for r, a in zip(part, js["answers"]):
                    fh.write(json.dumps({"key": r["key"], "answer": a}, ensure_ascii=False) + "\n")
                fh.flush()
                t_all += js["seconds"]
                n_all += len(part)
                print(f"  {k + len(part)}/{len(todo)}  {len(part) / js['seconds']:.2f} linhas/s", flush=True)
        if n_all:
            with (raw_dir / "timing.jsonl").open("a") as fh:
                fh.write(json.dumps({"model": args.model, "rep": args.rep, "set": s, "rows": n_all,
                                     "items": len(items), "gpu_seconds": round(t_all, 2),
                                     "batch": args.batch, "ts": time.time()}) + "\n")


if __name__ == "__main__":
    main()
