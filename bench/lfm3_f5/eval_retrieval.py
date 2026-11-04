#!/usr/bin/env python3
"""
eval_retrieval.py — avalia variantes de retrieval nas DUAS réguas (POC 151 + bench 1536).

Variantes (registro VARIANTS; Frente 1 = gate, Frente 2 = vizinhos, Frente 4 = manuais):
  v4_anchor      espelho antigo (âncora F3/F4: gate mirror, sem override)     -> paridade
  v4app_smoke    override + gate mirror (= 'v4app' do smoke)
  v4app          override + gate FIEL ao app (top1='nenhuma' -> sem boost)  [CONTROLE (d)]
  thr{0.6..0.9}  (a) limiar do filtro duro maior
  f{2,1.5,1}     (b) boost suave reduzido
  noclf_ovr      (c) só override de NR explícita, sem classificador
  noclf          sem classificador e sem override (referência do smoke)
  *_nb1          (Frente 2) + vizinhos ±1 sobre top-2  (métricas de CONJUNTO: hit/evrec)
  *_man{w}       (Frente 4) + 2 sinais de manual no RRF com peso w (+ _intent: gate por intenção)

Saída: data/retrieval/<variante>.json (agregados por régua/subconjunto/vocab) +
cache/rank_<variante>.jsonl (rankings por query, gitignored).

Uso: classifier/.venv/bin/python bench/lfm3_f5/eval_retrieval.py --variants v4_anchor,v4app
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common_f5 as C  # noqa: E402
from metrics import BenchScorer, SpanStore, content_hit, mean_dicts, score_151  # noqa: E402
from retriever_f5 import Cfg, F5Retriever  # noqa: E402

BASE_APP = Cfg(gate="app", explicit=True)
VARIANTS: Dict[str, Cfg] = {
    "v4_anchor": Cfg(gate="mirror", explicit=False),
    "v4app_smoke": Cfg(gate="mirror", explicit=True),
    "v4app": BASE_APP,
    "noclf": Cfg(gate="noclf", explicit=False),
    "noclf_ovr": Cfg(gate="noclf", explicit=True),
}
for t in (0.6, 0.7, 0.8, 0.9):
    VARIANTS[f"thr{t}"] = BASE_APP.with_(conf_thr=t)
for f in (2.0, 1.5, 1.0):
    VARIANTS[f"f{f:g}"] = BASE_APP.with_(factor=f)
VARIANTS["thr0.7_f2"] = BASE_APP.with_(conf_thr=0.7, factor=2.0)
VARIANTS["thr0.8_f1.5"] = BASE_APP.with_(conf_thr=0.8, factor=1.5)
VARIANTS["hardoff_f2"] = BASE_APP.with_(conf_thr=1.01, factor=2.0)   # nunca filtro duro


def variant_cfg(name: str) -> Cfg:
    """Resolve nome com sufixos _nb1 / _man{w} / _intent."""
    parts = name.split("+")
    cfg = VARIANTS[parts[0]]
    for p in parts[1:]:
        if p == "nb1":
            cfg = cfg.with_(neighbors=1)
        elif p.startswith("man"):
            cfg = cfg.with_(manual_w=float(p[3:]))
        elif p == "intent":
            cfg = cfg.with_(manual_gate="intent")
        elif p.startswith("margin"):
            cfg = cfg.with_(manual_gate="margin", manual_margin=float(p[6:] or 0))
        else:
            raise ValueError(p)
    return cfg


def run_variant(rtv: F5Retriever, name: str, store: SpanStore) -> Dict[str, Any]:
    cfg = variant_cfg(name)
    bs = BenchScorer(store)
    t0 = time.time()
    # ---------------- POC 151
    r151 = []
    for it in C.load_151():
        res = rtv.search(it["question"], cfg, top_k=5)
        ranked = [rtv.chunk(c) | {"id": c} for c in res["ids"]]
        extra = {}
        if cfg.neighbors:
            extra["nb"] = [rtv.chunk(c) | {"id": c} for c in res["expanded"]]
        r151.append({**score_151(ranked, it, extra), "mode": res["mode"]})
    # ---------------- bench 1536
    rb: List[Dict[str, Any]] = []
    for row in C.load_bench():
        res = rtv.search(row["question"], cfg, top_k=5)
        extra = {"nb": res["expanded"]} if cfg.neighbors else {}
        sc = bs.score(res["ids"], row, extra)
        texts2 = [rtv.chunk(c)["text"] for c in res["ids"][:2]]
        texts5 = [rtv.chunk(c)["text"] for c in res["ids"][:5]]
        sc["chit2"] = content_hit(texts2, row)
        sc["chit5"] = content_hit(texts5, row)
        if cfg.neighbors:
            sc["chit_nb"] = content_hit([rtv.chunk(c)["text"] for c in res["expanded"]], row)
        # NR certa no top-5 (arquivo-ouro de norma OU manual da mesma NR)
        gold_nr = C.nr_num(row["source_file"]) if row["gold_kind"] == "norma" else None
        sc["n_ctx_nb"] = float(len(res.get("expanded", res["top"][:2])))
        rb.append({**sc, "ids": res["ids"], "expanded": res.get("expanded"), "id": row["id"], "vocab": row["vocabulary_level"],
                   "kind": row["gold_kind"], "mode": res["mode"],
                   "n_manual_top5": sum(1 for c in res["ids"][:5] if c >= 1_000_000)})
    agg: Dict[str, Any] = {"variant": name, "cfg": cfg.__dict__ | {"rrf_w": list(cfg.rrf_w)},
                           "secs": round(time.time() - t0, 1)}
    agg["poc151"] = mean_dicts([{k: v for k, v in r.items() if k != "mode"} for r in r151])
    agg["poc151"]["modes"] = _count([r["mode"] for r in r151])
    num = lambda rs: [{k: v for k, v in r.items() if isinstance(v, float)} for r in rs]  # noqa: E731
    for sub, sel in (("bench_norma", lambda r: r["kind"] == "norma"),
                     ("bench_all", lambda r: True),
                     ("bench_manual", lambda r: r["kind"] == "manual")):
        rs = [r for r in rb if sel(r)]
        agg[sub] = mean_dicts(num(rs))
        agg[sub]["n"] = len(rs)
        agg[sub]["by_vocab"] = {v: mean_dicts(num([r for r in rs if r["vocab"] == v]))
                                for v in C.VOCABS} if sub == "bench_norma" else {}
        agg[sub]["modes"] = _count([r["mode"] for r in rs])
    C.write_json(C.DATA / "retrieval" / f"{name}.json", agg)
    with (C.CACHE / f"rank_{name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rb:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return agg


def _count(xs):
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def brief(agg: Dict[str, Any]) -> str:
    p, n, a, m = agg["poc151"], agg["bench_norma"], agg["bench_all"], agg["bench_manual"]
    s = (f"{agg['variant']:<26} 151 R@2 {p['r2']*100:5.1f} R@5 {p['r5']*100:5.1f} | "
         f"norma hit@2 {n['hit2']:.3f} hit@5 {n['hit5']:.3f} evrec@2 {n['evrec2']:.3f} "
         f"nDCG {n['ndcg5']:.3f} | all hit@5 {a['hit5']:.3f} | man hit@5 {m['hit5']:.3f} "
         f"chit5 {m['chit5']:.3f}")
    if "hit_nb" in n:
        s += f" | NB 151 {p['hit_nb']*100:.1f} norma hit {n['hit_nb']:.3f} evrec {n['evrec_nb']:.3f}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    args = ap.parse_args()
    names = args.variants.split(",")
    with_man = any("man" in n for n in names)
    rtv = F5Retriever(with_manuals=with_man)
    extra = rtv.man.span_store() if with_man else None
    store = SpanStore(extra)
    # pré-encoda todas as queries (cache em disco)
    rtv.qc.ensure([it["question"] for it in C.load_151()] + [r["question"] for r in C.load_bench()])
    for n in names:
        agg = run_variant(rtv, n, store)
        print(brief(agg), flush=True)


if __name__ == "__main__":
    main()
