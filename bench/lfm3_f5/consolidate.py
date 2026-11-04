#!/usr/bin/env python3
"""
consolidate.py — tabela-mestra da F5 (4 frentes) a partir de data/retrieval/*.json,
data/compress/proxy_*.json e data/e2e/summary_*.json -> data/consolidation.json + stdout.

Trava anti-mascaramento (retrieval4): uma variante só "conta" se sobe numa régua sem
derrubar a outra. Δ são vs o CONTROLE v4app (= app). Tolerância de queda: 0,5 p.p.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_f5 as C  # noqa: E402

TOL = 0.005


def load(name):
    p = C.DATA / "retrieval" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def row(name, ctrl):
    a = load(name)
    if a is None:
        return None
    p, n, al, m = a["poc151"], a["bench_norma"], a["bench_all"], a["bench_manual"]
    cp, cn = ctrl["poc151"], ctrl["bench_norma"]
    d151 = max(p["r2"] - cp["r2"], p["r5"] - cp["r5"])
    w151 = min(p["r2"] - cp["r2"], p["r5"] - cp["r5"])
    dbn = max(n["hit2"] - cn["hit2"], n["hit5"] - cn["hit5"])
    wbn = min(n["hit2"] - cn["hit2"], n["hit5"] - cn["hit5"])
    up = (d151 > TOL or dbn > TOL)
    passes = up and w151 >= -TOL and wbn >= -TOL
    r = {"variant": name,
         "151_r1": p["r1"], "151_r2": p["r2"], "151_r5": p["r5"], "151_mrr": p["mrr"],
         "norma_hit1": n["hit1"], "norma_hit2": n["hit2"], "norma_hit5": n["hit5"],
         "norma_evrec2": n["evrec2"], "norma_ndcg5": n["ndcg5"], "norma_mrr5": n["mrr5"],
         "all_hit5": al["hit5"], "manual_hit2": m["hit2"], "manual_hit5": m["hit5"],
         "manual_chit5": m["chit5"], "lay_hit5": n["by_vocab"]["lay"]["hit5"],
         "trava_ok": passes}
    if "hit_nb" in n:
        r.update({"151_hit_nb": p["hit_nb"], "norma_hit_nb": n["hit_nb"],
                  "norma_evrec_nb": n["evrec_nb"]})
    return r


FRENTE1 = ["v4_anchor", "v4app_smoke", "v4app", "thr0.6", "thr0.7", "thr0.8", "thr0.9",
           "f2", "f1.5", "f1", "thr0.7_f2", "thr0.8_f1.5", "hardoff_f2", "noclf_ovr", "noclf"]
FRENTE2 = ["v4app+nb1", "thr0.7_f2+nb1"]
FRENTE4 = ["v4app+man0.5", "v4app+man1", "v4app+man1.25", "v4app+man1.5", "v4app+man1.75",
           "v4app+man2", "v4app+man3", "v4app+man1+intent", "v4app+man2+intent",
           "v4app+man1.5+margin", "v4app+man2+margin", "v4app+man2+margin0.02",
           "v4app+man3+margin0.02", "thr0.7_f2+man0.5", "thr0.7_f2+man1"]


def main() -> None:
    ctrl = load("v4app")
    out = {"control": "v4app", "tolerance": TOL}
    for key, names in (("frente1_gate", FRENTE1), ("frente2_vizinhos", FRENTE2),
                       ("frente4_manuais", FRENTE4)):
        out[key] = [r for r in (row(n, ctrl) for n in names) if r]
    for f in sorted((C.DATA / "compress").glob("proxy_*.json")):
        out.setdefault("frente3_proxy", {})[f.stem] = json.loads(f.read_text())
    for f in sorted((C.DATA / "e2e").glob("summary_*.json")):
        out.setdefault("frente3_e2e", {})[f.stem] = json.loads(f.read_text())
    C.write_json(C.DATA / "consolidation.json", out)
    for key in ("frente1_gate", "frente2_vizinhos", "frente4_manuais"):
        print(f"== {key}")
        for r in out[key]:
            s = (f"  {r['variant']:<24} 151 R@2 {r['151_r2']*100:5.1f} R@5 {r['151_r5']*100:5.1f} "
                 f"MRR {r['151_mrr']:.3f} | norma h@1 {r['norma_hit1']:.3f} h@2 {r['norma_hit2']:.3f} "
                 f"h@5 {r['norma_hit5']:.3f} ev@2 {r['norma_evrec2']:.3f} nDCG {r['norma_ndcg5']:.3f} "
                 f"lay {r['lay_hit5']:.3f} | all h@5 {r['all_hit5']:.3f} | man h@5 "
                 f"{r['manual_hit5']:.3f} | trava {'OK' if r['trava_ok'] else '--'}")
            if "norma_hit_nb" in r:
                s += (f" | NB 151 {r['151_hit_nb']*100:.1f} norma {r['norma_hit_nb']:.3f} "
                      f"ev {r['norma_evrec_nb']:.3f}")
            print(s)
    print(f"[ok] {C.DATA / 'consolidation.json'}")


if __name__ == "__main__":
    main()
