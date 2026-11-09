#!/usr/bin/env python3
"""
consolidate_full.py — tabela da BATERIA COMPLETA (trava anti-overfit): distrator ×1536,
e2e op_d50 ×1536 (formato do app) e e2e ×151 coloquiais, por tag, com pareado vs base,
recorte vs FORA do recorte (as 1192 queries de normas que NÃO estão no recorte 344).

Uso: classifier/.venv/bin/python bench/lfm3_p1p2/eval/consolidate_full.py --tags v3s400 v4s400
"""
from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORTE = {10, 18, 34, 13, 31, 33}


def rj(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def sign_p(w, l):
    n = w + l
    if not n:
        return 1.0
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4)


def nr_of(src: str):
    m = re.search(r"nr[-_ ]?(\d+)", (src or "").lower())
    return int(m.group(1)) if m else None


def pct(rs, k):
    return round(100 * sum(1 for r in rs if r.get(k)) / max(1, len(rs)), 1) if rs else None


def paired(a, b, key="approved"):
    bb = {r["id"]: r[key] for r in b}
    w = sum(1 for r in a if r[key] and not bb.get(r["id"], False))
    l = sum(1 for r in a if not r[key] and bb.get(r["id"], False))
    return {"win": w, "loss": l, "sign_p": sign_p(w, l)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/d/cemig-scratch/p1p2/eval")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=str(HERE.parent / "data" / "consolidation_full.json"))
    args = ap.parse_args()
    w = Path(args.work)
    import pyarrow.parquet as pq
    import glob
    par = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/"
                        "snapshots/*/normas_regulamentadoras/train-*.parquet"))[0]
    src = {r["id"]: r["source_file"] for r in pq.read_table(par).to_pylist()}

    def load(tag):
        d = rj(w / "f2root" / "data" / "scored" / f"scored_p2-{tag}1536_distrator.jsonl")
        e = rj(w / "f6root" / "data" / "e2e" / f"scored_p2-{tag}-1536.jsonl")
        c = rj(w / "f6root" / "data" / "e2e" / f"scored_p2-{tag}-151.jsonl")
        return d, e, c

    base = load("base")
    res = {}
    for tag in ["base"] + args.tags:
        d, e, c = load(tag)
        out = {}
        for name, sub in (("all", lambda r: True), ("recorte", lambda r: nr_of(src[r["id"]]) in RECORTE),
                          ("fora_recorte", lambda r: nr_of(src[r["id"]]) not in RECORTE)):
            ds = [r for r in d if sub(r)]
            es = [r for r in e if sub(r)]
            gi = [r for r in es if r["has_gold_items"]]
            blk = {"distrator_n": len(ds), "distrator_hallucination": pct(ds, "hallucination"),
                   "distrator_refusal_correct": pct(ds, "refusal_correct"),
                   "e2e_n": len(es), "e2e_approved": pct(es, "approved"),
                   "e2e_hallucination": pct(es, "hallucination"), "e2e_refused": pct(es, "refused"),
                   "e2e_cite_gold_item": round(100 * sum(1 for r in gi if r["cite_item"]) / max(1, len(gi)), 1) if gi else None,
                   "e2e_gen_tok": round(sum(r["n_gen_tokens"] for r in es) / max(1, len(es)), 1) if es else None}
            if tag != "base":
                bd = [r for r in base[0] if sub(r)]
                be = [r for r in base[1] if sub(r)]
                if ds and bd:
                    blk["distrator_halluc_paired_worse_better"] = paired(ds, bd, "hallucination")
                if es and be:
                    blk["e2e_paired_vs_base"] = paired(es, be)
            out[name] = blk
        out["poc151"] = {"n": len(c), "approved": pct(c, "approved"), "hallucination": pct(c, "hallucination"),
                         "refused": pct(c, "refused"),
                         "gen_tok": round(sum(r["n_gen_tokens"] for r in c) / max(1, len(c)), 1) if c else None}
        if tag != "base" and c and base[2]:
            out["poc151"]["paired_vs_base"] = paired(c, base[2])
        res[tag] = out
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"{'tag':<9}{'escopo':<14}{'dist-h':>7}{'dist-rc':>8}{'e2e':>6}{'cite':>6}{'e2eH':>6}{'tok':>6}  pareado(e2e)")
    for t, o in res.items():
        for k in ("all", "recorte", "fora_recorte"):
            b = o[k]
            f = lambda x: f"{x:6.1f}" if isinstance(x, (int, float)) else "     -"  # noqa: E731
            print(f"{t:<9}{k:<14}{f(b['distrator_hallucination'])} {f(b['distrator_refusal_correct'])}"
                  f"{f(b['e2e_approved'])}{f(b['e2e_cite_gold_item'])}{f(b['e2e_hallucination'])}{f(b['e2e_gen_tok'])}  "
                  f"{b.get('e2e_paired_vs_base', '')} {b.get('distrator_halluc_paired_worse_better', '')}")
        p = o["poc151"]
        print(f"{t:<9}{'poc151':<14} aprov {p['approved']} aluc {p['hallucination']} recusa {p['refused']} {p.get('paired_vs_base', '')}")


if __name__ == "__main__":
    main()
