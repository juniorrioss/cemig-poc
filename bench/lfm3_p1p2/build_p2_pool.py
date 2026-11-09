#!/usr/bin/env python3
"""
build_p2_pool.py — monta os pools de prompts do P2 a partir das bandas do P1
(`bands_<tag>.jsonl` + universo). Determinístico (seed fixa). Um pool por run:

  v1   (run lfm3-p2-rlvr)     : pool RLVR (formato APP) banda produtiva + difíceis;
                                recusa sobreamostrada 2x no treino (--refuse-oversample 2)
  bal  (run lfm3-p2-rlvr-bal) : v1 + TODAS as recusas fáceis do pool RLVR (âncoras) -> ~36%
  v2   (run lfm3-p2-rlvr-v2)  : v1 + âncoras de recusa até 30% (contrato v2)
  v3   (run lfm3-p2-rlvr-v3)  : v2 + itens do data2x no formato ESTÁTICO (system F0,
                                "Contexto normativo consultado", distratores DUROS da
                                mesma NR nos absent) das bandas produtiva/difícil,
                                expect answer (gold) e refuse (absent); recusa ~35%

Uso: classifier/.venv/bin/python bench/lfm3_p1p2/build_p2_pool.py --pool v3
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_pipeline as P  # noqa: E402

KEEP = ("productive", "hard_keep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["v1", "bal", "v2", "v3"], required=True)
    ap.add_argument("--work", default=str(P.DEFAULT_WORK))
    ap.add_argument("--tag", default="f4ep2")
    args = ap.parse_args()
    w = Path(args.work)
    items = {i["uid"]: i for i in P.load_universe(w, ["data2x", "f4", "rlvr"])}
    bands = P.read_jsonl(w / f"bands_{args.tag}.jsonl")
    rl = [b for b in bands if b["source"] == "rlvr"]
    keep = [items[b["uid"]] for b in rl if b["band"] in KEEP]
    easy_ref = [items[b["uid"]] for b in rl
                if b["band"] == "easy_drop" and items[b["uid"]]["expect"] == "refuse"]
    random.Random(11).shuffle(easy_ref)
    ans = [k for k in keep if k["expect"] == "answer"]
    ref = [k for k in keep if k["expect"] == "refuse"]
    if args.pool == "v1":
        rows = keep
    elif args.pool == "bal":
        rows = keep + easy_ref
    else:
        need = int(0.30 * len(ans) / 0.70) - len(ref)
        rows = ans + ref + easy_ref[:max(0, need)]
    if args.pool == "v3":
        d2 = [items[b["uid"]] for b in bands if b["source"] == "data2x" and b["band"] in KEEP
              and items[b["uid"]]["expect"] in ("answer", "refuse")]
        d2_easy_ref = [items[b["uid"]] for b in bands if b["source"] == "data2x"
                       and b["band"] == "easy_drop" and items[b["uid"]]["expect"] == "refuse"]
        random.Random(13).shuffle(d2_easy_ref)
        n_ans = sum(1 for r in rows + d2 if r["expect"] == "answer")
        n_ref = sum(1 for r in rows + d2 if r["expect"] == "refuse")
        need = int(0.35 * n_ans / 0.65) - n_ref
        rows = rows + d2 + d2_easy_ref[:max(0, need)]
    out = w / f"p2_prompts_{args.pool}.jsonl"
    P.write_jsonl(out, rows)
    print(out, len(rows), dict(collections.Counter((r["source"], r["expect"]) for r in rows)))


if __name__ == "__main__":
    main()
