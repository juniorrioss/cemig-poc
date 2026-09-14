#!/usr/bin/env python3
"""
measure_truncation_2_6b.py — mede o truncamento do dataset com o tokenizer do 2.6B.

O 2.6B NÃO compartilha o tokenizer do 1.2B (vocab 124.893 vs 64.400; tokeniza mais
eficientemente ~18% menos tokens). Logo o truncamento do brief tem de ser RE-MEDIDO com o
tokenizer do 2.6B — o número do tools_v1 (8.71% a 2048 / 0% a 3072, medido no 1.2B) não vale.

Renderiza cada exemplo de treino com o TEMPLATE + tokenizer OFICIAIS do 2.6B (tools já baked
no system) e conta quantos passam de max_length, por família. Roda na Spark (base_hf mora lá).

Uso (Spark): ~/jupyterlab/.venv/bin/python ~/cemig-poc/tools_2_6b/measure_truncation_2_6b.py \
    --data ~/cemig-poc/tools_v1/data/train_full.jsonl --max-length 2048 3072 \
    --base ~/cemig-poc/base_hf --out ~/cemig-poc/tools_2_6b/data/truncation.json
Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default=str(Path.home() / "cemig-poc" / "base_hf"))
    ap.add_argument("--max-length", type=int, nargs="+", default=[2048, 3072])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

    lengths = []
    fam_of = []
    for r in rows:
        # tools já baked no system -> render com tools=None (apply_chat_template oficial).
        text = tok.apply_chat_template(r["messages"], tools=None, tokenize=False,
                                       add_generation_prompt=False)
        n = len(tok.encode(text))
        lengths.append(n)
        fam_of.append(r["meta"].get("family", "?"))

    lengths_sorted = sorted(lengths)
    n = len(lengths)
    p50 = lengths_sorted[int(0.50 * n)]
    p95 = lengths_sorted[int(0.95 * n)]
    p99 = lengths_sorted[min(int(0.99 * n), n - 1)]
    result = {"data": args.data, "base": args.base, "n": n,
              "max_token_len": max(lengths), "p50": p50, "p95": p95, "p99": p99,
              "per_max_length": {}}
    for ml in args.max_length:
        over = sum(1 for x in lengths if x > ml)
        over_by_fam = Counter()
        fam_total = Counter()
        for x, f in zip(lengths, fam_of):
            fam_total[f] += 1
            if x > ml:
                over_by_fam[f] += 1
        result["per_max_length"][str(ml)] = {
            "over": over, "over_pct": round(100 * over / n, 2),
            "over_by_family": {f: {"over": over_by_fam[f], "total": fam_total[f],
                                   "pct": round(100 * over_by_fam[f] / fam_total[f], 1)}
                               for f in sorted(fam_total)},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "max_token_len", "p50", "p95", "p99",
                                             "per_max_length")}, ensure_ascii=False, indent=1))
    print(f"[ok] -> {args.out}")


if __name__ == "__main__":
    main()
