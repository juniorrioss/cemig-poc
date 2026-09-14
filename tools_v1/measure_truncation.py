#!/usr/bin/env python3
"""
measure_truncation.py — mede o truncamento do dataset a max_length (exigência do brief).

Renderiza cada exemplo de treino com o tokenizer OFICIAL (tools no system + multiturno) e
conta quantos passam de max_length. O brief manda MEDIR e REPORTAR o truncamento e aumentar
max_len se o multiturno truncar. Roda no .venv-train (tem transformers).

Uso: ../.venv-train/bin/python measure_truncation.py --data data/train_full.jsonl --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import get_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_full.jsonl")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--json-out", default="data/truncation.json")
    args = ap.parse_args()

    tok = get_tokenizer()
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

    lengths = []
    over = 0
    over_by_fam: Counter = Counter()
    fam_total: Counter = Counter()
    for r in rows:
        # tools já estão no system (baked); renderiza com tools=None
        enc = tok.apply_chat_template(r["messages"], tools=None, tokenize=True,
                                      return_dict=True)
        n = len(enc["input_ids"])
        lengths.append(n)
        fam = r["meta"]["family"]
        fam_total[fam] += 1
        if n > args.max_length:
            over += 1
            over_by_fam[fam] += 1

    lengths.sort()
    n = len(lengths)
    def pct(p):
        return lengths[min(n - 1, int(p * n))]
    stats = {
        "n": n, "max_length": args.max_length,
        "over_count": over, "over_pct": round(100 * over / max(1, n), 2),
        "over_by_family": {k: f"{over_by_fam[k]}/{fam_total[k]}" for k in fam_total},
        "tokens_p50": pct(0.50), "tokens_p90": pct(0.90), "tokens_p95": pct(0.95),
        "tokens_p99": pct(0.99), "tokens_max": lengths[-1], "tokens_min": lengths[0],
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    Path(args.json_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
