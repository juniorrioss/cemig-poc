#!/usr/bin/env python3
"""
score_quant.py — régua honesta da F6 (MESMO worker de bench/lfm3_f6/score_f6.py, juiz 27B) sobre
as gerações por formato de quantização do f4-r32-ep2 no recorte 344, célula op_d50.

1) Normaliza data/gen/gen_<tag>.jsonl (saída crua do Spark: gen_gguf.py / gen_litert.py) para o
   formato de geração da F6 (metadados do prompt congelado + answer sem <think>/<|im_end|>) em
   data/e2e/gen_<tag>.jsonl;
2) Chama score_f6.main com F6.DATA/F6.CACHE apontados para este diretório (zero cópia de régua).

A referência bf16 transformers NÃO é regerada: são as rodadas op_d50 da F6 (f6-recorte, -rep2,
-rep3), já pontuadas pelo mesmo score_f6 — ver consolidate_quant.py.

Uso: classifier/.venv/bin/python bench/lfm3_quant/score_quant.py --tag gguf_q4_0-r1 [...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f6"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f5"))
import common_f6 as F6  # noqa: E402
import score_f6  # noqa: E402

DATA = HERE / "data"
META_KEYS = ("key", "id", "cell", "variant", "suite", "base_qa_id", "vocabulary_level",
             "question_type", "gold_kind", "source_file", "ctx_ntok", "ctx_ids", "n_manual_ctx",
             "context_type", "gold_items", "ctx_has_gold_span")


def clean(raw: str) -> str:
    """Mesmo pós-processamento do gen_e2e.py da F5 (split_thinking + strip de <|im_end|>)."""
    m = re.match(r"\s*<think>(.*?)</think>", raw or "", re.DOTALL)
    t = raw[m.end():] if m else (raw or "")
    return t.replace("<|im_end|>", "").strip()


def normalize(tag: str) -> Path:
    prompts = {json.loads(l)["key"]: json.loads(l) for l in open(DATA / "prompts_op344.jsonl")}
    out = DATA / "e2e" / f"gen_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA / "gen" / f"gen_{tag}.jsonl") if l.strip()]
    seen = set()
    with out.open("w", encoding="utf-8") as fh:
        for g in rows:
            if g["key"] in seen:
                continue
            seen.add(g["key"])
            p = prompts[g["key"]]
            rec = {k: p[k] for k in META_KEYS}
            rec.update({"tag": tag, "engine_format": g["engine_format"], "seed": g["seed"],
                        "n_prompt_tokens": g["n_prompt_tokens"], "raw": g["raw"],
                        "answer": clean(g["raw"]), "n_gen_tokens": g["n_gen_tokens"],
                        "stopped_eos": bool(g["stopped_eos"]), "tok_parity": g["tok_parity"],
                        "decode_tps": g.get("decode_tps"), "prefill_tps": g.get("prefill_tps")})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    assert len(seen) == len(prompts), f"{tag}: {len(seen)}/{len(prompts)} gerações"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    F6.DATA = DATA
    F6.CACHE = HERE / "cache"
    for tag in a.tag:
        normalize(tag)
        sys.argv = ["score_f6.py", "--suite", "recorte", "--tag", tag, "--workers", str(a.workers)]
        score_f6.main()


if __name__ == "__main__":
    main()
