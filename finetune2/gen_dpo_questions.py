#!/usr/bin/env python3
"""
gen_dpo_questions.py — pool CLEAN de perguntas coloquiais p/ a FASE B (fonte das saídas do 2.6B).

O dev_set (200) é pequeno demais para render 300-500 pares DPO. Este script gera um pool maior
(~800) de perguntas de campo a partir de chunks NÃO-holdout e NÃO-reservados (mesma higiene do
retrieval3/gen_devset.py: fala coloquial sem termos técnicos, sem citar item/norma). Cada pergunta
passa pelo filtro lexical vs holdout (Jaccard<0.4) — descartes registrados.

Saída: data/dpo_questions.jsonl (id, question, doc, section, chunk_id, relevant_chunk_ids).
NÃO é holdout; usada só como origem das respostas do 2.6B a serem editadas (DPO).

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (DATA_DIR, DEFAULT_VLLM_MODEL, DEFAULT_VLLM_URL, RESERVED_NRS,
                    LexicalFilter, call_vllm, holdout_gold_chunk_ids, load_chunks)

SYSTEM_PROMPT = (
    "Você é um eletricista/operário brasileiro de campo, leigo em termos jurídicos. Dado um TRECHO "
    "de uma Norma Regulamentadora, formule UMA pergunta curta, falada, do jeito que você "
    "perguntaria em voz alta no serviço — com gíria e linguagem do dia a dia.\n"
    "Regras rígidas:\n"
    "1. NÃO use os termos técnicos do trecho (ex.: 'talabarte' -> 'cinto/corda'; 'desenergização' "
    "-> 'desligar a energia').\n"
    "2. NÃO cite números de item nem o nome/número da norma.\n"
    "3. Uma única pergunta, curta e natural. Responda SOMENTE a pergunta, sem aspas nem preâmbulo."
)

USER_TEMPLATE = ("TRECHO (seção {section} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\nSua pergunta de campo:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--min-len", type=int, default=250)
    ap.add_argument("--url", default=DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    ap.add_argument("--out", default=str(DATA_DIR / "dpo_questions.jsonl"))
    ap.add_argument("--discards", default=str(DATA_DIR / "dpo_questions_lexical_discards.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks()
    gold = holdout_gold_chunk_ids()
    pool = [c for c in chunks if c["id"] not in gold and c["doc"] not in RESERVED_NRS
            and len(c["text"]) >= args.min_len]
    rng.shuffle(pool)
    sample = pool[:args.n]
    print(f"chunks elegíveis: {len(pool)} | amostrados: {len(sample)}", flush=True)

    lf = LexicalFilter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                try:
                    done.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
        print(f"checkpoint: {len(done)} perguntas já existentes", flush=True)
    else:
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# " + json.dumps({
                "artifact": "dpo_questions", "task": "poc-dpo-sintese",
                "purpose": "dpo_source_NOT_holdout", "model": args.model, "seed": args.seed,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    ok = len(done)
    for i, c in enumerate(sample, 1):
        if c["id"] in done:
            continue
        try:
            raw = call_vllm([{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": USER_TEMPLATE.format(
                                 section=c["section"], title=c["title"], text=c["text"][:1800])}],
                            url=args.url, model=args.model, temperature=0.7, max_tokens=96,
                            thinking=False)
        except Exception as e:  # noqa: BLE001
            print(f"  falha chunk {c['id']}: {e}", flush=True)
            continue
        q = raw.strip().strip('"').split("\n")[0].strip()
        if len(q) < 8:
            continue
        if not lf.accept(q, meta={"chunk_id": c["id"], "doc": c["doc"]}):
            continue
        rec = {"id": f"dpoq-{c['id']}", "question": q, "doc": c["doc"], "section": c["section"],
               "chunk_id": c["id"], "relevant_chunk_ids": [c["id"]]}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        ok += 1
        if ok % 50 == 0:
            print(f"  ok={ok}/{args.n} lex_disc={len(lf.discards)} "
                  f"{ok/max(1e-9,time.time()-t0):.2f}/s", flush=True)
    fh.close()
    lf.dump_discards(Path(args.discards))
    print(f"pool DPO pronto: {ok} perguntas | descartes lexicais: {len(lf.discards)} "
          f"| {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
