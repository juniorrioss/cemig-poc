"""
gen_f0.py — geração do baseline LFM3-1.7B bf16 (roda na DGX Spark, GPU).

Modelo AS IS: LiquidAI/LFM3-1.7B, bf16, sem quantização, chat template oficial
(add_generation_prompt=True, thinking-ON default do template), sampling do model
card: temperature=0.1, top_k=50, repetition_penalty=1.1.

Saída: data/gen_{cell}.jsonl com uma linha por query
(id, base_qa_id, vocabulary_level, cell, shuffle_order, raw completo,
thinking, answer, n_gen_tokens, stopped_eos). Checkpoint incremental por id.

Uso (na Spark):
    python gen_f0.py --cell closed_book [--limit N] [--vocabs clean,lay] [--batch 24]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common_f0 import CELLS, VOCABS, build_messages, load_rows, read_jsonl, split_thinking

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "LiquidAI/LFM3-1.7B"
EOS_ID = 124900  # <|im_end|>


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--parquet", default=str(HERE / "normas_train.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vocabs", default="all")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    out_path = DATA / f"gen_{args.cell}.jsonl"
    done_ids = {r["id"] for r in read_jsonl(out_path)}

    rows = load_rows(args.parquet)
    vocabs = VOCABS if args.vocabs == "all" else tuple(args.vocabs.split(","))
    todo = [r for r in rows if r["vocabulary_level"] in vocabs and r["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gen] cell={args.cell} vocabs={vocabs} pendentes={len(todo)} "
          f"(já feitos {len(done_ids)})")
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    t0 = time.time()
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        prompts, metas = [], []
        for row in batch:
            built = build_messages(row, args.cell)
            text = tok.apply_chat_template(
                built["messages"], add_generation_prompt=True, tokenize=False)
            prompts.append(text)
            metas.append(built)
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True,
                temperature=0.1, top_k=50, repetition_penalty=1.1,
                pad_token_id=tok.pad_token_id, eos_token_id=EOS_ID)
        for row, meta, in_ids, out_ids in zip(batch, metas, enc["input_ids"], out):
            gen = out_ids[in_ids.shape[0]:]
            # remove padding à direita pós-EOS
            gen_list = gen.tolist()
            stopped = EOS_ID in gen_list
            if stopped:
                gen_list = gen_list[: gen_list.index(EOS_ID)]
            raw = tok.decode(gen_list, skip_special_tokens=False)
            parts = split_thinking(raw)
            rec = {
                "id": row["id"], "base_qa_id": row["base_qa_id"],
                "vocabulary_level": row["vocabulary_level"],
                "question_type": row["question_type"], "cell": args.cell,
                "shuffle_order": meta["shuffle_order"],
                "passage_ids": meta["passage_ids"],
                "correct_position": meta["correct_position"],
                "raw": raw, "thinking": parts["thinking"],
                "answer": parts["answer"],
                "n_gen_tokens": len(gen_list), "stopped_eos": stopped,
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = start + len(batch)
        rate = done / (time.time() - t0)
        print(f"[gen] {done}/{len(todo)} ({rate:.2f} q/s, "
              f"ETA {(len(todo)-done)/max(rate,1e-6)/60:.1f} min)", flush=True)

    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
