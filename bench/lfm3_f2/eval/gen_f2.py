"""
gen_f2.py — geração de avaliação F2 (roda na Spark). Igual ao gen_f0.py do
baseline, mas: (a) aceita --model (checkpoint HF local, modelo com LoRA
mergeado, ou task-vector dir); (b) aceita --adapter (LoRA aplicado on-the-fly,
evita merges intermediários em disco); (c) --norms filtra as queries às normas
tratadas+controle (recorte do F2; a bateria completa é opcional); (d) saída
nomeada por --tag.

Saída: data/gen_{tag}_{cell}.jsonl (mesmo schema do F0 -> score_f0 reusável).

Uso (na Spark):
    python gen_f2.py --cell closed_book --model LiquidAI/LFM3-1.7B \
        --adapter runs/lfm3-f2-A-r8/checkpoint-842 --tag f2-A-r8-ep1 \
        --norms 10,18,34,13,31,33
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_f0 import CELLS, VOCABS, build_messages, load_rows, read_jsonl, split_thinking  # noqa: E402

DATA = HERE / "data"
EOS_ID = 124900  # <|im_end|>


def nr_of(source_file: str) -> int | None:
    m = re.search(r"nr[-_ ]?(\d+)", source_file.lower())
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parquet", default=str(HERE / "normas_train.parquet"))
    ap.add_argument("--norms", default="", help="ex.: 10,18,34,13,31,33 (vazio = todas)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vocabs", default="all")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=3072)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    out_path = DATA / f"gen_{args.tag}_{args.cell}.jsonl"
    done_ids = {r["id"] for r in read_jsonl(out_path)}

    rows = load_rows(args.parquet)
    vocabs = VOCABS if args.vocabs == "all" else tuple(args.vocabs.split(","))
    norms = {int(n) for n in args.norms.split(",") if n} if args.norms else None
    todo = [r for r in rows
            if r["vocabulary_level"] in vocabs and r["id"] not in done_ids
            and (norms is None or nr_of(r["source_file"]) in norms)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gen] tag={args.tag} cell={args.cell} pendentes={len(todo)} "
          f"(feitos {len(done_ids)})")
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[gen] adapter aplicado: {args.adapter}")
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
                "tag": args.tag,
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
