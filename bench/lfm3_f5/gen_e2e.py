#!/usr/bin/env python3
"""
gen_e2e.py — geração fim-a-fim da Frente 3 (roda na Spark/H100; batched).

Lê data/e2e/prompts.jsonl (e2e_build.py), aplica o chat template do LFM3 com
enable_thinking=False (igual ao harness do F4 / app), gera a síntese final com o
f4-r32-ep2 (merged data2x-ep4 + LoRA checkpoint-284) e o sampling de TODA a série LFM3
(temp 0.1, top_k 50, rep 1.1). Ordena por comprimento p/ batches homogêneos.
Checkpoint incremental por `key` (reentrante).

Uso (Spark, venv ~/cemig-poc/lfm3_f2/.venv):
  python gen_e2e.py --model ~/cemig-poc/lfm3_f4/lfm3-1.7b-data2x-ep4-merged \
      --adapter ~/cemig-poc/lfm3_f5/checkpoint-284 --tag f4-r32-ep2 --batch 32
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def split_thinking(text: str):
    m = re.match(r"\s*<think>(.*?)</think>", text or "", re.DOTALL)
    if m:
        return m.group(1), text[m.end():].strip()
    return "", (text or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default=str(HERE / "data" / "e2e" / "prompts.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--cells", default="")
    args = ap.parse_args()

    out_path = HERE / "data" / "e2e" / f"gen_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["key"] for l in out_path.read_text().splitlines() if l.strip()}
    items = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if args.cells:
        keep = set(args.cells.split(","))
        items = [it for it in items if it["cell"] in keep]
    todo = [it for it in items if it["key"] not in done]
    print(f"[gen] tag={args.tag} pendentes={len(todo)} (feitos {len(done)})", flush=True)
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[gen] adapter: {args.adapter}", flush=True)
    model.eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    for it in todo:
        it["_text"] = tok.apply_chat_template(it["messages"], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        it["_n"] = len(tok(it["_text"], add_special_tokens=False)["input_ids"])
    todo.sort(key=lambda x: x["_n"])
    t0 = time.time()
    for s in range(0, len(todo), args.batch):
        batch = todo[s:s + args.batch]
        enc = tok([b["_text"] for b in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=0.1, top_k=50, repetition_penalty=1.1,
                                 pad_token_id=pad, eos_token_id=eos)
        with out_path.open("a", encoding="utf-8") as fh:
            for b, o in zip(batch, out):
                gen = o[enc["input_ids"].shape[1]:].tolist()
                stopped = eos in gen
                if stopped:
                    gen = gen[: gen.index(eos)]
                raw = tok.decode(gen, skip_special_tokens=False)
                _, answer = split_thinking(raw)
                answer = answer.replace("<|im_end|>", "").strip()
                rec = {k: v for k, v in b.items() if k not in ("messages", "_text")}
                rec.update({"tag": args.tag, "n_prompt_tokens": rec.pop("_n"),
                            "raw": raw, "answer": answer, "n_gen_tokens": len(gen),
                            "stopped_eos": stopped})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dn = s + len(batch)
        rate = dn / (time.time() - t0)
        print(f"[gen] {dn}/{len(todo)} ({rate:.2f} it/s, ETA {(len(todo)-dn)/rate/60:.1f} min)",
              flush=True)
    print(f"[ok] {out_path}", flush=True)


if __name__ == "__main__":
    main()
