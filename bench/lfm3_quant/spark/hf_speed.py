#!/usr/bin/env python3
"""hf_speed.py — Spark: decode t/s single-stream (batch 1) do bf16 transformers (referência-teto)
nos primeiros N prompts congelados, mesmo sampling da série LFM3. Só VELOCIDADE: a qualidade do
bf16 transformers é a das 3 rodadas op_d50 da F6 (batch 8), não regerada aqui.

Uso (venv ~/cemig-poc/lfm3_f2/.venv): python hf_speed.py --n 20
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "cemig-poc"
Q = W / "lfm3_quant"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(str(W / "lfm3_f7/merged"), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(W / "lfm3_f7/merged"), dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True).eval()
eos = tok.convert_tokens_to_ids("<|im_end|>")
items = [json.loads(l) for l in open(Q / "prompts_op344_rendered.jsonl")][: a.n]
rows = []
for it in items:
    ids = torch.tensor([it["hf_ids"]], device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=eos)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=768, do_sample=True, temperature=0.1, top_k=50,
                             repetition_penalty=1.1, pad_token_id=eos, eos_token_id=eos)
    torch.cuda.synchronize(); t2 = time.time()
    n = out.shape[1] - ids.shape[1]
    dec = (n - 1) / max(1e-6, (t2 - t1) - (t1 - t0))
    rows.append({"key": it["key"], "n_gen": n, "prefill_s": round(t1 - t0, 3),
                 "decode_tps": round(dec, 2)})
    print(rows[-1], flush=True)
json.dump({"engine_format": "hf_bf16", "device": "Spark GB10 CUDA, batch 1",
           "decode_tps_mean": round(sum(r["decode_tps"] for r in rows) / len(rows), 2), "rows": rows},
          open(Q / "speed_hf_bf16.json", "w"), indent=1)
print("[ok]")
