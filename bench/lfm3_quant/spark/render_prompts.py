#!/usr/bin/env python3
"""render_prompts.py — Spark: renderiza UMA VEZ os 344 prompts op_d50 da F6 com o chat template
HF do merged f4-r32-ep2 (enable_thinking=False, == gen_e2e da F5/F6). O texto renderizado é
congelado e reusado por TODOS os motores (llama.cpp e LiteRT-LM); só o motor muda.

Uso (venv ~/cemig-poc/lfm3_f2/.venv): python render_prompts.py
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

W = Path.home() / "cemig-poc"
Q = W / "lfm3_quant"
tok = AutoTokenizer.from_pretrained(str(W / "lfm3_f7/merged"), trust_remote_code=True)
out = open(Q / "prompts_op344_rendered.jsonl", "w")
for line in open(Q / "prompts_op344.jsonl"):
    it = json.loads(line)
    text = tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out.write(json.dumps({"key": it["key"], "id": it["id"], "prompt_text": text,
                          "hf_ids": ids, "n_prompt": len(ids)}, ensure_ascii=False) + "\n")
print("[ok]")
