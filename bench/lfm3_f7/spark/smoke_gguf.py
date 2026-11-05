#!/usr/bin/env python3
"""smoke_gguf.py — Spark: smoke de 20 perguntas operacionais bf16-GGUF vs Q4_0-GGUF.

Renderiza com o chat template HF do merged (enable_thinking=False, == gen_e2e da F6), checa
paridade de tokenização HF x llama.cpp (/tokenize) e gera nos 2 llama-server com o MESMO seed
e o sampling da série LFM3 (temp 0.1, top_k 50, rep 1.1).
"""
import json, sys, time
from pathlib import Path
import requests
from transformers import AutoTokenizer

W = Path.home() / "cemig-poc/lfm3_f7"
tok = AutoTokenizer.from_pretrained(str(W / "merged"), trust_remote_code=True)
servers = {"bf16": "http://127.0.0.1:8420", "q4_0": "http://127.0.0.1:8421"}
items = [json.loads(l) for l in open(W / "prompts_smoke20.jsonl")]
out = open(W / "smoke20_gen.jsonl", "w")
for it in items:
    text = tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    hf_ids = tok(text, add_special_tokens=False)["input_ids"]
    rec = {"key": it["key"], "id": it["id"], "gold_kind": it["gold_kind"], "n_prompt": len(hf_ids),
           "prompt_text": text}
    for name, url in servers.items():
        lt = requests.post(f"{url}/tokenize", json={"content": text, "add_special": False,
                                                     "parse_special": True}).json()["tokens"]
        t0 = time.time()
        r = requests.post(f"{url}/completion", json={
            "prompt": text, "n_predict": 768, "temperature": 0.1, "top_k": 50, "top_p": 1.0,
            "min_p": 0.0, "repeat_penalty": 1.1, "repeat_last_n": 64, "seed": 42,
            "cache_prompt": False, "special": True}).json()
        rec[name] = {"tok_parity": lt == hf_ids, "n_tok_llama": len(lt), "answer_raw": r["content"],
                     "stop": r.get("stop_type"), "n_gen": r.get("tokens_predicted"),
                     "wall_s": round(time.time() - t0, 2)}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    print(it["key"], rec["bf16"]["tok_parity"], rec["q4_0"]["tok_parity"], flush=True)
print("[ok]")
