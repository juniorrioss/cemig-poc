#!/usr/bin/env python3
"""smoke_q4km.py — Spark: smoke 20 operacionais no GGUF Q4_K_M do f4-r32-ep2.

Protocolo IDÊNTICO ao bench/lfm3_f7/spark/smoke_gguf.py (mesmos prompts op_d50 da F6, chat
template do merged com enable_thinking=False, seed 42, sampling LFM3 temp 0.1/top_k 50/rep 1.1,
llama.cpp v0.9.0 CUDA) — só troca o GGUF. Saída no mesmo schema (chave "q4_k_m").
"""
import json, time
from pathlib import Path
import requests
from transformers import AutoTokenizer

W = Path.home() / "cemig-poc/lfm3_f7"
OUT = Path.home() / "cemig-poc/lfm3_q4km"
tok = AutoTokenizer.from_pretrained(str(W / "merged"), trust_remote_code=True)
servers = {"q4_k_m": "http://127.0.0.1:8450"}
items = [json.loads(l) for l in open(W / "prompts_smoke20.jsonl")]
out = open(OUT / "smoke20_q4km_gen.jsonl", "w")
for it in items:
    text = tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    hf_ids = tok(text, add_special_tokens=False)["input_ids"]
    rec = {"key": it["key"], "id": it["id"], "gold_kind": it["gold_kind"], "n_prompt": len(hf_ids)}
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
    print(it["key"], rec["q4_k_m"]["tok_parity"], flush=True)
print("[ok]")
