#!/usr/bin/env python3
"""gen_gguf.py — Spark: gera os 344 prompts op_d50 (texto congelado por render_prompts.py) num
llama-server (GGUF bf16 / Q8_0 / Q4_0 do f4-r32-ep2). Sampling da série LFM3 = smoke da F7
(temp 0.1, top_k 50, top_p 1, rep 1.1/64). Checa paridade de tokenização HF x llama.cpp.
Reentrante por key; paralelo (N requisições concorrentes = --parallel do servidor).

Uso: python gen_gguf.py --url http://127.0.0.1:8430 --fmt gguf_q4_0 --seed 42 --tag q4_0-r1
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

Q = Path.home() / "cemig-poc/lfm3_quant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rep", type=float, default=1.1)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(Q / "prompts_op344_rendered.jsonl")]
    outp = Q / "gen" / f"gen_{a.tag}.jsonl"
    outp.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["key"] for l in open(outp)} if outp.exists() else set()
    todo = [it for it in items if it["key"] not in done]
    print(f"[gen] {a.tag} pendentes={len(todo)}", flush=True)
    fh = open(outp, "a")

    def run(it):
        lt = requests.post(f"{a.url}/tokenize", json={"content": it["prompt_text"], "add_special": False,
                                                      "parse_special": True}).json()["tokens"]
        t0 = time.time()
        r = requests.post(f"{a.url}/completion", json={
            "prompt": it["prompt_text"], "n_predict": 768, "temperature": 0.1, "top_k": 50,
            "top_p": 1.0, "min_p": 0.0, "repeat_penalty": a.rep, "repeat_last_n": 64,
            "seed": a.seed, "cache_prompt": False, "special": True}, timeout=600).json()
        tm = r.get("timings", {})
        return {"key": it["key"], "id": it["id"], "engine_format": a.fmt, "tag": a.tag, "seed": a.seed,
                "tok_parity": lt == it["hf_ids"], "n_prompt_tokens": it["n_prompt"],
                "raw": r["content"], "n_gen_tokens": r.get("tokens_predicted"),
                "stopped_eos": r.get("stop_type") == "eos",
                "decode_tps": tm.get("predicted_per_second"), "prefill_tps": tm.get("prompt_per_second"),
                "wall_s": round(time.time() - t0, 2)}

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(run, todo):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"[gen] {n}/{len(todo)}", flush=True)
    print("[ok]", flush=True)


if __name__ == "__main__":
    main()
