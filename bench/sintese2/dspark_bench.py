#!/usr/bin/env python3
"""
dspark_bench.py — Mede o speedup de DECODE da decodificação especulativa DSpark no MiniCPM5-2B
(draft MiniCPM5-2B-DSpark) na RTX 5070, comparando dois servidores llama-server já no ar:
um SEM draft e um COM draft. Usa prompts RAG reais (10 primeiras das 151, contexto v3 topk2)
e mede tokens/s de decode (completion_tokens / wall) — média sobre as amostras.

REGRA DE VALIDADE: a latência de GPU não vale p/ o aparelho; o que interessa é o FATOR de
speedup de decode (draft on/off) e a aplicabilidade mobile (documentada no README).

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "bench" / "judge151"))

from retrieval_v3 import RetrieverV3, format_context  # noqa: E402

SYS = ("Você é o assistente técnico de campo da CEMIG. Responda em no máximo 4 frases, "
       "em prosa corrida, citando a norma e o item. Use só o contexto fornecido.")


def gen(url: str, sys_p: str, user: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return {"tok": u.get("completion_tokens", 0), "wall": wall}


def bench(url: str, samples: list, max_tokens: int, timeout: int) -> dict:
    tot_tok, tot_wall = 0, 0.0
    per = []
    for user in samples:
        r = gen(url, SYS, user, max_tokens, timeout)
        tot_tok += r["tok"]
        tot_wall += r["wall"]
        per.append(r["tok"] / r["wall"] if r["wall"] else 0)
    return {"total_tok": tot_tok, "total_wall": round(tot_wall, 2),
            "decode_tok_s": round(tot_tok / tot_wall, 1) if tot_wall else 0,
            "mean_per_req_tok_s": round(sum(per) / len(per), 1) if per else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-nodraft", required=True)
    ap.add_argument("--url-draft", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=str(_HERE / "data" / "dspark_bench.json"))
    args = ap.parse_args()

    qa = [json.loads(l) for l in (_ROOT / "corpus" / "qa_pairs_v2.jsonl").read_text().splitlines() if l.strip()]
    retr = RetrieverV3()
    samples = []
    for q in qa[: args.n]:
        sr = retr.search(q["question"], top_k=2, item_id=q["id"])
        ctx = format_context(sr["chunks"])
        samples.append(f"Contexto normativo consultado:\n{ctx}\n\nPergunta do eletricista:\n{q['question']}")

    print(f"Warmup + bench SEM draft ({args.n} amostras)...")
    gen(args.url_nodraft, SYS, samples[0], 32, args.timeout)  # warmup
    no = bench(args.url_nodraft, samples, args.max_tokens, args.timeout)
    print(f"Warmup + bench COM draft DSpark ({args.n} amostras)...")
    gen(args.url_draft, SYS, samples[0], 32, args.timeout)  # warmup
    dr = bench(args.url_draft, samples, args.max_tokens, args.timeout)

    speedup = round(dr["decode_tok_s"] / no["decode_tok_s"], 2) if no["decode_tok_s"] else 0
    report = {"engine": "cuda (RTX 5070); latência GPU não vale p/ aparelho",
              "n_samples": args.n, "max_tokens": args.max_tokens,
              "no_draft": no, "dspark_draft": dr, "decode_speedup": speedup}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== DSpark speculative decoding (RTX 5070) =====")
    print(f"SEM draft:  decode {no['decode_tok_s']} tok/s (média/req {no['mean_per_req_tok_s']})")
    print(f"COM DSpark: decode {dr['decode_tok_s']} tok/s (média/req {dr['mean_per_req_tok_s']})")
    print(f"SPEEDUP de decode: {speedup}×")
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
