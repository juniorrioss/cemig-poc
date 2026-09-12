#!/usr/bin/env python3
"""
generate.py — Gera respostas de síntese sobre um PACK de contexto (oráculo ou retrieved).

Serve os dois papéis da tarefa com a MESMA lógica de síntese:
  --engine openai  : servidor OpenAI-compat (vLLM 27B do juiz OU llama-server do 2.6B).
  Modelo/URL/sampling configuráveis por flag.

Prompt de síntese: o "melhor prompt conhecido" (`numeros`) por default (achado do
bench/prompt_teto), com opção `baseline`.

Higiene: paralelo (ThreadPoolExecutor), checkpoint incremental, não-interativo,
procedência no metadata. Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "bench" / "prompt_teto"))

from prompts import VARIANTS  # noqa: E402 (baseline, numeros, ...)

# Sampling oficial LFM (model card) — igual à produção/sintese2.
LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}
# Determinístico para o teto reprodutível do 27B.
DET_SAMPLING = {"temperature": 0.0}


def synth(url: str, model: str, system: str, user: str, max_tokens: int,
          timeout: int, sampling: Dict[str, Any], think_off_kwargs: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        **sampling,
    }
    if model:
        payload["model"] = model
    if think_off_kwargs:
        # vLLM 27B (Qwen) desliga o thinking por chat_template_kwargs.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return {
        "response": (msg.get("content") or "").strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_s_gpu": round(wall, 3),
        "finish_reason": d["choices"][0].get("finish_reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    system = VARIANTS[args.variant]
    pack = json.loads(Path(args.pack).read_text(encoding="utf-8"))
    all_items = pack["items"]

    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING
    url = args.url.rstrip("/")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            if it.get("response") or it.get("__error__") is None:
                done[it["item_id"]] = it

    meta = {
        "task": "poc-slm-oraculo",
        "label": args.label, "model": args.model, "variant": args.variant,
        "context_source": pack["metadata"].get("corpus", pack["metadata"].get("retriever", "?")),
        "url": url, "sampling": sampling, "max_tokens": args.max_tokens,
        "system_prompt": system, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pack_meta": pack["metadata"],
    }

    def work(rec: Dict[str, Any]) -> Dict[str, Any]:
        user = (f"Contexto normativo consultado:\n{rec['context']}\n\n"
                f"Pergunta do eletricista:\n{rec['question']}")
        last_err = None
        for attempt in range(4):
            try:
                s = synth(url, args.model, system, user, args.max_tokens,
                          args.timeout, sampling, args.think_off)
                return {
                    "item_id": rec["item_id"], "question": rec["question"],
                    "doc": rec.get("doc", ""), "section": rec.get("section", ""),
                    "golden_answer": rec.get("golden_answer", ""),
                    "response": s["response"], "retrieval_hit": rec.get("retrieval_hit"),
                    "captain_case": rec.get("captain_case", False),
                    "prompt_tokens": s["prompt_tokens"],
                    "completion_tokens": s["completion_tokens"],
                    "wall_s_gpu": s["wall_s_gpu"], "finish_reason": s["finish_reason"],
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        return {"item_id": rec["item_id"], "question": rec["question"],
                "response": "", "__error__": str(last_err),
                "captain_case": rec.get("captain_case", False)}

    pending = [r for r in all_items if r["item_id"] not in done]
    results: Dict[str, Any] = dict(done)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r["item_id"] for r in pending}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["item_id"]] = rec
            completed += 1
            if completed % 20 == 0:
                order = {r["item_id"]: i for i, r in enumerate(all_items)}
                items_sorted = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
                out_path.write_text(json.dumps({"metadata": meta, "items": items_sorted},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  [{args.label}/{args.variant}] {completed}/{len(pending)}", flush=True)

    order = {r["item_id"]: i for i, r in enumerate(all_items)}
    items_sorted = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
    out_path.write_text(json.dumps({"metadata": meta, "items": items_sorted},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(items_sorted)
    errs = sum(1 for it in items_sorted if it.get("__error__"))
    avg_tok = sum(it.get("completion_tokens", 0) for it in items_sorted) / max(1, n)
    print(f"[ok] {args.label}/{args.variant}: {n} itens ({errs} erros), "
          f"tok médio={avg_tok:.0f} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="oracle_pack.json ou retrieved_pack.json")
    ap.add_argument("--url", required=True, help="http://127.0.0.1:PORT/v1 (llama-server) ou juiz")
    ap.add_argument("--model", default="", help="id do modelo (obrigatório p/ vLLM; vazio p/ llama-server)")
    ap.add_argument("--label", required=True, help="rótulo p/ metadata/nome (ex.: 27b, 2.6b_bf16, 2.6b_q4)")
    ap.add_argument("--variant", default="numeros", choices=list(VARIANTS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--deterministic", action="store_true", help="temp=0 (teto do 27B)")
    ap.add_argument("--think-off", action="store_true", help="chat_template_kwargs enable_thinking=false (vLLM)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
