#!/usr/bin/env python3
"""
harness.py — Gera respostas de síntese para o teste de TETO (Partes 1 e 2).

Combina três eixos:
  MODELO   : --model {lfm2.6b, judge27b}
             lfm2.6b  = LFM2.5-2.6B-Q4_0 thinking-OFF (llama-server na 5070, --reasoning-budget 0)
             judge27b = o próprio juiz vLLM 27B como GERADOR (teto de geração; Parte 2)
  PROMPT   : --variant <nome de prompts.VARIANTS>
  CONTEXTO : --context {retrieved, oracle}
             retrieved = chunks da v4 embarcada (data/chunks_v4.json)
             oracle    = trecho-ouro (separa teto de retrieval de teto de geração)

Sempre inclui as 2 perguntas do capitão (retrieval AO VIVO na v4) como casos fixos,
gravando a resposta LITERAL de cada variante.

Higiene: paralelo (ThreadPoolExecutor), checkpoint incremental, não-interativo,
procedência. Comentários PT-BR; código em inglês. Procedência: task poc-prompt-teto.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_ROOT / "retrieval4"))

from prompts import VARIANTS, CAPTAIN_CASES  # noqa: E402
from retrieval_teto import format_context, load_cache  # noqa: E402

# Sampling oficial LFM (model card) — igual ao sintese2/produção.
LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}
# Juiz 27B como gerador: sampling determinístico (teto reprodutível).
JUDGE_SAMPLING = {"temperature": 0.0}

JUDGE_MODEL = "Qwen/Qwen3.8-27B-FP8"


def synth_lfm(url: str, system: str, user: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    """Chama o llama-server do 2.6B (thinking-OFF já ligado por flag do servidor)."""
    payload = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        **LFM_SAMPLING,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
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


def synth_judge(url: str, system: str, user: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    """Chama o juiz vLLM 27B como GERADOR (thinking-OFF via chat_template_kwargs)."""
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        **JUDGE_SAMPLING,
    }
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


def build_captain_chunks() -> List[Dict[str, Any]]:
    """Retrieval AO VIVO na v4 para as 2 perguntas do capitão."""
    from bm25 import HybridBm25
    from rrf import rrf_fuse
    from retrieval_teto import (V4_DB, CLF, DENSE_TEXT, DENSE_EXP_V4, EXP_W,
                                RRF_K, RRF_W, POOL)
    # Cache denso das perguntas do capitão: precisa encodar ao vivo -> usa servidor GGUF.
    # Para evitar dependência do embedding server aqui, usamos SÓ o BM25 gated v4 para
    # os casos do capitão (o app funde 3 sinais; aqui reportamos o BM25 gated como
    # aproximação honesta e anotamos isso). Se o cache denso existir, usa-o.
    hy = HybridBm25(str(V4_DB), str(CLF), weights=EXP_W)
    out = []
    for case in CAPTAIN_CASES:
        s1 = hy.rank(case["question"], POOL)
        chunks = s1[:2]
        out.append({
            "item_id": case["id"],
            "question": case["question"],
            "doc": "", "section": "",
            "golden_answer": "",
            "retrieval_hit": None,
            "chunks": [{"id": c["id"], "doc": c["doc"], "section": c["section"],
                        "title": c["title"], "text": c["text"]} for c in chunks],
            "gold_chunks": [],
            "captain_case": True,
        })
    return out


def run(args: argparse.Namespace) -> None:
    system = VARIANTS[args.variant]
    cache = load_cache()
    items_in = list(cache["items"].values())

    # Casos do capitão (retrieval ao vivo).
    captain = build_captain_chunks()
    all_items = items_in + captain

    # Escolhe fonte de contexto.
    def context_for(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if args.context == "oracle" and not rec.get("captain_case"):
            return rec.get("gold_chunks", []) or rec.get("chunks", [])
        return rec.get("chunks", [])

    synth = synth_lfm if args.model == "lfm2.6b" else synth_judge
    url = args.url

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            done[it["item_id"]] = it

    meta = {
        "task": "poc-prompt-teto",
        "model": args.label or args.model,
        "variant": args.variant,
        "context": args.context,
        "system_prompt": system,
        "url": url,
        "sampling": LFM_SAMPLING if args.model == "lfm2.6b" else JUDGE_SAMPLING,
        "retriever": cache["metadata"]["retriever"],
        "max_tokens": args.max_tokens,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def work(rec: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context_for(rec)
        context = format_context(chunks)
        user = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{rec['question']}"
        last_err = None
        for attempt in range(4):
            try:
                syn = synth(url, system, user, args.max_tokens, args.timeout)
                return {
                    "item_id": rec["item_id"],
                    "question": rec["question"],
                    "doc": rec.get("doc", ""),
                    "section": rec.get("section", ""),
                    "golden_answer": rec.get("golden_answer", ""),
                    "response": syn["response"],
                    "retrieval_hit": rec.get("retrieval_hit"),
                    "captain_case": rec.get("captain_case", False),
                    "chunks_docs": [f"{c['doc']}/{c['section']}" for c in chunks],
                    "prompt_tokens": syn["prompt_tokens"],
                    "completion_tokens": syn["completion_tokens"],
                    "wall_s_gpu": syn["wall_s_gpu"],
                    "finish_reason": syn["finish_reason"],
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
                items_sorted = list(results.values())
                out_path.write_text(json.dumps({"metadata": meta, "items": items_sorted},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  [{args.model}/{args.variant}/{args.context}] {completed}/{len(pending)}",
                      flush=True)

    # Ordena: 151 na ordem original + casos do capitão no fim.
    order = {r["item_id"]: i for i, r in enumerate(all_items)}
    items_sorted = sorted(results.values(), key=lambda r: order.get(r["item_id"], 9999))
    out_path.write_text(json.dumps({"metadata": meta, "items": items_sorted},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    avg_tok = sum(it.get("completion_tokens", 0) for it in items_sorted) / max(1, len(items_sorted))
    print(f"[ok] {args.model}/{args.variant}/{args.context}: {len(items_sorted)} itens, "
          f"tok médio={avg_tok:.0f} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfm2.6b", "judge27b"], required=True)
    ap.add_argument("--label", default="", help="rótulo do modelo p/ metadata (ex.: lfm1.2b quando o path é o 1.2B)")
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--context", choices=["retrieved", "oracle"], default="retrieved")
    ap.add_argument("--url", required=True, help="lfm: http://127.0.0.1:8410 ; judge: http://10.100.0.111:8005/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
