#!/usr/bin/env python3
"""
concision_dev.py — Fase DEV do experimento de concisão do 2.6B (thinking-OFF).

Roda cada variação de system prompt (concision_prompts.VARIANTS) nas 20 perguntas do
smoke set (bench/data/smoke_qa_20.jsonl) — FORA das 151 reais e do holdout do classifier —
com o pipeline de retrieval idêntico ao app, para ESCOLHER a variante vencedora de concisão
antes de gastar as 151 completas + juiz.

Métrica de seleção (sem juiz, barata): estilo/concisão objetivos —
  - tokens médios de resposta (alvo <= 200);
  - frações que respeitam "<= 4 frases" e "sem markdown";
  - fração que cita norma+item inline (regex NR-xx + item numérico).
A qualidade factual final é medida só na variante escolhida, nas 151, pelo juiz vLLM.

Uso:
  python3 concision_dev.py --url http://127.0.0.1:8397 \
      --model-file LFM2.5-2.6B-Q4_0.gguf --out data/concision_dev.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

from concision_prompts import VARIANTS  # noqa: E402
from retrieval import HybridRetriever, format_context, retrieval_hit  # noqa: E402

SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}

# Detecta markdown/títulos/listas (proibidos): #, *, -, 1. no início de linha, negrito **.
_MD_RE = re.compile(r"(^|\n)\s*(#{1,6}\s|[*\-+]\s|\d+[.)]\s)|\*\*|__|```")
# Cita norma + item inline: "NR-10, item 10.5.1" / "NR-35 item 35.5" / "nr-06 6.3".
_CITE_RE = re.compile(r"\bnr[\s._-]*\d{1,2}\b.{0,20}?\b\d{1,2}(\.\d{1,3}){1,3}\b", re.IGNORECASE)


def load_qa(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_pair(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc": str(qa.get("doc", "") or "").lower(),
        "section": str(qa.get("section", "") or ""),
        "chunk_id": qa.get("chunk_id", -1),
        "relevant_chunk_ids": qa.get("relevant_chunk_ids", []),
    }


def count_sentences(text: str) -> int:
    """Contagem simples de frases por terminadores . ! ? (ignora abreviações de item)."""
    # Remove itens como "10.5.1" para não contar seus pontos como fim de frase.
    cleaned = re.sub(r"\d\.\d", "0", text)
    parts = [p for p in re.split(r"[.!?]+", cleaned) if p.strip()]
    return len(parts)


def synthesize(url: str, model: str, system: str, user: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        **SAMPLING,
    }
    t0 = time.perf_counter()
    res = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    res.raise_for_status()
    d = res.json()
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return {
        "response": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or ""),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": len((msg.get("reasoning_content") or "").split()),
        "wall_s_gpu": round(wall, 3),
        "finish_reason": d["choices"][0].get("finish_reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    qa = load_qa(Path(args.qa))
    retriever = HybridRetriever(args.db, args.model_pkl)
    print(f"DEV concisão: {len(qa)} perguntas smoke | variantes: {list(VARIANTS)}")

    results: Dict[str, Any] = {"metadata": {
        "phase": "dev-concision",
        "model_file": args.model_file,
        "reasoning": "off",
        "qa": Path(args.qa).name,
        "engine": "cuda (RTX 5070); latência GPU não vale p/ aparelho",
        "sampling": SAMPLING,
        "max_tokens": args.max_tokens,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "variants": {}}

    # Pré-computa retrieval uma vez (determinístico, igual p/ todas as variantes).
    retr_cache: Dict[str, Dict[str, Any]] = {}
    for it in qa:
        sr = retriever.search(it["question"], top_k=args.top_k)
        retr_cache[it["id"]] = sr

    for vkey, system in VARIANTS.items():
        items = []
        for it in qa:
            sr = retr_cache[it["id"]]
            context = format_context(sr["chunks"])
            user = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{it['question']}"
            syn = synthesize(args.url, args.model_file, system, user, args.max_tokens, args.timeout)
            resp = syn["response"]
            items.append({
                "item_id": it["id"],
                "question": it["question"],
                "response": resp,
                "completion_tokens": syn["completion_tokens"],
                "reasoning_tokens": syn["reasoning_tokens"],
                "n_sentences": count_sentences(resp),
                "has_markdown": bool(_MD_RE.search(resp)),
                "has_inline_cite": bool(_CITE_RE.search(resp)),
                "retrieval_hit": retrieval_hit(sr["chunks"], gold_pair(it)),
                "wall_s_gpu": syn["wall_s_gpu"],
            })
        n = len(items)
        toks = [i["completion_tokens"] for i in items]
        summary = {
            "n": n,
            "mean_tokens": round(sum(toks) / n, 1),
            "max_tokens": max(toks),
            "pct_le_4_sentences": round(100 * sum(1 for i in items if i["n_sentences"] <= 4) / n, 1),
            "pct_no_markdown": round(100 * sum(1 for i in items if not i["has_markdown"]) / n, 1),
            "pct_inline_cite": round(100 * sum(1 for i in items if i["has_inline_cite"]) / n, 1),
            "mean_sentences": round(sum(i["n_sentences"] for i in items) / n, 1),
        }
        results["variants"][vkey] = {"summary": summary, "items": items}
        print(f"[{vkey:11s}] tok_med={summary['mean_tokens']:6.1f} max={summary['max_tokens']:4d} "
              f"<=4frases={summary['pct_le_4_sentences']:5.1f}% sem_md={summary['pct_no_markdown']:5.1f}% "
              f"cita_inline={summary['pct_inline_cite']:5.1f}%")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8397")
    ap.add_argument("--model-file", default="LFM2.5-2.6B-Q4_0.gguf")
    ap.add_argument("--qa", default=str(_ROOT / "bench" / "data" / "smoke_qa_20.jsonl"))
    ap.add_argument("--db", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--model-pkl", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--out", default=str(_HERE / "data" / "concision_dev.json"))
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
