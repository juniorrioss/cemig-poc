#!/usr/bin/env python3
"""
harness.py — Shootout de sintetizadores (síntese v2) sobre o retrieval v3 (fusão RRF 3-sinais)
nas 151 perguntas reais. Gera respostas E2E para cada CANDIDATO x CONDIÇÃO, contra um
llama-server já no ar (RTX 5070, -ngl 99), com checkpoint incremental.

CANDIDATOS (o servidor decide o modelo/flags; aqui informamos família p/ sampling+thinking):
  - lfm1.2b       LFM2.5-1.2B-Instruct-QAD-Q4_0  (embarcado; sampling Liquid; sem thinking)
  - lfm2.6b_noth  LFM2.5-2.6B-Q4_0 thinking-OFF  (--reasoning-budget 0 no servidor)
  - minicpm2b     MiniCPM5-2B-Q4_K_M             (raciocínio; enable_thinking=false; temp 1.0/top_p .95)
  - minicpm1b     MiniCPM5-1B-Q4_K_M             (idem 2B)

CONDIÇÕES (prompts.py):
  - baseline  C1: system prompt conciso atual (citação inline pelo modelo)
  - full      C1+C2+C3+C4: few-shot + citação estruturada (anexada pelo app) + ordenação
              (melhor chunk por último). Ablações: fewshot | citation | ordering.

REGRA DE VALIDADE (brief): qualidade em GPU vale; latência de GPU NÃO vale p/ aparelho.
Marcamos engine:cuda e reportamos tokens (thinking incluso) p/ projeção mobile.

Retrieval: RetrieverV3 (mesmo do judge151, cache denso da 5070 + BM25-gated-exp ao vivo).
Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "bench" / "judge151"))

from prompts import condition_config  # noqa: E402
from retrieval_v3 import RetrieverV3, format_context, retrieval_hit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sintese2.harness")

# Sampling oficial por família (model cards).
SAMPLING = {
    "lfm": {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05},
    "minicpm": {"temperature": 1.0, "top_p": 0.95},
}

# Método de desligar raciocínio por família:
#   lfm      -> via flag do servidor (--reasoning-budget 0); nada no payload.
#   minicpm  -> chat_template_kwargs enable_thinking=false no payload.
_NR_ITEM = re.compile(r"^\d+(?:\.\d+)*$")


def load_qa(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_pair(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc": str(qa.get("doc", "") or "").lower(),
        "section": str(qa.get("section", "") or ""),
        "chunk_id": qa.get("chunk_id", -1),
        "relevant_chunk_ids": qa.get("relevant_chunk_ids", []),
    }


def structured_source(chunk: Dict[str, Any]) -> str:
    """Monta 'Fonte: NR-XX, item Y.Y.Y' a partir dos metadados do chunk (simula o app)."""
    doc = str(chunk.get("doc", "")).upper().replace("NR-", "NR-")
    section = str(chunk.get("section", "")).strip()
    if section and _NR_ITEM.match(section):
        return f"Fonte: {doc}, item {section}."
    if section:
        return f"Fonte: {doc}, {section}."
    return f"Fonte: {doc}."


def synthesize(url: str, family: str, thinking_off_via_payload: bool, system_prompt: str,
               user_content: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    """Chama o llama-server. Sampling por família; thinking-off via payload p/ MiniCPM."""
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        **SAMPLING[family],
    }
    if thinking_off_via_payload:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    res = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    res.raise_for_status()
    d = res.json()
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    reasoning = msg.get("reasoning_content") or ""
    return {
        "response": (msg.get("content") or "").strip(),
        "reasoning": reasoning,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": len(reasoning.split()) if reasoning else 0,
        "wall_s_gpu": round(wall, 3),
        "finish_reason": d["choices"][0].get("finish_reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    qa = load_qa(Path(args.qa))
    cfg = condition_config(args.condition)
    system_prompt = cfg["system_prompt"]
    structured = cfg["structured_citation"]
    best_last = cfg["best_last"]
    logger.info("Config %s | condição %s | family=%s | structured_cit=%s | best_last=%s",
                args.config_key, args.condition, args.family, structured, best_last)

    retriever = RetrieverV3()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: Dict[str, Dict[str, Any]] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            done[it["item_id"]] = it
        logger.info("Checkpoint: %d itens reaproveitados.", len(done))

    meta = {
        "config_key": args.config_key,
        "condition": args.condition,
        "model_file": args.model_file,
        "family": args.family,
        "thinking": args.thinking,
        "engine": "cuda",  # latência GPU não vale p/ aparelho
        "server_url": args.url,
        "sampling": SAMPLING[args.family],
        "structured_citation": structured,
        "best_chunk_last": best_last,
        "retriever": "v3_rrf3",
        "max_tokens": args.max_tokens,
        "system_prompt": system_prompt,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    thinking_off_payload = args.family == "minicpm" and args.thinking == "off"

    items: List[Dict[str, Any]] = []
    for i, qa_item in enumerate(qa):
        qid = qa_item["id"]
        if qid in done:
            items.append(done[qid])
            continue

        raw_q = qa_item["question"]
        gold = gold_pair(qa_item)

        t_ret = time.perf_counter()
        sr = retriever.search(raw_q, top_k=args.top_k, item_id=qid)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        chunks = sr["chunks"]
        hit = retrieval_hit(chunks, gold)

        # C4 — ORDENAÇÃO: melhor chunk por último (mais perto da pergunta).
        # O RRF devolve o melhor em chunks[0]; a fonte estruturada cita o melhor (rank 1).
        best_chunk = chunks[0] if chunks else None
        ctx_chunks = list(reversed(chunks)) if best_last else list(chunks)
        context = format_context(ctx_chunks)
        user_content = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{raw_q}"

        syn = None
        last_err = None
        for attempt in range(3):
            try:
                syn = synthesize(args.url, args.family, thinking_off_payload, system_prompt,
                                 user_content, args.max_tokens, args.timeout)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Item %s tentativa %d: %s", qid, attempt + 1, e)
                time.sleep(3)
        if syn is None:
            logger.error("Item %s falhou após 3 tentativas: %s", qid, last_err)
            break

        # C3 — CITAÇÃO ESTRUTURADA: anexa a fonte do melhor chunk (o app faria isso).
        response_final = syn["response"]
        appended = ""
        if structured and best_chunk is not None and response_final:
            # Não anexa se o modelo recusou ("não sei").
            if "não sei com base" not in response_final.lower():
                appended = " " + structured_source(best_chunk)
                response_final = response_final.rstrip() + appended

        rec = {
            "item_id": qid,
            "question": raw_q,
            "golden_answer": qa_item.get("golden_answer", ""),
            "doc": gold["doc"],
            "section": gold["section"],
            "response": response_final,
            "response_raw": syn["response"],
            "appended_source": appended.strip(),
            "reasoning_text": syn["reasoning"],
            "retrieval_hit": hit,
            "stage1_decision": sr["decision"],
            "chunks_docs": [f"{c['doc']}/{c['section']}" for c in chunks],
            "retrieval_ms": round(ret_ms, 2),
            "prompt_tokens": syn["prompt_tokens"],
            "completion_tokens": syn["completion_tokens"],
            "reasoning_tokens": syn["reasoning_tokens"],
            "wall_s_gpu": syn["wall_s_gpu"],
            "finish_reason": syn["finish_reason"],
        }
        items.append(rec)
        done[qid] = rec
        out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        if (i + 1) % 15 == 0 or i == 0:
            logger.info("[%s/%s] %d/%d | hit=%s tok=%d(+%dth) %.1fs",
                        args.config_key, args.condition, i + 1, len(qa), hit,
                        syn["completion_tokens"], syn["reasoning_tokens"], syn["wall_s_gpu"])

    order = {q["id"]: i for i, q in enumerate(qa)}
    items.sort(key=lambda r: order.get(r["item_id"], 0))
    out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_hit = sum(1 for it in items if it["retrieval_hit"])
    avg_tok = sum(it["completion_tokens"] for it in items) / max(1, len(items))
    logger.info("Config %s/%s concluída: %d itens, hit=%d (%.1f%%), tok médio=%.0f. -> %s",
                args.config_key, args.condition, len(items), n_hit,
                100 * n_hit / max(1, len(items)), avg_tok, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["baseline", "full", "fewshot", "citation", "ordering"])
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--family", required=True, choices=["lfm", "minicpm"])
    ap.add_argument("--thinking", default="off", choices=["on", "off"])
    ap.add_argument("--url", default="http://127.0.0.1:8399")
    ap.add_argument("--qa", default=str(_ROOT / "corpus" / "qa_pairs_v2.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
