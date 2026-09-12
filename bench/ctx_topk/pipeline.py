#!/usr/bin/env python3
"""
pipeline.py — Gera respostas E2E das 151 reais com o pipeline v4 EMBARCADO variando
topK e modo-de-trecho (inteiro vs TRIM), para a bancada JANELA×TRECHOS (ordem 1 do capitão).

Sintetizador: LFM2.5-1.2B-Instruct-QAD-Q4_0 (o EMBARCADO) na RTX 5070 (llama-server CUDA,
-ngl 99, --parallel). System prompt = AskPipeline.SYNTHESIS_SYSTEM_PROMPT (produção, conciso).
Sampling oficial Liquid: temp=0.1, top_k=50, repeat_penalty=1.05, seed fixa (determinismo).

CÉLULAS DE QUALIDADE (o que varia a RESPOSTA):
  topK ∈ {2,3,4,5} × trecho ∈ {full, trim} = 8 células.
  n_ctx NÃO altera a resposta (com sampling determinístico, só dimensiona o KV cache; desde
  que o prompt caiba, a saída é idêntica) — por isso n_ctx é eixo de VIABILIDADE+LATÊNCIA,
  medido no aparelho na PARTE 2, não uma célula de qualidade. (Validado: mesmo prompt, saída
  idêntica em n_ctx 2048 vs 4096.)

TRIM: recorte determinístico por célula. max_chars por trecho é calibrado p/ caber `topK`
trechos no orçamento-alvo de contexto. Barato, sem LLM (trim.py).

Paralelismo: N requisições concorrentes ao llama-server (--parallel). Checkpoint incremental
(grava a cada item). Procedência nos metadados. Higiene: timeout + retry por request.

Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests

from retrieval_v4 import RetrieverV4, retrieval_hit
from trim import trim_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ctx_topk.pipeline")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

# System prompt IDÊNTICO ao AskPipeline.SYNTHESIS_SYSTEM_PROMPT (produção, conciso p/ voz).
SYNTHESIS_SYSTEM_PROMPT = (
    "Você é o assistente técnico de campo da CEMIG. O eletricista OUVE sua resposta por voz, então seja curto e direto.\n"
    "REGRAS OBRIGATÓRIAS (nunca viole):\n"
    "- Responda em NO MÁXIMO 4 frases curtas.\n"
    "- PROIBIDO usar markdown, títulos, negrito, listas, bullets ou numeração. Escreva em prosa corrida.\n"
    "- Cite a norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...').\n"
    "- Baseie-se EXCLUSIVAMENTE no contexto normativo fornecido; não invente procedimentos.\n"
    "- Se o contexto não responder, diga apenas: 'Não sei com base nas normas consultadas.'"
)

SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05, "seed": 42}

# max_chars por trecho no modo TRIM, calibrado p/ caber topK trechos no orçamento de contexto.
# Orçamento-alvo de contexto de trechos ~ 900 tok ≈ 3420 chars (1 tok ~ 3.8 chars, estimador
# do AskPipeline). Divide-se pelo topK; piso de 500 chars p/ não destruir a citação.
TRIM_BUDGET_CHARS = 3420


def load_qa(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_pair(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc": str(qa.get("doc", "") or "").lower(),
        "section": str(qa.get("section", "") or ""),
        "chunk_id": qa.get("chunk_id", -1),
        "relevant_chunk_ids": qa.get("relevant_chunk_ids", []),
    }


def format_context(chunks: List[Dict[str, Any]], raw_q: str, trim: bool, top_k: int) -> str:
    """Formata o contexto IDÊNTICO ao AskPipeline.kt: '[i] (doc - section - title):\\ntext'.

    Se trim=True, recorta cada texto determinísticamente p/ caber top_k trechos no orçamento.
    """
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    max_chars = max(500, TRIM_BUDGET_CHARS // max(1, top_k)) if trim else None
    parts = []
    for idx, c in enumerate(chunks):
        text = c["text"]
        if trim and max_chars is not None:
            text = trim_chunk(text, raw_q, max_chars)
        parts.append(f"[{idx + 1}] ({c['doc']} - {c['section']} - {c['title']}):\n{text}")
    return "\n\n".join(parts)


def synthesize(url: str, model: str, user_content: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_s_gpu": round(wall, 3),
        "finish_reason": d["choices"][0].get("finish_reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    qa = load_qa(Path(args.qa))
    logger.info("Carregadas %d perguntas", len(qa))
    trim = args.trecho == "trim"
    top_k = args.top_k

    # Retriever v4 embarcado — encoda todas as queries de uma vez.
    questions_by_id = {q["id"]: q["question"] for q in qa}
    retriever = RetrieverV4(questions_by_id, emb_url=args.emb_url)
    logger.info("Retriever v4 (RRF k=10 w=2,1,2) | clf: %s", retriever.clf_name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: Dict[str, Dict[str, Any]] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            done[it["item_id"]] = it
        logger.info("Checkpoint: %d itens reaproveitados", len(done))

    meta = {
        "config_key": args.config_key,
        "top_k": top_k,
        "trecho": args.trecho,
        "trim_budget_chars": TRIM_BUDGET_CHARS if trim else None,
        "model_file": args.model_file,
        "engine": "cuda",  # latência GPU não vale p/ aparelho
        "retriever": "v4_rrf3 (k=10 w=2,1,2, índice android/index.db)",
        "sampling": SAMPLING,
        "max_tokens": args.max_tokens,
        "synthesis_system_prompt": SYNTHESIS_SYSTEM_PROMPT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1) Retrieval (serial, barato) + montagem de prompt por item.
    todo = []
    for qa_item in qa:
        qid = qa_item["id"]
        if qid in done:
            continue
        raw_q = qa_item["question"]
        gold = gold_pair(qa_item)
        sr = retriever.search(qid, raw_q, top_k=top_k)
        chunks = sr["chunks"]
        hit = retrieval_hit(chunks, gold)
        context = format_context(chunks, raw_q, trim, top_k)
        user_content = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{raw_q}"
        todo.append({
            "qa_item": qa_item, "gold": gold, "chunks": chunks, "hit": hit,
            "decision": sr["decision"], "user_content": user_content,
        })

    # 2) Síntese PARALELA (--parallel do llama-server).
    def work(t: Dict[str, Any]) -> Dict[str, Any]:
        qa_item = t["qa_item"]
        last_err = None
        for attempt in range(3):
            try:
                syn = synthesize(args.url, args.model_file, t["user_content"], args.max_tokens, args.timeout)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2)
        else:
            raise RuntimeError(f"item {qa_item['id']} falhou: {last_err}")
        return {
            "item_id": qa_item["id"],
            "question": qa_item["question"],
            "golden_answer": qa_item.get("golden_answer", ""),
            "doc": t["gold"]["doc"],
            "section": t["gold"]["section"],
            "response": syn["response"],
            "retrieval_hit": t["hit"],
            "stage1_decision": t["decision"],
            "chunks_docs": [f"{c['doc']}/{c['section']}" for c in t["chunks"]],
            "prompt_tokens": syn["prompt_tokens"],
            "completion_tokens": syn["completion_tokens"],
            "wall_s_gpu": syn["wall_s_gpu"],
            "finish_reason": syn["finish_reason"],
        }

    items = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(work, t): t for t in todo}
            n = 0
            for fut in as_completed(futures):
                rec = fut.result()
                items.append(rec)
                done[rec["item_id"]] = rec
                n += 1
                if n % 20 == 0 or n == 1:
                    logger.info("[%s] %d/%d | last hit=%s ptok=%d ctok=%d",
                                args.config_key, n, len(todo), rec["retrieval_hit"],
                                rec["prompt_tokens"], rec["completion_tokens"])
                    order = {q["id"]: i for i, q in enumerate(qa)}
                    items.sort(key=lambda r: order.get(r["item_id"], 0))
                    out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")

    order = {q["id"]: i for i, q in enumerate(qa)}
    items.sort(key=lambda r: order.get(r["item_id"], 0))
    out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_hit = sum(1 for it in items if it["retrieval_hit"])
    avg_ptok = sum(it["prompt_tokens"] for it in items) / max(1, len(items))
    logger.info("Config %s: %d itens, retrieval_hit=%d (%.1f%%), prompt_tok médio=%.0f. -> %s",
                args.config_key, len(items), n_hit, 100 * n_hit / max(1, len(items)), avg_ptok, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, help="Ex.: k2_full | k5_trim")
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--trecho", choices=["full", "trim"], required=True)
    ap.add_argument("--model-file", default="LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf")
    ap.add_argument("--url", default="http://127.0.0.1:8410")
    ap.add_argument("--emb-url", default="http://127.0.0.1:8399")
    ap.add_argument("--qa", default=str(_ROOT / "corpus" / "qa_pairs_v2.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
