#!/usr/bin/env python3
"""
pipeline.py — Gera respostas E2E para as 151 perguntas reais com o pipeline híbrido do app
(classificador NR gated -> BM25 top-2 -> síntese com o system prompt de produção), para
três sintetizadores rodando na RTX 5070 (llama-server CUDA, -ngl 99).

Configs (brief):
  a. lfm1.2b     -> LFM2.5-1.2B-Instruct-QAD-Q4_0 (o EMBARCADO — baseline);
  b. lfm2.6b     -> LFM2.5-2.6B-Q4_0 (reasoning ON — como vem, é modelo de raciocínio);
  c. lfm2.6b_noth-> LFM2.5-2.6B-Q4_0 (reasoning OFF via --reasoning-budget 0).

REGRA DE VALIDADE (brief): qualidade em GPU vale; latência de GPU NÃO vale p/ aparelho.
Marcamos engine:cuda e reportamos tokens gerados (thinking incluso) p/ projetar latência
mobile pela tabela do device-bench. O servidor é gerido externamente (run_all.sh) para
reaproveitar o carregamento e permitir checkpoint incremental.

Higiene: timeout por request, checkpoint incremental (grava a cada item), procedência
registrada nos metadados. Sampling oficial Liquid: temp=0.1, top_k=50, repeat_penalty=1.05.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from retrieval import HybridRetriever, format_context, retrieval_hit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("judge151.pipeline")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

# System prompt de síntese IDÊNTICO ao AskPipeline.SYNTHESIS_SYSTEM_PROMPT (produção).
SYNTHESIS_SYSTEM_PROMPT = (
    "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras "
    "(NR-10, NR-06, NR-35, NR-12, NR-18 e demais NRs aplicáveis).\n"
    "Suas diretrizes mandatórias:\n"
    "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
    "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. "
    "Não adicione procedimentos não contidos nas normas.\n"
    "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
    "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
    "'Não sei com base nas normas consultadas.' Não tente adivinhar."
)

# Sampling oficial Liquid (brief).
SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}


def load_qa(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gold_pair(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc": str(qa.get("doc", "") or "").lower(),
        "section": str(qa.get("section", "") or ""),
        "chunk_id": qa.get("chunk_id", -1),
        "relevant_chunk_ids": qa.get("relevant_chunk_ids", []),
    }


def synthesize(url: str, model: str, user_content: str, max_tokens: int, timeout: int,
               system_prompt: str = SYNTHESIS_SYSTEM_PROMPT) -> Dict[str, Any]:
    """Chama o llama-server (Turno 2). Retorna resposta, tokens e latência de parede (GPU)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        "reasoning": (msg.get("reasoning_content") or ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": len((msg.get("reasoning_content") or "").split()),
        "wall_s_gpu": round(wall, 3),
        "finish_reason": d["choices"][0].get("finish_reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    qa = load_qa(Path(args.qa))
    logger.info("Carregadas %d perguntas de %s", len(qa), args.qa)

    # Concisão: permite trocar o system prompt de síntese (experimento do 2.6B thinking-OFF).
    system_prompt = SYNTHESIS_SYSTEM_PROMPT
    if getattr(args, "prompt_key", None) and args.prompt_key != "base":
        from concision_prompts import VARIANTS  # import local p/ não quebrar quem não usa
        system_prompt = VARIANTS[args.prompt_key]
        logger.info("System prompt de síntese: variante de concisão '%s'", args.prompt_key)

    retriever = HybridRetriever(args.db, args.model_pkl)
    logger.info("Classificador: %s | índice: %s", retriever.clf_name, Path(args.db).name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint incremental: retoma itens já feitos (vllm/VPN oscilam).
    done: Dict[str, Dict[str, Any]] = {}
    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            done[it["item_id"]] = it
        logger.info("Checkpoint: %d itens já processados serão reaproveitados.", len(done))

    meta = {
        "config_key": args.config_key,
        "model_file": args.model_file,
        "reasoning": args.reasoning,
        "engine": "cuda",  # REGRA DE VALIDADE: latência GPU não vale p/ aparelho
        "server_url": args.url,
        "sampling": SAMPLING,
        "max_tokens": args.max_tokens,
        "db": Path(args.db).name,
        "classifier": retriever.clf_name,
        "prompt_key": getattr(args, "prompt_key", "base"),
        "synthesis_system_prompt": system_prompt,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    items: List[Dict[str, Any]] = []
    for i, qa_item in enumerate(qa):
        qid = qa_item["id"]
        if qid in done:
            items.append(done[qid])
            continue

        raw_q = qa_item["question"]
        gold = gold_pair(qa_item)

        # Estágio 1+2: híbrido gated + BM25 top-2 (fala bruta).
        t_ret = time.perf_counter()
        sr = retriever.search(raw_q, top_k=args.top_k)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        chunks = sr["chunks"]
        hit = retrieval_hit(chunks, gold)

        # Turno 2: síntese com system prompt de produção.
        context = format_context(chunks)
        user_content = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{raw_q}"

        last_err = None
        syn = None
        for attempt in range(3):
            try:
                syn = synthesize(args.url, args.model_file, user_content, args.max_tokens, args.timeout, system_prompt)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Item %s tentativa %d falhou: %s", qid, attempt + 1, e)
                time.sleep(3)
        if syn is None:
            logger.error("Item %s falhou após 3 tentativas: %s", qid, last_err)
            break  # para p/ retomar via checkpoint quando o servidor voltar

        rec = {
            "item_id": qid,
            "question": raw_q,
            "golden_answer": qa_item.get("golden_answer", ""),
            "doc": gold["doc"],
            "section": gold["section"],
            "response": syn["response"],
            "reasoning_text": syn["reasoning"],
            "retrieval_hit": hit,
            "stage1_decision": sr["decision"],
            "fts_query": sr["fts_query"],
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

        # Checkpoint incremental a cada item.
        out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("[%s] %d/%d | hit=%s tok=%d(+%dth) %.1fs",
                        args.config_key, i + 1, len(qa), hit,
                        syn["completion_tokens"], syn["reasoning_tokens"], syn["wall_s_gpu"])

    # Grava final ordenado.
    order = {q["id"]: i for i, q in enumerate(qa)}
    items.sort(key=lambda r: order.get(r["item_id"], 0))
    out_path.write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_done = len(items)
    n_hit = sum(1 for it in items if it["retrieval_hit"])
    logger.info("Config %s concluída: %d/%d itens, retrieval_hit=%d (%.1f%%). Saída: %s",
                args.config_key, n_done, len(qa), n_hit, 100 * n_hit / max(1, n_done), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline E2E híbrido (151) -> respostas p/ o juiz.")
    ap.add_argument("--config-key", required=True, help="Ex.: lfm1.2b | lfm2.6b | lfm2.6b_noth")
    ap.add_argument("--model-file", required=True, help="Nome do modelo servido pelo llama-server (-m basename)")
    ap.add_argument("--reasoning", default="on", choices=["on", "off"], help="Só p/ metadados/rótulo.")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--qa", default=str(_ROOT / "corpus" / "qa_pairs_v2.jsonl"))
    ap.add_argument("--db", default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--model-pkl", default=str(_ROOT / "classifier" / "models" / "classic_winner.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--prompt-key", default="base", help="base | v1_rigido | v2_exemplo | v3_radio")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
