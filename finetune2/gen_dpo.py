#!/usr/bin/env python3
"""
gen_dpo.py — Pares DPO (FASE B) da síntese: rejected = saída REAL do 2.6B com nota baixa E
chunk-certo presente; chosen = a MESMA resposta MINIMAMENTE EDITADA pelo 27B (conserta erro
factual/citação preservando o estilo do 2.6B). Minimalidade validada por difflib ratio >= 0.6;
pares onde o 27B reescreveu demais são descartados.

MURALHA 1 (CRÍTICA): as 604 respostas do 2.6B já julgadas (sintese2/judge151) foram TODAS
geradas sobre o holdout 151 (corpus/qa_pairs_v2.jsonl). Usá-las como fonte de treino violaria
'holdout fora de TUDO' e contaminaria a avaliação final nas mesmas 151. Portanto este script
GERA respostas frescas do 2.6B sobre o DEV-SET sintético limpo (retrieval3/data/dev_set.jsonl —
0 chunks do holdout, Jaccard<0.06 vs holdout), preservando EXATAMENTE a metodologia DPO do brief
(rejected=saída real ruim do 2.6B com chunk-certo; chosen=edição mínima do 27B). A decisão está
registrada no status/README.

Pipeline por item do dev-set:
  1. retrieval v3 (top-2) -> contexto; marca retrieval_hit (chunk-ouro no top-2);
  2. sintetiza com o 2.6B thinking-OFF (server local) usando o prompt de produção v1_rigido;
  3. juiz vLLM 27B (4 eixos) -> pass_gate;
  4. CANDIDATO A REJECTED: retrieval_hit == True E NOT pass_gate (prioriza falha de citação/fid);
  5. CHOSEN: edição MÍNIMA pelo 27B (conserta só o erro factual/citação, preserva estrutura/estilo);
  6. valida minimalidade (difflib.SequenceMatcher ratio >= --min-ratio, default 0.6) e que a edição
     mudou algo (ratio < 1.0); descarta fora da faixa.

Saída: data/dpo.jsonl {prompt, chosen, rejected, meta{...procedência, ratio, eixos}}.
Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(ROOT / "bench" / "judge151"))
sys.path.insert(0, str(ROOT / "bench"))

from common import (DATA_DIR, DEFAULT_VLLM_MODEL, DEFAULT_VLLM_URL,  # noqa: E402
                    SYNTHESIS_SYSTEM_PROMPT, call_vllm, extract_json, load_jsonl)
from retrieval_local import RetrieverLocal as RetrieverV3, format_context, retrieval_hit  # noqa: E402
from judge import call_llm_judge  # noqa: E402  (bench/judge.py)

# Sampling Liquid (idêntico à produção) para o 2.6B thinking-OFF.
LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}

EDIT_SYSTEM = (
    "Você é um auditor técnico especialista em Normas Regulamentadoras brasileiras. Recebe uma "
    "RESPOSTA gerada por um modelo e o CONTEXTO normativo + RESPOSTA-OURO. Sua tarefa é fazer a "
    "MENOR edição possível na RESPOSTA para corrigir APENAS erros factuais e de citação (norma/item "
    "errados ou ausentes), preservando ao máximo a estrutura, o vocabulário e o estilo originais.\n"
    "Regras rígidas:\n"
    "- NÃO reescreva do zero. Mantenha as mesmas frases; troque só o que estiver factualmente errado "
    "ou a citação incorreta/ausente.\n"
    "- Mantenha prosa corrida, no máximo 4 frases, sem markdown/listas.\n"
    "- A resposta corrigida deve citar a norma e o item CORRETOS conforme o contexto/ouro.\n"
    "- Se a resposta original já estiver factualmente correta, devolva-a praticamente inalterada.\n"
    "Responda SOMENTE um JSON: {\"resposta_corrigida\": \"...\"}."
)

EDIT_USER = (
    "CONTEXTO NORMATIVO:\n{context}\n\n"
    "RESPOSTA-OURO (referência de fato/citação):\n{gold}\n\n"
    "RESPOSTA DO MODELO (a ser minimamente corrigida):\n{resp}\n\n"
    "Devolva o JSON com \"resposta_corrigida\" — a MENOR edição que corrige fato/citação."
)


def synth_2_6b(url: str, system_prompt: str, user_content: str,
               max_tokens: int, timeout: int, retries: int = 3) -> Optional[str]:
    payload = {
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}],
        "max_tokens": max_tokens, **LFM_SAMPLING,
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return None


def gate_pass(m: Dict[str, Any]) -> bool:
    g = (m["acerto_factual"] * 0.35 + m["fidelidade_contexto"] * 0.30
         + m["citacao_fonte"] * 0.20 + m["qualidade_pt"] * 0.15)
    return (m["acerto_factual"] >= 3 and m["fidelidade_contexto"] >= 3
            and m["citacao_fonte"] >= 3 and g >= 3.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(DATA_DIR / "dpo_questions.jsonl"),
                    help="pool CLEAN de perguntas (NÃO holdout); default data/dpo_questions.jsonl")
    ap.add_argument("--server-url", default="http://127.0.0.1:8401", help="llama-server do 2.6B")
    ap.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    ap.add_argument("--vllm-model", default=DEFAULT_VLLM_MODEL)
    ap.add_argument("--out", default=str(DATA_DIR / "dpo.jsonl"))
    ap.add_argument("--raw", default=str(DATA_DIR / "dpo_raw.jsonl"),
                    help="respostas cruas do 2.6B + juízo (checkpoint/procedência)")
    ap.add_argument("--target", type=int, default=400, help="alvo de pares DPO (300-500)")
    ap.add_argument("--min-ratio", type=float, default=0.6, help="difflib ratio mínimo (minimalidade)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = load_jsonl(Path(args.dev))
    if args.limit:
        dev = dev[:args.limit]
    retriever = RetrieverV3()

    out_path = Path(args.out)
    raw_path = Path(args.raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint: reaproveita respostas cruas já geradas+julgadas.
    raw_done: Dict[str, Dict[str, Any]] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                r = json.loads(line)
                raw_done[r["item_id"]] = r
        print(f"checkpoint raw: {len(raw_done)} itens já processados", flush=True)

    raw_fh = raw_path.open("a", encoding="utf-8")
    if not raw_done:
        raw_fh.write("# " + json.dumps({
            "artifact": "dpo_raw", "task": "poc-dpo-sintese", "phase": "B",
            "source": "dev_set_synthetic_NOT_holdout", "synth_model": "LFM2.5-2.6B-Q4_0_thinkingOFF",
            "judge": args.vllm_model, "created_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False) + "\n")

    t0 = time.time()
    n_hit_low = 0
    for i, item in enumerate(dev):
        qid = item["id"]
        raw = raw_done.get(qid)
        if raw is None:
            raw_q = item["question"]
            sr = retriever.search(raw_q, top_k=2, item_id=qid)
            chunks = sr["chunks"]
            hit = retrieval_hit(chunks, item)
            context = format_context(chunks)
            user_content = (f"Contexto normativo consultado:\n{context}\n\n"
                            f"Pergunta do eletricista:\n{raw_q}")
            resp = synth_2_6b(args.server_url, SYNTHESIS_SYSTEM_PROMPT, user_content,
                              args.max_tokens, args.timeout)
            if resp is None:
                print(f"  falha síntese {qid}", flush=True)
                continue
            # Juízo (usa a resposta-ouro sintética do dev-set — o chunk-fonte).
            gold = _gold_answer_for(item, chunks)
            ev = call_llm_judge(question=raw_q, golden_answer=gold, model_response=resp,
                                doc_expected=item.get("doc", ""),
                                section_expected=item.get("section", ""),
                                cli_tool="vllm", vllm_url=args.vllm_url, vllm_model=args.vllm_model)
            raw = {
                "item_id": qid, "question": raw_q, "doc": item.get("doc", ""),
                "section": item.get("section", ""), "context": context,
                "golden_answer": gold, "response": resp, "retrieval_hit": hit,
                "metrics": ev, "pass_gate": gate_pass(ev),
                "chunks_docs": [f"{c['doc']}/{c['section']}" for c in chunks],
            }
            raw_fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
            raw_fh.flush()
            raw_done[qid] = raw
        if raw["retrieval_hit"] and not raw["pass_gate"]:
            n_hit_low += 1
        if (i + 1) % 20 == 0:
            print(f"  raw {i+1}/{len(dev)} | cand rejected (hit+low)={n_hit_low} "
                  f"{(i+1)/max(1e-9,time.time()-t0):.2f}/s", flush=True)
    raw_fh.close()

    # Seleção de rejected + edição mínima -> chosen.
    candidates = [r for r in raw_done.values() if r["retrieval_hit"] and not r["pass_gate"]]
    # Prioriza falha de citação/fidelidade (o gargalo do brief).
    candidates.sort(key=lambda r: (r["metrics"]["citacao_fonte"] + r["metrics"]["fidelidade_contexto"]))
    print(f"candidatos rejected (hit+low): {len(candidates)}", flush=True)

    pairs: List[Dict[str, Any]] = []
    n_too_edited = 0
    n_no_change = 0
    for r in candidates:
        if len(pairs) >= args.target:
            break
        edit_user = EDIT_USER.format(context=r["context"], gold=r["golden_answer"], resp=r["response"])
        try:
            raw_edit = call_vllm([{"role": "system", "content": EDIT_SYSTEM},
                                  {"role": "user", "content": edit_user}],
                                 url=args.vllm_url, model=args.vllm_model,
                                 temperature=0.2, max_tokens=512, thinking=False)
        except Exception as e:  # noqa: BLE001
            print(f"  falha edição {r['item_id']}: {e}", flush=True)
            continue
        obj = extract_json(raw_edit)
        chosen = ""
        if obj and "resposta_corrigida" in obj:
            chosen = " ".join(str(obj["resposta_corrigida"]).strip().split())
        if not chosen or len(chosen) < 15:
            continue
        ratio = difflib.SequenceMatcher(None, r["response"], chosen).ratio()
        if ratio >= 1.0 or r["response"].strip() == chosen.strip():
            n_no_change += 1
            continue
        if ratio < args.min_ratio:
            n_too_edited += 1
            continue
        # Constrói o prompt no formato de chat de produção (system+user do 2.6B).
        prompt_text = (f"{SYNTHESIS_SYSTEM_PROMPT}\n\nContexto normativo consultado:\n"
                       f"{r['context']}\n\nPergunta do eletricista:\n{r['question']}")
        pairs.append({
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": r["response"],
            "meta": {
                "item_id": r["item_id"], "doc": r["doc"], "section": r["section"],
                "diff_ratio": round(ratio, 4),
                "rejected_metrics": r["metrics"],
                "source": "dev_set_synthetic_NOT_holdout",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        })
        if len(pairs) % 25 == 0:
            print(f"  pares={len(pairs)}/{args.target} too_edited={n_too_edited} "
                  f"no_change={n_no_change}", flush=True)

    out_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                        encoding="utf-8")
    ratios = [p["meta"]["diff_ratio"] for p in pairs]
    print(f"DPO pronto: {len(pairs)} pares | descartes: too_edited={n_too_edited} "
          f"no_change={n_no_change} | ratio médio={sum(ratios)/max(1,len(ratios)):.3f} "
          f"| {(time.time()-t0)/60:.1f}min -> {out_path}", flush=True)


def _gold_answer_for(item: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    """O dev-set não traz golden_answer; usa o texto do chunk-ouro como referência de fato/citação.

    Constrói uma referência mínima: 'Conforme a NR-XX, item Y: <texto do chunk-fonte>' para o
    juiz e o editor ancorarem norma+item corretos. O chunk-ouro é o relevant_chunk_ids[0].
    """
    gold_ids = set(item.get("relevant_chunk_ids", []) or [])
    if item.get("chunk_id", -1) >= 0:
        gold_ids.add(item["chunk_id"])
    for c in chunks:
        if c["id"] in gold_ids:
            return (f"Conforme a {c['doc'].upper()}, item {c['section']}: {c['text']}")
    # fallback: usa doc/section declarados no item + primeiro chunk
    doc = str(item.get("doc", "")).upper()
    sec = item.get("section", "")
    return f"Referência: {doc}, item {sec}."


if __name__ == "__main__":
    main()
