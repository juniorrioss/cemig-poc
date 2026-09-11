#!/usr/bin/env python3
"""
rerank_llm.py — Etapa 4 (teto/candidato): reranker LLM listwise sobre top-N da fusão.

Dois usos:
  1. TETO (--base-url vLLM 27B): mede o ganho máximo alcançável reordenando os top-N —
     se nem o 27B sobe o gold p/ top-2, nenhum reranker mobile o fará (decisão de custo).
  2. CANDIDATO mobile (--base-url llama-server local com LFM2.5): mede custo real
     (N forwards curtos / 1 chamada listwise) do reranker embarcável.

O prompt mostra a pergunta + N candidatos (norma + seção + resumo). O LLM devolve a
ordem dos índices por relevância. Não-interativo, com timeout e retry.

Uso (teto):
    ../classifier/.venv/bin/python rerank_llm.py --cand results/fusion_cand768.json \
        --base-url http://10.100.0.111:8005/v1 --model Qwen/Qwen3.8-27B-FP8 \
        --out results/rerank_llm_ceiling.json --label 27B-ceiling
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from corpus.eval_retrieval import check_hit  # noqa: E402

SYSTEM = (
    "Você é um especialista em Normas Regulamentadoras (NRs) de segurança do trabalho. "
    "Dada a dúvida de um trabalhador e uma lista de trechos candidatos, ordene os trechos "
    "do MAIS ao MENOS relevante para responder a dúvida.\n"
    "Responda SOMENTE com os números dos candidatos na nova ordem, separados por vírgula "
    "(ex.: 3,1,5,2,...). Não explique."
)


def _load_gold() -> Dict[str, Dict[str, Any]]:
    gold = {}
    for name in ["corpus/qa_pairs_v2.jsonl", "bench/data/smoke_qa_20.jsonl"]:
        for line in (_ROOT / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gold[r["id"]] = {"doc": str(r.get("doc", "") or "").lower(),
                             "section": str(r.get("section", "") or ""),
                             "chunk_id": r.get("chunk_id", -1),
                             "relevant_chunk_ids": r.get("relevant_chunk_ids", [])}
    return gold


def call(base_url: str, model: str, question: str, cands: List[Dict[str, Any]],
         timeout: int = 90) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        snippet = c["text"][:280].replace("\n", " ")
        lines.append(f"[{i}] {c['doc'].upper()} {c['section']}: {snippet}")
    user = f"Dúvida: {question}\n\nCandidatos:\n" + "\n".join(lines) + "\n\nNova ordem (números):"
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 64, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    m = out["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "").strip()


def parse_order(raw: str, n: int) -> List[int]:
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    seen, order = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            order.append(x - 1)
    for i in range(n):  # completa o que faltar na ordem original
        if i not in seen and i not in order:
            order.append(i)
    return order[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="llm-rerank")
    args = ap.parse_args()

    cache = json.loads(Path(args.cand).read_text(encoding="utf-8"))
    gold = _load_gold()
    qa_ids = [q for q in cache if q.startswith("qa-")]

    reranked: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.time()

    def work(qid):
        obj = cache[qid]
        cands = obj["cands"]
        if not cands:
            return qid, []
        for attempt in range(3):
            try:
                raw = call(args.base_url, args.model, obj["question"], cands)
                order = parse_order(raw, len(cands))
                return qid, [cands[i] for i in order]
            except Exception:
                time.sleep(2)
        return qid, cands

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for qid, res in ex.map(work, qa_ids):
            reranked[qid] = res
    el = time.time() - t0

    def metrics(order_by_id, ids):
        n = len(ids); h1 = h2 = h5 = 0; mrr = 0.0
        for qid in ids:
            res = order_by_id.get(qid, [])
            rank = None
            for idx, r in enumerate(res[:5]):
                if check_hit(r, gold.get(qid, {})):
                    rank = idx + 1
                    break
            if rank:
                h1 += rank <= 1; h2 += rank <= 2; h5 += rank <= 5; mrr += 1.0 / rank
        return {"n": n, "recall_at_1": round(100*h1/n, 2), "recall_at_2": round(100*h2/n, 2),
                "recall_at_5": round(100*h5/n, 2), "mrr": round(mrr/n, 4)}

    m_pre = metrics({q: cache[q]["cands"] for q in qa_ids}, qa_ids)
    m_rr = metrics(reranked, qa_ids)

    print("\n" + "=" * 84)
    print(f"  ETAPA 4 — RERANKER LLM {args.label} — GATE holdout 151")
    print("=" * 84)
    print(f"| {'Configuração':<42} | {'R@1':>6} | {'R@2':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 44 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")
    print(f"| {'Fusão top-N (pré-rerank)':<42} | {m_pre['recall_at_1']:>5.1f}% | "
          f"{m_pre['recall_at_2']:>5.1f}% | {m_pre['recall_at_5']:>5.1f}% | {m_pre['mrr']:>7.4f} |")
    print(f"| {'Fusão + reranker LLM':<42} | {m_rr['recall_at_1']:>5.1f}% | "
          f"{m_rr['recall_at_2']:>5.1f}% | {m_rr['recall_at_5']:>5.1f}% | {m_rr['mrr']:>7.4f} |")
    print("=" * 84)
    print(f"\nTempo total {el:.1f}s ({el/len(qa_ids)*1000:.0f} ms/consulta, {args.workers} workers)")

    Path(args.out).write_text(json.dumps(
        {"label": args.label, "model": args.model,
         "fused_pre_rerank_151": m_pre, "reranked_151": m_rr,
         "wall_s": round(el, 1)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
