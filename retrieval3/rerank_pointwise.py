#!/usr/bin/env python3
"""
rerank_pointwise.py — Etapa 4 (candidato mobile): reranker pointwise sim/não.

Candidato exato do brief: "o próprio LFM2.5 com prompt de relevância sim/não por chunk
(meça custo real: N forwards curtos)". Para cada candidato, pergunta ao LLM se o trecho
responde à dúvida e usa a probabilidade de 'sim' (via logprobs) como score de reordenação.

Framing query↔query: apresenta a EXPANSÃO coloquial do chunk (perguntas que ele responde),
regime em que o modelo pequeno decide melhor do que sobre o juridiquês normativo.

Mede o GATE no holdout 151 e o custo real (N forwards curtos por consulta).

Uso (llama-server LFM2.5 local):
    ../classifier/.venv/bin/python rerank_pointwise.py --cand results/fusion3_cand.json \
        --base-url http://127.0.0.1:8090/v1 --doc-field expansion \
        --out results/rerank_pw_lfm.json --label LFM2.5-pointwise
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
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
    "Você é um classificador de relevância para busca em Normas Regulamentadoras (NRs). "
    "Dada a dúvida de um trabalhador e um trecho, responda se o trecho ajuda a responder "
    "a dúvida. Responda APENAS 'sim' ou 'não'."
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


def score_yes(base_url: str, model: str, question: str, doc: str, timeout: int = 60) -> float:
    """Retorna P(sim) via logprobs do 1º token; fallback binário se sem logprobs."""
    user = f"Dúvida: {question}\n\nTrecho: {doc[:500]}\n\nO trecho responde a dúvida (sim/não)?"
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 2, "stream": False,
        "logprobs": True, "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    choice = out["choices"][0]
    content = (choice["message"].get("content") or "").strip().lower()
    # Tenta extrair logprobs do 1º token
    try:
        top = choice["logprobs"]["content"][0]["top_logprobs"]
        p_yes = p_no = 0.0
        for t in top:
            tok = t["token"].strip().lower()
            p = math.exp(t["logprob"])
            if tok.startswith("sim") or tok in ("s", "yes"):
                p_yes += p
            elif tok.startswith("não") or tok.startswith("nao") or tok in ("n", "no"):
                p_no += p
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)
    except (KeyError, IndexError, TypeError):
        pass
    # Fallback: decisão binária pelo texto
    return 1.0 if content.startswith("sim") else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--doc-field", default="expansion", choices=["text", "expansion"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="pointwise")
    args = ap.parse_args()

    cache = json.loads(Path(args.cand).read_text(encoding="utf-8"))
    gold = _load_gold()
    qa_ids = [q for q in cache if q.startswith("qa-")]

    reranked: Dict[str, List[Dict[str, Any]]] = {}
    total_pairs = 0
    t0 = time.time()

    def work(qid):
        obj = cache[qid]
        cands = obj["cands"]
        if not cands:
            return qid, [], 0
        scored = []
        for c in cands:
            doc = c.get("expansion") if args.doc_field == "expansion" else c["text"]
            doc = doc or c["text"]
            try:
                s = score_yes(args.base_url, args.model, obj["question"], doc)
            except Exception:
                s = 0.0
            scored.append((c, s))
        # Estável: mantém ordem da fusão como desempate
        order = sorted(enumerate(scored), key=lambda x: (-x[1][1], x[0]))
        return qid, [c for _, (c, _) in order], len(cands)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for qid, res, npairs in ex.map(work, qa_ids):
            reranked[qid] = res
            total_pairs += npairs
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
    ms_query = el / len(qa_ids) * 1000

    print("\n" + "=" * 84)
    print(f"  ETAPA 4 — RERANKER POINTWISE {args.label} — GATE holdout 151 (field={args.doc_field})")
    print("=" * 84)
    print(f"| {'Configuração':<42} | {'R@1':>6} | {'R@2':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 44 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")
    print(f"| {'Fusão 3-sinais (pré-rerank)':<42} | {m_pre['recall_at_1']:>5.1f}% | "
          f"{m_pre['recall_at_2']:>5.1f}% | {m_pre['recall_at_5']:>5.1f}% | {m_pre['mrr']:>7.4f} |")
    print(f"| {'Fusão + reranker pointwise':<42} | {m_rr['recall_at_1']:>5.1f}% | "
          f"{m_rr['recall_at_2']:>5.1f}% | {m_rr['recall_at_5']:>5.1f}% | {m_rr['mrr']:>7.4f} |")
    print("=" * 84)
    print(f"\nCusto: {el:.1f}s total, {ms_query:.0f} ms/consulta ({args.workers} workers, "
          f"{total_pairs/len(qa_ids):.0f} forwards/consulta)")

    Path(args.out).write_text(json.dumps(
        {"label": args.label, "model": args.model, "doc_field": args.doc_field,
         "fused_pre_rerank_151": m_pre, "reranked_151": m_rr,
         "cost": {"wall_s": round(el, 1), "ms_per_query_wall": round(ms_query, 1),
                  "forwards_per_query": round(total_pairs/len(qa_ids), 1),
                  "workers": args.workers}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
