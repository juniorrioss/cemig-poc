#!/usr/bin/env python3
"""
rerank.py — Etapa 4: reranker cross-encoder mobile sobre os top-N da fusão.

Reordena os candidatos da fusão (dump_fusion.py) com um cross-encoder multilíngue
destilado (mMARCO inclui PT). Mede o GATE no holdout 151 e o custo real: N forwards
curtos (query+chunk). Candidato mobile-first: cross-encoder MiniLM (int8 exportável).

Roda no .venv-train (torch cu128). Reordena o cache de candidatos e grava a nova ordem;
a métrica é calculada aqui mesmo (reusa check_hit) sem depender de sklearn.

Uso:
    ../.venv-train/bin/python rerank.py \
        --cand results/fusion_cand.json \
        --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
        --out results/rerank_mminilm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from corpus.eval_retrieval import check_hit  # noqa: E402


def _load_gold() -> Dict[str, Dict[str, Any]]:
    gold = {}
    for name in ["corpus/qa_pairs_v2.jsonl", "bench/data/smoke_qa_20.jsonl"]:
        for line in (_ROOT / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gold[r["id"]] = {
                "doc": str(r.get("doc", "") or "").lower(),
                "section": str(r.get("section", "") or ""),
                "chunk_id": r.get("chunk_id", -1),
                "relevant_chunk_ids": r.get("relevant_chunk_ids", []),
            }
    return gold


def _metrics(order_by_id: Dict[str, List[Dict[str, Any]]], gold: Dict[str, Dict[str, Any]],
             ids: List[str]) -> Dict[str, float]:
    n = len(ids)
    h1 = h2 = h5 = 0
    mrr = 0.0
    for qid in ids:
        res = order_by_id.get(qid, [])
        g = gold.get(qid, {})
        rank = None
        for idx, r in enumerate(res[:5]):
            if check_hit(r, g):
                rank = idx + 1
                break
        if rank is not None:
            if rank <= 1:
                h1 += 1
            if rank <= 2:
                h2 += 1
            if rank <= 5:
                h5 += 1
            mrr += 1.0 / rank
    return {"n": n, "recall_at_1": round(100 * h1 / n, 2),
            "recall_at_2": round(100 * h2 / n, 2),
            "recall_at_5": round(100 * h5 / n, 2), "mrr": round(mrr / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--doc-field", default="text", choices=["text", "expansion", "both", "max"],
                    help="o que apresentar ao CE: texto normativo, expansão coloquial, "
                         "concatenado (both) ou máximo dos dois scores (max)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="reranker")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    cache = json.loads(Path(args.cand).read_text(encoding="utf-8"))
    gold = _load_gold()

    t0 = time.time()
    try:
        ce = CrossEncoder(args.model, max_length=args.max_len, device="cuda",
                          trust_remote_code=True)
    except TypeError:
        ce = CrossEncoder(args.model, max_length=args.max_len, device="cuda")
    load_s = time.time() - t0

    reranked: Dict[str, List[Dict[str, Any]]] = {}
    fused_order: Dict[str, List[Dict[str, Any]]] = {}
    total_pairs = 0
    t0 = time.time()
    for qid, obj in cache.items():
        q = obj["question"]
        cands = obj["cands"]
        fused_order[qid] = cands  # ordem da fusão (baseline pré-rerank)
        if not cands:
            reranked[qid] = []
            continue
        if args.doc_field == "max":
            # Dois forwards por candidato: max(score_texto, score_expansão)
            p_txt = [[q, c["text"]] for c in cands]
            p_exp = [[q, c.get("expansion") or c["text"]] for c in cands]
            s_txt = ce.predict(p_txt, batch_size=32, show_progress_bar=False)
            s_exp = ce.predict(p_exp, batch_size=32, show_progress_bar=False)
            scores = [max(float(a), float(b)) for a, b in zip(s_txt, s_exp)]
            total_pairs += 2 * len(cands)
        else:
            if args.doc_field == "expansion":
                docs = [c.get("expansion") or c["text"] for c in cands]
            elif args.doc_field == "both":
                docs = [(c["text"] + " " + (c.get("expansion") or "")) for c in cands]
            else:
                docs = [c["text"] for c in cands]
            pairs = [[q, d] for d in docs]
            scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
            total_pairs += len(pairs)
        order = sorted(zip(cands, scores), key=lambda cs: -float(cs[1]))
        reranked[qid] = [dict(c, rerank_score=float(s)) for c, s in order]
    rerank_s = time.time() - t0
    ms_per_pair = rerank_s / max(total_pairs, 1) * 1000

    # Latência de rerank de 1 consulta (N forwards) — medida isolada
    sample = next((o for o in cache.values() if o["cands"]), None)
    q_ms = None
    if sample:
        mult = 2 if args.doc_field == "max" else 1
        pairs = [[sample["question"], c["text"]] for c in sample["cands"]] * mult
        ce.predict(pairs, batch_size=32)  # warmup
        t0 = time.time()
        for _ in range(10):
            ce.predict(pairs, batch_size=32)
        q_ms = (time.time() - t0) / 10 * 1000

    # Conjuntos de ids
    qa_ids = [qid for qid in cache if qid.startswith("qa-")]
    all_ids = list(cache.keys())

    m_fused = _metrics(fused_order, gold, qa_ids)
    m_rr = _metrics(reranked, gold, qa_ids)

    print("\n" + "=" * 84)
    print(f"  ETAPA 4 — RERANKER {args.label} — GATE holdout 151")
    print("=" * 84)
    print(f"| {'Configuração':<42} | {'R@1':>6} | {'R@2':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 44 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")
    print(f"| {'Fusão top-N (pré-rerank)':<42} | {m_fused['recall_at_1']:>5.1f}% | "
          f"{m_fused['recall_at_2']:>5.1f}% | {m_fused['recall_at_5']:>5.1f}% | {m_fused['mrr']:>7.4f} |")
    print(f"| {'Fusão + reranker CE':<42} | {m_rr['recall_at_1']:>5.1f}% | "
          f"{m_rr['recall_at_2']:>5.1f}% | {m_rr['recall_at_5']:>5.1f}% | {m_rr['mrr']:>7.4f} |")
    print("=" * 84)
    print(f"\nCusto rerank: {ms_per_pair:.1f} ms/par (GPU) | ~{q_ms:.0f} ms p/ consulta "
          f"(N={len(sample['cands']) if sample else 0} forwards, GPU) | load {load_s:.1f}s")

    out = {
        "label": args.label, "model": args.model,
        "fused_pre_rerank_151": m_fused, "reranked_151": m_rr,
        "cost": {"ms_per_pair_gpu": round(ms_per_pair, 2),
                 "ms_per_query_gpu": round(q_ms, 1) if q_ms else None,
                 "n_forwards": len(sample["cands"]) if sample else 0,
                 "load_s": round(load_s, 1)},
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
