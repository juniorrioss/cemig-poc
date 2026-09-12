#!/usr/bin/env python3
"""
eval_asr_fusion_true.py — TRAVA 4 (fiel): fusão completa reencodando a QUERY na transcrição ASR.

Diferente de eval_asr_recall (denso=piso limpo), aqui reencodamos o vetor denso da CONSULTA
a partir da TRANSCRIÇÃO do Whisper (não do texto limpo), casando contra os índices densos
de documento v3-text e v4-exp já reencodados. É o teste de campo mais fiel: fala -> ASR ->
retrieval completo (BM25 + 2 densos) sobre a transcrição.

Requer o servidor GGUF de embedding no ar (:8399). Compara v3 vs combo v4 vencedor.

Uso (classifier/.venv): ../classifier/.venv/bin/python eval_asr_fusion_true.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import HybridBm25  # noqa: E402
from eval_common import QAItem, check_hit, load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from dense_v4 import (DOC_PROMPT, QUERY_PROMPT, embed_batch, load_chunks,  # noqa: E402
                      load_exp)

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
V3_DB = str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db")
V4_DB = str(_HERE / "indices" / "index_hf_36nr_expv4.db")
URL = "http://127.0.0.1:8399"


def load_asr() -> Dict[str, str]:
    m = {}
    for line in (_HERE / "data" / "asr_transcriptions.jsonl").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        m[r["id"]] = r["transcription"]
    return m


def doc_embeddings():
    """Reencoda documentos: v3-text, v4-text, v3-exp, v4-exp (uma vez)."""
    e3 = load_exp(_ROOT / "retrieval3" / "data" / "expansions.jsonl", "v3")
    e4 = load_exp(_HERE / "data" / "expansions_v4.jsonl", "v4")
    rows = load_chunks(str(_ROOT / "corpus" / "index_hf_36nr.db"))
    v3_text, v4_exp = [], []
    for r in rows:
        ev3, ev4 = e3.get(r["id"], ""), e4.get(r["id"], "")
        v3_text.append(r["text"] + " " + ev3 if ev3 else r["text"])
        v4_exp.append(ev4 or r["text"])

    def enc(texts):
        parts = []
        for i in range(0, len(texts), 32):
            parts.append(embed_batch(URL, [DOC_PROMPT + t for t in texts[i:i + 32]]))
        return np.vstack(parts)
    return rows, {"v3_text": enc(v3_text), "v4_exp": enc(v4_exp)}


def rank_from_sims(sims, rows, ids, pool=60):
    top = np.argsort(-sims)[:pool]
    return [{"id": int(ids[j]), "doc": rows[j]["doc"], "section": rows[j]["section"],
             "title": rows[j]["title"], "text": rows[j]["text"], "score": float(sims[j])}
            for j in top]


def recall(items, rank_fn):
    n = len(items)
    h2 = h5 = 0
    mrr = 0.0
    for it in items:
        res = rank_fn(it)
        gold = it.gold_pair()
        rank = None
        for idx, r in enumerate(res[:5]):
            if check_hit(r, gold):
                rank = idx + 1
                break
        if rank:
            if rank <= 2:
                h2 += 1
            if rank <= 5:
                h5 += 1
            mrr += 1.0 / rank
    return {"n": n, "recall_at_2": round(100 * h2 / n, 1),
            "recall_at_5": round(100 * h5 / n, 1), "mrr": round(mrr / n, 4)}


def main() -> None:
    clean = load_holdout("qa_v2")
    asr = load_asr()
    rows, demb = doc_embeddings()
    ids = np.array([r["id"] for r in rows])

    # Query embeddings sobre a TRANSCRIÇÃO (fiel) e sobre o texto LIMPO (referência)
    order = [it for it in clean if asr.get(it.id, "").strip()]
    q_asr = embed_batch(URL, [QUERY_PROMPT + asr[it.id] for it in order])
    q_clean = embed_batch(URL, [QUERY_PROMPT + it.question for it in order])

    hy3 = HybridBm25(V3_DB, CLF, weights=EXP_W)
    hy4 = HybridBm25(V4_DB, CLF, weights=EXP_W)

    def make_items(use_asr: bool):
        out = []
        for it in order:
            q = asr[it.id] if use_asr else it.question
            out.append(QAItem(id=it.id, question=q, doc=it.doc, section=it.section,
                              chunk_id=it.chunk_id, relevant_chunk_ids=it.relevant_chunk_ids,
                              query_terms=it.query_terms, source="asr" if use_asr else "clean"))
        return out

    def dense_rank(qemb, key):
        idx = {it.id: i for i, it in enumerate(order)}
        return lambda it: rank_from_sims(qemb[idx[it.id]] @ demb[key].T, rows, ids)

    print("=" * 84)
    print("  TRAVA 4 FIEL — fala->ASR->retrieval completo (query denso reencodada na transcrição)")
    print("=" * 84)
    print(f"| {'config':<46} | {'R@2':>5} | {'R@5':>5} | {'MRR':>6} |")
    print("|" + "-" * 48 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 8 + "|")

    out = {}
    scenarios = [
        ("v3 LIMPO (k30 iguais)", False, hy3, q_clean, "v3_text", None, 30.0, (1, 1, 1)),
        ("v3 ASR   (k30 iguais)", True, hy3, q_asr, "v3_text", None, 30.0, (1, 1, 1)),
        ("v4 LIMPO (k10 w2,1,2)", False, hy4, q_clean, "v3_text", "v4_exp", 10.0, (2, 1, 2)),
        ("v4 ASR   (k10 w2,1,2)", True, hy4, q_asr, "v3_text", "v4_exp", 10.0, (2, 1, 2)),
    ]
    # v3 usa dense v3_text + (exp v3 do cache limpo não reencodado) -> p/ honestidade,
    # v3 aqui usa só 2 sinais reencodados (bm + v3_text) para comparar em pé de igualdade.
    for name, use_asr, hy, qemb, kt, ke, k, wv in scenarios:
        items = make_items(use_asr)
        idx = {it.id: i for i, it in enumerate(order)}
        dt = dense_rank(qemb, kt)
        de = dense_rank(qemb, ke) if ke else None

        def rf(it, hy=hy, dt=dt, de=de, k=k, wv=wv):
            sig = [hy.rank(it.question, 60), dt(it)]
            w = [wv[0], wv[1]]
            if de is not None:
                sig.append(de(it))
                w.append(wv[2])
            return rrf_fuse(sig, w, k=k, limit=5)
        m = recall(items, rf)
        out[name] = m
        print(f"| {name:<46} | {m['recall_at_2']:>4.1f}% | {m['recall_at_5']:>4.1f}% | {m['mrr']:>6.4f} |")
    print("=" * 84)
    Path(_HERE / "results" / "asr_fusion_true.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Salvo em results/asr_fusion_true.json")


if __name__ == "__main__":
    main()
