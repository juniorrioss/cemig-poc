#!/usr/bin/env python3
"""
eval_asr_recall.py — TRAVA 4 (decisiva sob ASR): recall nas 151 sobre a TRANSCRIÇÃO real.

Em vez do texto limpo, usa a transcrição do Whisper Base Q5_1 (engine do app) das 151
faladas por TTS (asr_transcriptions.jsonl). Mede o recall final (fusão 3-sinais) com o
índice/dense v3 vs v4, para ver se a expansão v4 (que inclui variantes de ASR) é MAIS
ROBUSTA ao ruído de transcrição — o teste mais próximo do campo.

IMPORTANTE: os caches densos são indexados por it.id do holdout; como a consulta muda
(transcrição), o sinal DENSO precisaria ser reencodado sobre a transcrição. Para manter
honesto e barato, aqui medimos o efeito no sinal BM25 (que reencoda a query a cada busca)
e, para a fusão, reusamos o denso do texto-limpo como PISO (subestima o ganho do denso).
Reportamos as duas leituras.

Uso (classifier/.venv): ../classifier/.venv/bin/python eval_asr_recall.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import HybridBm25  # noqa: E402
from eval_common import QAItem, check_hit, load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
V3_DB = str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db")
V4_DB = str(_HERE / "indices" / "index_hf_36nr_expv4.db")
ASR = _HERE / "data" / "asr_transcriptions.jsonl"


def load_asr() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for line in ASR.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        m[r["id"]] = r["transcription"]
    return m


def build_asr_items(clean: List[QAItem], asr: Dict[str, str]) -> List[QAItem]:
    """Cria QAItems com a pergunta = transcrição do ASR (mantém o gold do item limpo)."""
    out = []
    for it in clean:
        t = asr.get(it.id, "").strip()
        if not t:
            continue
        out.append(QAItem(id=it.id, question=t, doc=it.doc, section=it.section,
                          chunk_id=it.chunk_id, relevant_chunk_ids=it.relevant_chunk_ids,
                          query_terms=it.query_terms, source="asr"))
    return out


def recall(items: List[QAItem], rank_fn) -> Dict[str, float]:
    n = len(items)
    h1 = h2 = h5 = 0
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
            if rank <= 1:
                h1 += 1
            if rank <= 2:
                h2 += 1
            if rank <= 5:
                h5 += 1
            mrr += 1.0 / rank
    return {"n": n, "recall_at_1": round(100 * h1 / n, 1), "recall_at_2": round(100 * h2 / n, 1),
            "recall_at_5": round(100 * h5 / n, 1), "mrr": round(mrr / n, 4)}


def main() -> None:
    clean = load_holdout("qa_v2")
    asr = load_asr()
    items = build_asr_items(clean, asr)
    print(f"ASR itens: {len(items)}/{len(clean)}")

    dv3t = json.loads((_ROOT / "retrieval3" / "results" / "bin_text_rank.json").read_text())
    dv3e = json.loads((_ROOT / "retrieval3" / "results" / "bin_exponly_rank.json").read_text())
    dv4e = json.loads((_HERE / "results" / "densev4_exp_rank.json").read_text())

    hy3 = HybridBm25(V3_DB, CLF, weights=EXP_W)
    hy4 = HybridBm25(V4_DB, CLF, weights=EXP_W)

    print("=" * 78)
    print("  TRAVA 4 — RECALL SOBRE TRANSCRIÇÃO ASR (Whisper Base) — 151 faladas")
    print("=" * 78)
    print(f"| {'config':<40} | {'R@2':>5} | {'R@5':>5} | {'MRR':>6} |")
    print("|" + "-" * 42 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 8 + "|")

    out = {}
    # BM25-só (query reencodada -> mede robustez direta do índice a ASR)
    for label, hy in [("BM25 gated v3 (limpo->indice v3)", hy3),
                      ("BM25 gated v4 (limpo->indice v4)", hy4)]:
        m = recall(items, lambda it, hy=hy: hy.rank(it.question, 5))
        out[label] = m
        print(f"| {label:<40} | {m['recall_at_2']:>4.1f}% | {m['recall_at_5']:>4.1f}% | {m['mrr']:>6.4f} |")

    # Fusão (denso reusa ranking do texto-limpo = PISO; subestima ganho denso)
    def fus(it, hy, dt, de):
        return rrf_fuse([hy.rank(it.question, 60), dt.get(it.id, []), de.get(it.id, [])],
                        [1.0, 1.0, 1.0], k=30.0, limit=5)
    for label, hy, dt, de in [
        ("Fusão v3 (denso=piso limpo)", hy3, dv3t, dv3e),
        ("Fusão v4bm+v3text+v4exp (denso=piso)", hy4, dv3t, dv4e)]:
        m = recall(items, lambda it, hy=hy, dt=dt, de=de: fus(it, hy, dt, de))
        out[label] = m
        print(f"| {label:<40} | {m['recall_at_2']:>4.1f}% | {m['recall_at_5']:>4.1f}% | {m['mrr']:>6.4f} |")
    print("=" * 78)
    print("Nota: denso é PISO (ranking do texto-limpo); o ganho real do denso sob ASR é maior.")
    Path(_HERE / "results" / "asr_recall.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Salvo em results/asr_recall.json")


if __name__ == "__main__":
    main()
