#!/usr/bin/env python3
"""
consolidate.py — Etapa 5: tabela final etapa-a-etapa (R@1/2/5 + MRR) + custo mobile.

Recomputa, no MESMO harness honesto (caminho do app), todas as etapas do Retrieval v3
sobre o holdout 151 (métrica oficial) e 171 (all), e emite a tabela de consolidação e
o JSON com o pipeline vencedor. Custos mobile vêm dos metadados de build/rerank.

Uso (classifier/.venv):
    ../classifier/.venv/bin/python consolidate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from bm25 import Bm25Retriever, HybridBm25
from eval_common import APP_W, app_fts_query, evaluate, footer, header, load_holdout
from rrf import rrf_fuse

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
BASE_DB = str(_ROOT / "corpus" / "index_hf_36nr.db")
EXP_DB = str(_HERE / "indices" / "index_hf_36nr_exp.db")
CLF = str(_ROOT / "classifier" / "models" / "classic_winner.pkl")
EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def _load(name):
    p = _HERE / "results" / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    dtext = _load("dense_rank_gemma768_exp.json")
    dexp = _load("dense_rank_gemma768_exponly.json")

    rows = []  # (etapa, label, metric_dict, mobile_note)

    for subset in ["qa_v2", "all"]:
        ho = load_holdout(subset)
        n = len(ho)

        # Etapa 0: main atual
        bm_base = Bm25Retriever(BASE_DB)
        hy_base = HybridBm25(BASE_DB, CLF)
        m0a = evaluate(ho, lambda it: bm_base.search_raw(it.question, 5))
        m0b = evaluate(ho, lambda it: hy_base.rank(it.question, 5))

        # Etapa 1: + expansão de documento (BM25 gated no índice expandido)
        hy_exp = HybridBm25(EXP_DB, CLF, weights=EXP_W)
        m1 = evaluate(ho, lambda it: hy_exp.rank(it.question, 5))

        # Etapa 3: + denso EmbeddingGemma (fusão 2 e 3 sinais)
        def f_s1sT(it):
            return rrf_fuse([hy_exp.rank(it.question, 60), dtext.get(it.id, [])], [1, 1], k=60, limit=5)

        def f_3way(it):
            return rrf_fuse([hy_exp.rank(it.question, 60), dtext.get(it.id, []), dexp.get(it.id, [])],
                            [1, 1, 1], k=30, limit=5)
        m3a = evaluate(ho, f_s1sT)
        m3b = evaluate(ho, f_3way)

        header(f"RETRIEVAL v3 — CONSOLIDAÇÃO ({subset}, {n} perguntas)")
        print(m0a.line("E0 main: BM25 fala bruta"))
        print(m0b.line("E0 main: híbrido classificador+BM25"))
        print(m1.line("E1: + expansão de documento (BM25 gated)"))
        print(m3a.line("E3: E1 + denso Gemma texto (RRF)"))
        print(m3b.line("E3: E1 + denso Gemma texto+exp-only (RRF 3-sinais) *VENCEDOR MOBILE*"))
        footer()

        if subset == "qa_v2":
            rows = {
                "E0_baseline_raw": m0a.as_dict(),
                "E0_hybrid_main": m0b.as_dict(),
                "E1_expansion_gated": m1.as_dict(),
                "E3_fusion_2sig": m3a.as_dict(),
                "E3_fusion_3sig_WINNER": m3b.as_dict(),
            }

        bm_base.close(); hy_base.bm.close(); hy_exp.bm.close()

    # Reranker (medições prévias, só 151)
    rr_27b = _load("rerank_llm_3way.json")
    rr_lfm = _load("rerank_lfm_3way.json")

    dense_meta = _load("dense_gemma768_exp_meta.json")
    dexp_meta = _load("dense_gemma768_exponly_meta.json")

    out = {
        "holdout_151": rows,
        "reranker_151": {
            "cloud_27B_listwise_top10": rr_27b.get("reranked_151", {}),
            "mobile_LFM2.5_listwise_top10": rr_lfm.get("reranked_151", {}),
        },
        "mobile_cost": {
            "bm25_expanded_index_mb": round(Path(EXP_DB).stat().st_size / 1e6, 2),
            "dense_model": dense_meta.get("model"),
            "dense_dim": dense_meta.get("dim"),
            "dense_index_text_mb": dense_meta.get("index_mb"),
            "dense_index_exponly_mb": dexp_meta.get("index_mb"),
            "encode_query_ms_gpu_5070": dense_meta.get("encode_query_ms_gpu"),
        },
        "winner": {
            "pipeline": "BM25-gated-expanded + EmbeddingGemma-300M dense(text+exp-only) via RRF k=30",
            "r2_151": rows["E3_fusion_3sig_WINNER"]["recall_at_2"],
            "r5_151": rows["E3_fusion_3sig_WINNER"]["recall_at_5"],
            "meta_r2_55_mobile_only": rows["E3_fusion_3sig_WINNER"]["recall_at_2"] >= 55.0,
            "meta_r2_55_with_cloud_27B_rerank": rr_27b.get("reranked_151", {}).get("recall_at_2", 0) >= 55.0,
        },
    }
    (_HERE / "results" / "consolidation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nConsolidação salva em results/consolidation.json")
    print(f"\nVENCEDOR MOBILE: R@2={out['winner']['r2_151']:.1f}% R@5={out['winner']['r5_151']:.1f}% (151)")
    print(f"Meta 55% R@2 mobile-only: {'ATINGIDA' if out['winner']['meta_r2_55_mobile_only'] else 'NÃO'} "
          f"| com reranker 27B (cloud): {'ATINGIDA' if out['winner']['meta_r2_55_with_cloud_27B_rerank'] else 'NÃO'} "
          f"({rr_27b.get('reranked_151', {}).get('recall_at_2', 0):.1f}%)")


if __name__ == "__main__":
    main()
