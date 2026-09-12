#!/usr/bin/env python3
"""
eval_recall_v4.py — TRAVA 5 (a régua que decide): recall final nas 151 reais.

Nenhum ganho de acurácia do classificador conta se o R@2/R@5 final não subir. Este script
mede o recall DECISIVO plugando um classificador v4 (pkl) no pipeline completo da v3:
  s1 = BM25 gated-expandido (usa o classificador -> ÚNICO sinal afetado pela FRENTE A)
  sT = denso EmbeddingGemma-300M texto+expansão (cache de rankings; independe do clf)
  sE = denso EmbeddingGemma-300M expansão-only  (cache de rankings; independe do clf)
  fusão = RRF(k=30, pesos iguais) -> top-5   (config vencedora da v3)

Reusa o harness honesto (eval_common: holdout, check_hit, app_fts_query) e a fusão v3
(rrf.rrf_fuse). Caminho de busca idêntico ao app. O holdout NUNCA calibra nada.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python eval_recall_v4.py --clf models/v4_v1free.pkl \
      --db ../retrieval3/indices/index_hf_36nr_exp.db \
      --dense-text ../retrieval3/results/bin_text_rank.json \
      --dense-exp  ../retrieval3/results/bin_exponly_rank.json --label v1+free

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from bm25 import HybridBm25  # noqa: E402
from eval_common import evaluate, footer, header, load_holdout  # noqa: E402
from rrf import rrf_fuse  # noqa: E402

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", required=True, help="pipeline sklearn do classificador (pkl)")
    ap.add_argument("--db", default=str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db"))
    ap.add_argument("--dense-text", default=str(_ROOT / "retrieval3" / "results" / "bin_text_rank.json"))
    ap.add_argument("--dense-exp", default=str(_ROOT / "retrieval3" / "results" / "bin_exponly_rank.json"))
    ap.add_argument("--rrf-k", type=float, default=30.0)
    ap.add_argument("--topk", type=int, default=2, help="top-k de corte do boost/gate (info)")
    ap.add_argument("--label", default="v4")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    holdout = load_holdout("qa_v2")
    dtext = json.loads(Path(args.dense_text).read_text(encoding="utf-8"))
    dexp = json.loads(Path(args.dense_exp).read_text(encoding="utf-8"))
    hy = HybridBm25(args.db, args.clf, weights=EXP_W)

    def bm25_only(it):
        return hy.rank(it.question, 5)

    def fusion(it):
        s1 = hy.rank(it.question, 60)
        sT = dtext.get(it.id, [])
        sE = dexp.get(it.id, [])
        return rrf_fuse([s1, sT, sE], [1.0, 1.0, 1.0], k=args.rrf_k, limit=5)

    header(f"RÉGUA DECISIVA — recall final 151 — clf={args.label}")
    m_bm = evaluate(holdout, bm25_only)
    print(m_bm.line("BM25 gated-exp (só clf, s1)"))
    m_fu = evaluate(holdout, fusion)
    print(m_fu.line(f"Fusão 3-sinais RRF k={args.rrf_k:.0f}"))
    footer()

    if args.json_out:
        out = {"label": args.label, "clf": args.clf, "rrf_k": args.rrf_k,
               "bm25_gated_exp": m_bm.as_dict(), "fusion_3sig": m_fu.as_dict()}
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Salvo em {args.json_out}")


if __name__ == "__main__":
    main()
