#!/usr/bin/env python3
"""
retrieve_f3.py — item 1b do brief: roda o retriever REAL da POC (v4 embarcado:
BM25 gated-expandido + denso texto + denso exp-only, RRF k=10 w=2,1,2) sobre o
corpus COMPLETO (corpus 36 NRs, índice v4 do app) para CADA pergunta do pool
de treino da F3, e salva o top-8 com scores + rótulo por passagem.

Rótulo por sobreposição com a ref ouro do QA (critério registrado no README):
  - cada QA rastreia unit_id -> unidade de conhecimento com `fato` literal da
    norma (funil s1 da F2/F3);
  - containment = |tokens(fato) ∩ tokens(chunk.text)| / |tokens(fato)|
    (norm_tokens: minúsculas sem acento, alfanumérico);
  - label = "gold"    se chunk.doc == nr-do-QA  e containment >= 0.70
            "partial" se chunk.doc == nr-do-QA  e 0.30 <= containment < 0.70
            "distractor" caso contrário.

Reuso: RetrieverV4 de bench/ctx_topk/retrieval_v4.py (espelho provado do
HybridRetriever.searchV3 embarcado, paridade R@2 51.7%). Encoder denso =
llama-server GGUF embedding em :8399 (mesmo encoder on-device).

Saída: data/retrieval/topk_{batch}.jsonl
  {qa_id, nr, unit_id, ref, question, gold_containment_max,
   topk: [{id, doc, section, title, rank, rrf_score, label, containment}]}

Uso (classifier/.venv; llama-server :8399 no ar):
    ../../classifier/.venv/bin/python retrieve_f3.py [--limit 50] [--batch-tag f2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "ctx_topk"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))

from common_synth import norm_tokens, read_jsonl, write_jsonl  # noqa: E402

F2_SYNTH = REPO / "bench" / "lfm3_f2" / "data" / "synth"
F3_SYNTH = HERE / "data" / "synth"
OUT_DIR = HERE / "data" / "retrieval"

F2_NRS = [10, 18, 34]
TOP_K = 8


def load_pool(which: str) -> tuple[list[dict], dict[str, dict]]:
    """QAs aprovados (F2 e/ou F3) + unidades p/ rotulagem."""
    qas: list[dict] = []
    units: dict[str, dict] = {}
    dirs = []
    if which in ("all", "f2"):
        dirs.append((F2_SYNTH, F2_NRS))
    if which in ("all", "f3"):
        import re
        f3_nrs = sorted({int(re.search(r"nr(\d+)", p.name).group(1))
                         for p in F3_SYNTH.glob("qa_passed_nr*.jsonl")})
        dirs.append((F3_SYNTH, f3_nrs))
    for base, nrs in dirs:
        for nr in nrs:
            qas.extend(read_jsonl(base / f"qa_passed_nr{nr}.jsonl"))
            for u in read_jsonl(base / f"units_nr{nr}.jsonl"):
                units[u["unit_id"]] = u
    return qas, units


def containment(fato_tokens: set[str], chunk_text: str) -> float:
    if not fato_tokens:
        return 0.0
    ct = set(norm_tokens(chunk_text))
    return len(fato_tokens & ct) / len(fato_tokens)


def label_chunk(chunk: dict, qa_nr: int, cont: float) -> str:
    same_doc = chunk["doc"].lower() == f"nr-{qa_nr:02d}" or chunk["doc"].lower() == f"nr-{qa_nr}"
    if same_doc and cont >= 0.70:
        return "gold"
    if same_doc and cont >= 0.30:
        return "partial"
    return "distractor"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", choices=["all", "f2", "f3"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--emb-url", default="http://127.0.0.1:8399")
    args = ap.parse_args()

    qas, units = load_pool(args.pool)
    if args.limit:
        qas = qas[: args.limit]
    out_path = Path(args.out) if args.out else OUT_DIR / f"topk_{args.pool}.jsonl"
    done = {r["qa_id"] for r in read_jsonl(out_path)}
    todo = [q for q in qas if q["qa_id"] not in done]
    print(f"[retrieve] pool={args.pool} n={len(qas)} pendentes={len(todo)}")
    if not todo:
        return

    from retrieval_v4 import RetrieverV4  # noqa: PLC0415 (encoda tudo no init)
    questions = {q["qa_id"]: q["question"] for q in todo}
    rtv = RetrieverV4(questions, emb_url=args.emb_url)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_gold_hit = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, qa in enumerate(todo):
            u = units[qa["unit_id"]]
            ft = set(norm_tokens(u["fato"]))
            res = rtv.search(qa["qa_id"], qa["question"], top_k=TOP_K)
            topk = []
            any_gold = False
            for rank, c in enumerate(res["chunks"]):
                cont = containment(ft, c["text"])
                lab = label_chunk(c, qa["nr"], cont)
                any_gold = any_gold or lab == "gold"
                topk.append({"id": c["id"], "doc": c["doc"], "section": c["section"],
                             "title": c["title"], "rank": rank,
                             "rrf_score": round(float(c.get("rrf", c.get("score", 0.0))), 6),
                             "label": lab, "containment": round(cont, 4)})
            n_gold_hit += int(any_gold)
            rec = {"qa_id": qa["qa_id"], "nr": qa["nr"], "unit_id": qa["unit_id"],
                   "ref": qa["ref"], "question": qa["question"],
                   "gate": res["decision"]["mode"],
                   "gold_in_top8": any_gold, "topk": topk}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(todo)} (gold@8 {100*n_gold_hit/(i+1):.1f}%)",
                      flush=True)
    print(f"[ok] {out_path.name}: {len(todo)} novas, gold@8 "
          f"{100*n_gold_hit/max(1,len(todo)):.1f}%")


if __name__ == "__main__":
    main()
