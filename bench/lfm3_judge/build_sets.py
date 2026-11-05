#!/usr/bin/env python3
"""
build_sets.py — monta os 3 conjuntos de teste com verdade conhecida (schema único de item).

  V  variantes sintéticas (variants_j.py) sobre o benchmark_mobile (gold + evidência-ouro);
  R  respostas REAIS do f4-r32-ep2 na F6 (recorte344, células op e ctrl) com o veredito
     Qwen27b da régua honesta (approved/hallucination/refused) já salvo no parquet;
  B  benchmark_mobile_results @faeb942: respostas do Energy-Mobile com os contextos dos 4
     retrievers do mobilerag (naive/bm25/mobilerag/mobilerag_plus), com veredito do
     gpt-oss-120b (árbitro independente) e rubrica binária do Qwen; + o NOSSO retriever
     (ponto de operação F6) como 5º, só no nível de CONTEXTO (não há resposta do
     Energy-Mobile para ele — o leitor LiteRT não roda aqui; ver RESULTS.md).

Orçamento de contexto do 5º retriever: os do mobilerag foram cortados a 1.664 tokens do
Energy-Mobile (system+pergunta+docs; média 3.460–5.011 chars). Para o nosso: top-5 na ordem
do ranking, documentos inteiros enquanto couberem em BUDGET_CHARS (o 1º é cortado se
sozinho exceder) — mesma regra "descarta os de menor ranking primeiro".

Uso: classifier/.venv/bin/python bench/lfm3_judge/build_sets.py [--nv 150] [--nb 300]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_j as C  # noqa: E402
from variants_j import build_variants  # noqa: E402

BUDGET_CHARS = 5000  # ≈ mediana dos 4 contextos deles (4.032–5.065 chars)


def load_bench_rows():
    gab = {g["base_qa_id"]: g["facts"] for g in C.read_jsonl(C.GAB384)}
    cols = ["id", "base_qa_id", "question", "answer", "question_type", "evidence",
            "vocabulary_level", "source_file"]
    rows = pq.read_table(str(C.BENCH_PARQUET), columns=cols).to_pylist()
    for r in rows:
        r["facts"] = gab.get(r["base_qa_id"], [])
    return rows


def set_v(rows, nv: int, seed: int):
    para = json.loads((C.DATA / "eval_vela_paraphrase_cache.json").read_text(encoding="utf-8"))
    return build_variants(rows, nv, seed, para)


def set_r(rows):
    by_id = {r["id"]: r for r in rows}
    out = []
    for cell in ("op", "ctrl"):
        p = C.F6_EVAL / f"lfm3-f4-r32-ep2-f6-recorte344-{cell}-d50.parquet"
        for e in pq.read_table(str(p)).to_pylist():
            b = by_id[e["id"]]
            out.append({
                "uid": f"R:{cell}:{e['id']}", "set": "R", "qid": e["id"], "base_qa_id": e["base_qa_id"],
                "cell": cell, "question": e["question"], "gold": e["answer_gold"],
                "evidence": e["evidence_text"], "context": e["context_given"], "answer": e["answer_model"],
                "facts": b["facts"], "vocabulary_level": e["vocabulary_level"],
                "question_type": e["question_type"], "source_file": e["source_file"],
                "gold_kind": e["gold_kind"], "ctx_has_gold": e["ctx_has_gold"],
                "qwen": {"approved": e["approved"], "hallucination": e["hallucination"],
                         "refused": e["refused"], "coverage": e["coverage"],
                         "tautology": e["tautology"], "note": e["judge_note"]},
            })
    return out


def _our_contexts():
    """Contexto do NOSSO retriever (F6 op) por id do benchmark, com orçamento de chars."""
    con = sqlite3.connect(str(C.V4_DB))
    chunks = {i: (d, s, t) for i, d, s, t in con.execute("SELECT id, doc, section, text FROM chunks")}
    con.close()
    con = sqlite3.connect(str(C.CACHE / "manuals.db"))
    chunks.update({i: (d, f, t) for i, d, f, t in con.execute("SELECT id, doc, file, text FROM mchunks")})
    con.close()
    out = {}
    for r in C.read_jsonl(C.CACHE / f"rank_{C.OUR_VARIANT}.jsonl"):
        parts, used = [], 0
        for pos, cid in enumerate(r["ids"][:5], 1):
            doc, sec, text = chunks[cid]
            block = f"[{pos}] Arquivo: {doc} {sec} | chunk: {cid}\n{text}"
            if used + len(block) > BUDGET_CHARS:
                if not parts:
                    parts.append(block[:BUDGET_CHARS])
                break
            parts.append(block)
            used += len(block) + 2
        out[r["id"]] = {"context": "\n\n".join(parts), "hit5": r["hit5"], "hit2": r["hit2"],
                        "kind": r["kind"], "ids": r["ids"][:5]}
    return out


def set_b(rows, nb: int, seed: int):
    by_id = {r["id"]: r for r in rows}
    bmr = pq.read_table(str(C.BMR_PARQUET)).to_pylist()
    rng = random.Random(seed)
    # amostra por base_qa_id (as 4 formulações juntas), como o eval-vela fazia por pergunta
    groups = sorted({r["base_qa_id"] for r in bmr})
    chosen = set(rng.sample(groups, min(nb // 4 if nb >= 4 else nb, len(groups))))
    ours = _our_contexts()
    out = []
    for r in bmr:
        if r["base_qa_id"] not in chosen:
            continue
        b = by_id[r["id"]]
        evidence = "\n\n".join(e["text"] for e in b["evidence"] if e.get("text"))
        base = {"qid": r["id"], "base_qa_id": r["base_qa_id"], "question": r["question"],
                "gold": r["answer"], "evidence": evidence, "facts": b["facts"],
                "vocabulary_level": r["vocabulary_level"], "question_type": r["question_type"],
                "source_file": r["source_file"], "rubric": list(r["rubric"] or [])}
        for m in C.METHODS:
            out.append(dict(base, uid=f"B:{m}:{r['id']}", set="B", method=m,
                            context=r[f"{m}_context"], answer=r[f"{m}_answer"],
                            ev_cov=round(C.evidence_coverage(evidence, r[f"{m}_context"]), 4),
                            orig={"gptoss": r[f"{m}_judge"], "gptoss_score": r[f"{m}_judge_score"],
                                  "gptoss_reason": r[f"{m}_judge_reason"],
                                  "qwen_rubric": r[f"{m}_rubric_is_correct"],
                                  "qwen_rubric_reason": r[f"{m}_rubric_justificativa"],
                                  "evidence_recall": r[f"{m}_evidence_recall"], "hit": r[f"{m}_hit"]}))
        o = ours[r["id"]]
        out.append(dict(base, uid=f"B:ours:{r['id']}", set="B", method="ours", context=o["context"],
                        answer=None, ev_cov=round(C.evidence_coverage(evidence, o["context"]), 4),
                        orig={"hit5": o["hit5"], "hit2": o["hit2"], "gold_kind": o["kind"], "ids": o["ids"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv", type=int, default=150, help="perguntas para as variantes")
    ap.add_argument("--nb", type=int, default=300, help="perguntas do benchmark (múltiplo de 4)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = load_bench_rows()
    V = set_v(rows, args.nv, args.seed)
    R = set_r(rows)
    B = set_b(rows, args.nb, args.seed)
    for name, s in (("V", V), ("R", R), ("B", B)):
        C.write_jsonl(C.DATA / "sets" / f"set_{name}.jsonl", s)
    print("V", len(V), Counter(x["variant"] for x in V))
    print("R", len(R), Counter((x["cell"], x["qwen"]["approved"]) for x in R))
    print("B", len(B), Counter(x["method"] for x in B))
    import statistics as st
    for m in list(C.METHODS) + ["ours"]:
        xs = [x for x in B if x["method"] == m]
        print(f"  {m:15s} ctx chars med={st.median(len(x['context']) for x in xs):.0f} "
              f"ev_cov={st.mean(x['ev_cov'] for x in xs):.3f}")


if __name__ == "__main__":
    main()
