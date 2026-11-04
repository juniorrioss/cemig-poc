#!/usr/bin/env python3
"""
export_parquet_f6.py — parquets canônicos da F6 (schema estendido da F2/F4).

Um parquet por checkpoint × suíte × célula: data/eval/lfm3-f4-r32-ep2-f6-<suite>-<cell>.parquet.
context_type ganha os valores da F6:
  retrieved_top2_raw                    top-2 cru do retriever (app atual)
  retrieved_top5_compressed             top-5 -> compressão 50%, só normas no contexto
  retrieved_top5_compressed_manuals     idem, com >=1 chunk de MANUAL entre os 5 candidatos
context_given = texto EXATO do turno tool; system_prompt = system BAKED do F4 (registrado).

Uso: classifier/.venv/bin/python bench/lfm3_f6/export_parquet_f6.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402
import common_f5 as C  # noqa: E402
import tool_schema_f4 as T  # noqa: E402

CKPT = "f4-r32-ep2"
GEN_PARAMS = json.dumps({"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1,
                         "max_new_tokens": 768, "dtype": "bfloat16", "batch": 8,
                         "chat_template": "official", "thinking": "off",
                         "compression": {"ratio": 0.5, "budget_ref": "v4app top2 raw",
                                         "scorer": "hybrid lam0.3 prior0.05"}})


def main() -> None:
    out = F6.DATA / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sysmsg = T.baked_system_content()
    sha = hashlib.sha256(sysmsg.encode()).hexdigest()[:16]
    ref = "f4_baked_tools_v1"
    (out / "system_prompts.json").write_text(json.dumps(
        {ref: {"sha256_16": sha, "text": sysmsg}}, ensure_ascii=False, indent=2), encoding="utf-8")
    bench = {r["id"]: r for r in C.load_bench()}
    qa = {it["id"]: it for it in C.load_151()}
    for suite, tag in (("recorte344", "f6-recorte"), ("poc151", "f6-151")):
        prompts = {p["key"]: p for p in C.read_jsonl(F6.DATA / "e2e" /
                   f"prompts_{'recorte' if suite == 'recorte344' else '151'}.jsonl")}
        gens = {g["key"]: g for g in C.read_jsonl(F6.DATA / "e2e" / f"gen_{tag}.jsonl")}
        scored = C.read_jsonl(F6.DATA / "e2e" / f"scored_{tag}.jsonl")
        by_cell = {}
        for s in scored:
            p, g = prompts[s["key"]], gens[s["key"]]
            if suite == "recorte344":
                b = bench[s["id"]]
                bm = {"base_qa_id": b["base_qa_id"], "question": b["question"],
                      "answer_gold": b["answer"], "question_type": b["question_type"],
                      "source_id": b["source_id"], "vocabulary_level": b["vocabulary_level"],
                      "source_file": b["source_file"], "gold_kind": b["gold_kind"],
                      "evidence_text": "\n\n".join(e["text"] for e in b["evidence"])}
            else:
                it = qa[s["id"]]
                bm = {"base_qa_id": None, "question": it["question"],
                      "answer_gold": it.get("golden_answer"), "question_type": None,
                      "source_id": f"{it['doc']}#{it['section']}", "vocabulary_level": "coloquial",
                      "source_file": it["doc"], "gold_kind": "norma", "evidence_text": None}
            by_cell.setdefault(s["cell"], []).append({
                "id": s["id"], **bm,
                "answer_model": s["answer"], "raw_output": g.get("raw", ""), "thinking": "",
                "n_gen_tokens": s["n_gen_tokens"], "stopped_eos": s["stopped_eos"],
                "coverage": s["coverage"], "n_facts": s["n_facts"], "tautology": s["tautology"],
                "hallucination": s["hallucination"], "refused": s["refused"],
                "truncated": s["truncated"], "refusal_correct": None, "refusal_undue": None,
                "citation_present": s["citation_present"], "cite_gold_item": s["cite_item"],
                "approved": s["approved"], "answered_correct_anyway": None,
                "judge_note": s["judge_note"],
                "context_given": p["context_given"], "context_type": p["context_type"],
                "ctx_ids": json.dumps(p["ctx_ids"]), "n_manual_ctx": p["n_manual_ctx"],
                "ctx_ntok": p["ctx_ntok"], "n_prompt_tokens": s["n_prompt_tokens"],
                "ctx_has_gold": s["ctx_has_gold_span"], "retrieval_variant": p["variant"],
                "system_prompt_sha": sha, "system_prompt_ref": ref,
                "checkpoint": CKPT, "cell": f"f6-{suite}-{s['cell']}", "gen_params": GEN_PARAMS,
            })
        for cell, recs in by_cell.items():
            path = out / f"lfm3-{CKPT}-f6-{suite}-{cell.replace('_', '-')}.parquet"
            pq.write_table(pa.Table.from_pylist(recs), path, compression="zstd")
            print(f"[ok] {path.name}: {len(recs)} linhas")


if __name__ == "__main__":
    main()
