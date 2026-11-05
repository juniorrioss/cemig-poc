#!/usr/bin/env python3
"""
export_parquet_quant.py — parquets canônicos por formato × rodada (schema estendido da F6 +
colunas novas): data/eval/lfm3-f4-r32-ep2-quant-recorte344-op-d50-<engine_format>-r<k>.parquet.

Colunas novas vs F6: engine_format, round, seed, tok_parity, decode_tps (registrado durante a
geração; indicativo), prefill_tps. A referência hf_bf16 já está nos parquets da F6
(bench/lfm3_f6/data/eval/lfm3-f4-r32-ep2-f6-recorte344-op-d50.parquet, engine implícito).

Uso: <venv>/bin/python bench/lfm3_quant/export_parquet_quant.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "bench" / "lfm3_f6", REPO / "bench" / "lfm3_f5", REPO / "bench" / "lfm3_f4"):
    sys.path.insert(0, str(p))
import common_f6  # noqa: E402,F401  (ajusta sys.path da F5)
import common_f5 as C  # noqa: E402
import tool_schema_f4 as T  # noqa: E402

DATA = HERE / "data"
CKPT = "f4-r32-ep2"
ENGINES = {
    "gguf": {"runtime": "llama.cpp v0.9.0 (cd959ba) llama-server CUDA sm_121", "device": "Spark GB10"},
    "litert": {"runtime": "LiteRT-LM 0.19.0 (Python), Engine nova por prompt, Conversation + template identidade",
               "device": "Spark GB10 (CPU=XNNPACK 6 threads | GPU=WebGPU/Vulkan fp16 activations)"},
}


def main() -> None:
    out = DATA / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sysmsg = T.baked_system_content()
    sha = hashlib.sha256(sysmsg.encode()).hexdigest()[:16]
    bench = {r["id"]: r for r in C.load_bench()}
    prompts = {p["key"]: p for p in C.read_jsonl(DATA / "prompts_op344.jsonl")}
    for sp in sorted((DATA / "e2e").glob("scored_*.jsonl")):
        tag = sp.stem[len("scored_"):]
        fmt, rnd = tag.rsplit("-r", 1)
        gens = {g["key"]: g for g in C.read_jsonl(DATA / "gen" / f"gen_{tag}.jsonl")}
        recs = []
        for s in C.read_jsonl(sp):
            p, g, b = prompts[s["key"]], gens[s["key"]], bench[s["id"]]
            eng = ENGINES["gguf" if fmt.startswith("gguf") else "litert"]
            recs.append({
                "id": s["id"], "base_qa_id": b["base_qa_id"], "question": b["question"],
                "answer_gold": b["answer"], "question_type": b["question_type"],
                "source_id": b["source_id"], "vocabulary_level": b["vocabulary_level"],
                "source_file": b["source_file"], "gold_kind": b["gold_kind"],
                "evidence_text": "\n\n".join(e["text"] for e in b["evidence"]),
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
                "system_prompt_sha": sha, "system_prompt_ref": "f4_baked_tools_v1",
                "checkpoint": CKPT, "cell": "quant-recorte344-op_d50",
                "engine_format": fmt, "round": int(rnd), "seed": g.get("seed"),
                "tok_parity": g.get("tok_parity"), "decode_tps": g.get("decode_tps"),
                "prefill_tps": g.get("prefill_tps"),
                "gen_params": json.dumps({"temperature": 0.1, "top_k": 50, "top_p": 1.0,
                                          "repetition_penalty": 1.0 if "rep1.0" in fmt else 1.1,
                                          "repeat_window": 64, "max_new_tokens": 768,
                                          "seed": g.get("seed"), "thinking": "off", **eng}),
            })
        path = out / f"lfm3-{CKPT}-quant-recorte344-op-d50-{fmt.replace('_', '-')}-r{rnd}.parquet"
        pq.write_table(pa.Table.from_pylist(recs), path, compression="zstd")
        print(f"[ok] {path.name}: {len(recs)} linhas")


if __name__ == "__main__":
    main()
