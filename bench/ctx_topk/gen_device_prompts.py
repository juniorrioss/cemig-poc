#!/usr/bin/env python3
"""
gen_device_prompts.py — Monta os prompts REAIS (system + contexto v4 + pergunta) das 20
perguntas do roteiro/extra para as configs de custo levadas ao S24+, exatamente como o
AskPipeline montaria. Grava um .txt por (config, pergunta) p/ o llama-cli --file no aparelho
e um manifest JSON com a NR esperada e os docs recuperados (p/ acerto de norma).

Configs de custo (PART 2): k2_full (default atual), k5_full (máx trechos, pior custo),
k5_trim (máx trechos com recorte barato). São as 3 que decidem o trade-off janela×trechos.

Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_v4 import RetrieverV4
from pipeline import format_context, SYNTHESIS_SYSTEM_PROMPT

_HERE = Path(__file__).resolve().parent
DEVICE_Q = _HERE / "data" / "device_questions.jsonl"
OUT_DIR = _HERE / "data" / "device_prompts"

CONFIGS = {"k2_full": (2, False), "k5_full": (5, False), "k5_trim": (5, True)}


def main() -> None:
    qs = [json.loads(l) for l in DEVICE_Q.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions_by_id = {q["id"]: q["question"] for q in qs}
    retriever = RetrieverV4(questions_by_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"configs": {}, "questions": {q["id"]: {"doc": q["doc"], "source": q["source"],
                                                        "question": q["question"]} for q in qs}}

    for cfg, (top_k, trim) in CONFIGS.items():
        cfg_dir = OUT_DIR / cfg
        cfg_dir.mkdir(exist_ok=True)
        recs = {}
        for q in qs:
            qid = q["id"]
            sr = retriever.search(qid, q["question"], top_k=top_k)
            chunks = sr["chunks"]
            context = format_context(chunks, q["question"], trim, top_k)
            user_content = f"Contexto normativo consultado:\n{context}\n\nPergunta do eletricista:\n{q['question']}"
            # Prompt cru p/ llama-cli: usamos o template de chat do próprio modelo via -sys + -p.
            (cfg_dir / f"{qid}.sys.txt").write_text(SYNTHESIS_SYSTEM_PROMPT, encoding="utf-8")
            (cfg_dir / f"{qid}.usr.txt").write_text(user_content, encoding="utf-8")
            recs[qid] = {
                "chunks_docs": [c["doc"] for c in chunks],
                "top_doc": chunks[0]["doc"] if chunks else "",
                "expected_doc": q["doc"],
                "retrieval_hit_doc": q["doc"] in {c["doc"] for c in chunks},
                "user_chars": len(user_content),
            }
        manifest["configs"][cfg] = {"top_k": top_k, "trim": trim, "items": recs}

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prompts gravados em {OUT_DIR} para configs {list(CONFIGS)}")
    for cfg in CONFIGS:
        recs = manifest["configs"][cfg]["items"]
        hit = sum(1 for r in recs.values() if r["retrieval_hit_doc"])
        avg_chars = sum(r["user_chars"] for r in recs.values()) / len(recs)
        print(f"  {cfg}: doc-hit {hit}/{len(recs)} | user_content médio {avg_chars:.0f} chars")


if __name__ == "__main__":
    main()
