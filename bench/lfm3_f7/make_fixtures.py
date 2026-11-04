#!/usr/bin/env python3
"""
make_fixtures.py — F7: fixtures de PARIDADE app (Kotlin) × espelho Python.

1) TEMPLATE (interpretador .venv-train, transformers 5.x): renderiza com o tokenizer/
   chat_template do checkpoint (apply_chat_template, enable_thinking=False) 4 cenários
   -> android/app/src/test/resources/lfm3_render/*.txt (Lfm3ToolFormatParityTest).
       A decisão 1 turno · B síntese (tool) · C multiturno (histórico com tool + nova fala)
       D histórico com resposta direta (reuso) + nova fala
   Uso: .venv-train/bin/python bench/lfm3_f7/make_fixtures.py --template

2) PIPELINE (classifier/.venv, sem rede: embeddings de query do cache qemb.npz da F5):
   para N perguntas do recorte (op_d50) grava o vetor de query, o resultado do classificador,
   os rankings de cada sinal, o top-5 fundido, o orçamento e o contexto comprimido esperado
   -> android/app/src/test/resources/f7_parity/cases.json (F7ParityTest).
   Uso: classifier/.venv/bin/python bench/lfm3_f7/make_fixtures.py --pipeline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "android" / "app" / "src" / "test" / "resources"
MERGED = Path("/tmp/f4_base/lfm3-1.7b-data2x-ep4-merged")   # tokenizer+template do treino


def template() -> None:
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f5"))
    import tool_schema_f4 as T
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MERGED), trust_remote_code=True)
    sysm = T.make_baked_system_message()
    q1 = "Qual a distância segura pra trabalhar perto de rede de 13,8 kV? Segundo a NR-10."
    ctx1 = ("[1] (nr-10 - Anexo II - Zona de risco e zona controlada):\nNR-10 · Anexo II — "
            "(item 10.2) Tabela de raios: ≥10 e 15 → 0,38/1,38 […] 𝐌 teste 'aspas' \\ barra")
    a1 = "Pela NR-10, Anexo II, para 13,8 kV o raio da zona de risco é 0,38 m."
    q2 = "e se não der pra desligar?"
    q3 = "obrigado!"
    a3 = "De nada! Qualquer dúvida sobre as normas, é só perguntar."
    cases = {
        "A_decision": [sysm, T.make_user_message(q1)],
        "B_synthesis": [sysm, T.make_user_message(q1), T.make_tool_call_message(q1, "NR-10"),
                        T.make_tool_result_message(ctx1)],
        "C_multiturn": [sysm, T.make_user_message(q1), T.make_tool_call_message(q1, "NR-10"),
                        T.make_tool_result_message(ctx1), T.make_assistant_message(a1),
                        T.make_user_message(q2)],
        "D_reuse_direct": [sysm, T.make_user_message(q1), T.make_tool_call_message(q1, None),
                           T.make_tool_result_message(ctx1), T.make_assistant_message(a1),
                           T.make_user_message(q3), T.make_assistant_message(a3),
                           T.make_user_message(q2)],
    }
    out = RES / "lfm3_render"
    out.mkdir(parents=True, exist_ok=True)
    for name, msgs in cases.items():
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
    (out / "inputs.json").write_text(json.dumps(
        {"q1": q1, "ctx1": ctx1, "a1": a1, "q2": q2, "q3": q3, "a3": a3}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (RES / "lfm3_tools_system_prompt.txt").write_text(T.baked_system_content(), encoding="utf-8")
    print(f"[ok] {len(cases)} fixtures -> {out}")


def pipeline(n: int) -> None:
    import os
    os.environ.setdefault("LFM3_TOKENIZER", str(MERGED / "tokenizer.json"))
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f6"))
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f5"))
    from e2e_build_f6 import OP, RATIO, SCORER, Builder
    from compress import ntok
    from eval_retrieval import variant_cfg
    from retriever_f5 import R4
    from eval_common import app_fts_query
    b = Builder()
    P = [json.loads(l) for l in (REPO / "bench/lfm3_f6/data/e2e/prompts_recorte.jsonl").open()]
    P = [p for p in P if p["cell"] == "op_d50"]
    # mistura: todos com manual no contexto primeiro, depois norma (diversidade de gate)
    P.sort(key=lambda p: (-p["n_manual_ctx"], p["id"]))
    sel = P[: n // 2] + P[-(n - n // 2):]
    cfg = variant_cfg(OP)
    cases = []
    for p in sel:
        q = p["messages"][1]["content"]
        qv = b.rtv.qc.get(q)
        s1, mode = b.rtv.bm25_signal(q, cfg)
        sT = b.rtv._knn(qv, b.rtv.ids_t, b.rtv.mat_t)
        man_on = b.rtv.manual_wanted(q, cfg, qv, sT)
        res = b.rtv.search(q, cfg, top_k=5)
        ctrl = b.rtv.search(q, variant_cfg("v4app"), top_k=5)["ids"][:2]
        budget = int(ntok(b.comp.render_raw(ctrl)) * RATIO)
        comp = b.comp.compress(qv, res["ids"][:5], budget, q=q, **SCORER)
        preds, p1 = b.rtv.classify(q)
        assert comp["text"] == p["messages"][-1]["content"], p["id"]
        mb, md = b.rtv.man.signals(q, qv)
        sE = b.rtv._knn(qv, b.rtv.ids_e, b.rtv.mat_e)
        cases.append({
            "id": p["id"], "question": q, "fts_query": app_fts_query(q),
            "qvec": [round(float(x), 8) for x in qv],
            "clf_top": preds, "clf_p1": p1, "gate_mode": mode,
            "bm25_ids": [r["id"] for r in s1][:60], "manual_on": bool(man_on),
            "dense_text_ids": [r["id"] for r in sT], "dense_exp_ids": [r["id"] for r in sE],
            "text_top1_sim": sT[0]["score"], "manual_max_sim": float((b.rtv.man.mat @ qv).max()),
            "manual_bm25_ids": [r["id"] for r in mb], "manual_dense_ids": [r["id"] for r in md],
            "fused_top5": res["ids"][:5], "ctrl_top2": ctrl, "budget": budget,
            "context": comp["text"],
        })
    out = RES / "f7_parity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {len(cases)} casos -> {out} (manual_on={sum(c['manual_on'] for c in cases)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("-n", type=int, default=40)
    a = ap.parse_args()
    if a.template:
        template()
    if a.pipeline:
        pipeline(a.n)


if __name__ == "__main__":
    main()
