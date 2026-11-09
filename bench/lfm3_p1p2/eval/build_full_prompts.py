#!/usr/bin/env python3
"""
build_full_prompts.py — prompts e2e op_d50 para a BATERIA COMPLETA (trava anti-overfit
do capitão: o recorte 344 compartilha normas com o treino e não basta como prova).

- 1536 do benchmark_mobile: MESMO Builder/célula op_d50 da F6 (retriever op + top-5 -> 50%,
  formato do APP); o recorte 344 sai byte a byte igual ao `prompts_recorte.jsonl` da F6
  (verificado aqui e publicado em data/full_prompts_parity.json).
- 151 coloquiais da POC: copiados da F6 (`prompts_151.jsonl`, célula op_d50).

Saída: eval/prompts_op_d50_1536.jsonl, eval/prompts_op_d50_151.jsonl
Uso: LFM3_TOKENIZER=... classifier/.venv/bin/python bench/lfm3_p1p2/eval/build_full_prompts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f6"))

import e2e_build_f6 as E  # noqa: E402
import common_f5 as C  # noqa: E402
from compress import ntok  # noqa: E402
from metrics import covered  # noqa: E402
from retriever_f5 import MANUAL_ID_OFFSET  # noqa: E402


def main() -> None:
    b = E.Builder()
    variant, mode = E.CELLS_RECORTE["op_d50"]
    f6 = {json.loads(l)["id"]: json.loads(l) for l in
          (REPO / "bench" / "lfm3_f6" / "data" / "e2e" / "prompts_recorte.jsonl").open()
          if json.loads(l)["cell"] == "op_d50"}
    same = diff = 0
    out = HERE / "prompts_op_d50_1536.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in C.load_bench():
            q = row["question"]
            c = b.cell(q, variant, mode)
            rec = {"key": f"{row['id']}::op_d50", "id": row["id"], "cell": "op_d50",
                   "variant": variant, "suite": "recorte344",   # rótulo de schema do score_f6
                   "base_qa_id": row["base_qa_id"], "vocabulary_level": row["vocabulary_level"],
                   "question_type": row["question_type"], "gold_kind": row["gold_kind"],
                   "source_file": row["source_file"], "ctx_ntok": ntok(c["text"]),
                   "ctx_ids": c["ids"], "n_manual_ctx": sum(1 for x in c["ids"] if x >= MANUAL_ID_OFFSET),
                   "context_type": E._ctx_type(mode, c["ids"]), "context_given": c["text"],
                   "gold_items": b.gold_items(row),
                   "ctx_has_gold_span": any(
                       b.align.get(cid, {}).get("file") == row["source_file"]
                       and covered(ss, row["ev"]) > 0 for cid, ss in c["spans"].items()),
                   "messages": b.messages(q, c["text"]),
                   "in_recorte": C.nr_num(row["source_file"]) in C.RECORTE_NORMS}
            if row["id"] in f6:
                if f6[row["id"]]["messages"] == rec["messages"]:
                    same += 1
                else:
                    diff += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    b.rtv.qc.save()
    n151 = 0
    with (HERE / "prompts_op_d50_151.jsonl").open("w", encoding="utf-8") as fh:
        for l in (REPO / "bench" / "lfm3_f6" / "data" / "e2e" / "prompts_151.jsonl").open():
            if json.loads(l)["cell"] == "op_d50":
                fh.write(l)
                n151 += 1
    par = {"recorte344_identical_to_f6": same, "different": diff, "n151": n151}
    (HERE.parent / "data").mkdir(exist_ok=True)
    (HERE.parent / "data" / "full_prompts_parity.json").write_text(json.dumps(par, indent=2))
    print(par)


if __name__ == "__main__":
    main()
