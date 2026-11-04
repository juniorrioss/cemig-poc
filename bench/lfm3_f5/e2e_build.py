#!/usr/bin/env python3
"""
e2e_build.py — Frente 3 fim-a-fim: monta os prompts das células de compressão no RECORTE
de 344 (normas 10/18/34/13/31/33) para o LFM3 f4-r32-ep2.

Formato = o do APP híbrido/F4 (o modelo treinado em tool-calling): system BAKED com a
ferramenta, turno user com a fala crua, turno assistant com a chamada buscar_norma(fala
crua, nr explícita) INJETADA e turno tool com o contexto da célula. O modelo gera só a
síntese final (thinking OFF). Assim o que muda entre células é SÓ o conteúdo do turno tool.

Células (retrieval = v4app, igual ao app; compressor = compress.Compressor):
  a_raw2      topK=2 cru (controle, como o app)
  b_c2_r0.5   topK=2 comprimido a 50% dos tokens de (a)
  b_c2_r0.7   topK=2 comprimido a 70%
  c_nb_r1.0   topK=2+vizinhos±1 comprimido ao MESMO orçamento de (a)
  c_nb_r0.7   idem a 70%
  d_c5_r0.7   topK=5 comprimido a 70%   (opcional do brief)
  d_c5_r0.5   topK=5 comprimido a 50%

Saída: data/e2e/prompts.jsonl (1 linha por id×célula: messages + meta + ctx_ntok +
gold_items p/ a métrica de citação de item).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common_f5 as C  # noqa: E402
import eval_compress as EC  # noqa: E402
import tool_schema_f4 as T  # noqa: E402
from compress import Compressor, ntok  # noqa: E402
from eval_retrieval import variant_cfg  # noqa: E402
from metrics import covered  # noqa: E402
from retriever_f5 import F5Retriever, R4  # noqa: E402

CELLS = ["a_raw2", "b_c2_r0.5", "b_c2_r0.7", "c_nb_r1.0", "c_nb_r0.7", "d_c5_r0.7", "d_c5_r0.5"]
SCORER = {"mode": "hybrid", "lam": 0.3, "prior": 0.05}


def gold_items(comp: Compressor, row, align) -> List[str]:
    """Códigos de item das unidades (de TODOS os chunks do arquivo-ouro) que cobrem a evidência."""
    items = set()
    for cid, ch in comp.chunks.items():
        if align[cid]["file"] != row["source_file"]:
            continue
        for u in comp.units[str(cid)]["units"]:
            if u["item"] and covered(u["spans"], row["ev"]) > 0:
                items.add(u["item"])
    return sorted(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v4app")
    ap.add_argument("--out", default=str(C.DATA / "e2e" / "prompts.jsonl"))
    args = ap.parse_args()
    EC.SCORER.update(SCORER)
    cfg = variant_cfg(args.variant)
    rtv, comp = F5Retriever(), Compressor()
    align = {int(k): v for k, v in json.loads((C.CACHE / "align.json").read_text()).items()}
    rows = [r for r in C.load_bench() if C.nr_num(r["source_file"]) in C.RECORTE_NORMS]
    assert len(rows) == 344, len(rows)
    sys_msg = T.make_baked_system_message()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            q = row["question"]
            cells = EC.cells_for(rtv, comp, q, cfg, [0.5, 0.7, 1.0])
            gi = gold_items(comp, row, align)
            ex = R4.detect_explicit_nr(q)          # = detect_nr do F4 / app
            nr_arg = f"NR-{int(ex[3:])}" if ex else None
            for cell in CELLS:
                c = cells[cell]
                msgs = [sys_msg, T.make_user_message(q),
                        T.make_tool_call_message(q, nr_arg),
                        T.make_tool_result_message(c["text"])]
                rec = {"key": f"{row['id']}::{cell}", "id": row["id"], "cell": cell,
                       "base_qa_id": row["base_qa_id"], "vocabulary_level": row["vocabulary_level"],
                       "question_type": row["question_type"], "gold_kind": row["gold_kind"],
                       "source_file": row["source_file"], "ctx_ntok": ntok(c["text"]),
                       "ctx_ids": c["ids"], "gold_items": gi,
                       "ctx_has_gold_span": any(
                           align.get(cid, {}).get("file") == row["source_file"]
                           and covered(ss, row["ev"]) > 0 for cid, ss in c["spans"].items()),
                       "messages": msgs}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"[ok] {out}: {n} prompts ({len(rows)} queries x {len(CELLS)} células)")


if __name__ == "__main__":
    main()
