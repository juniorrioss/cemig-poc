#!/usr/bin/env python3
"""
e2e_build_f6.py — Passos 2 e 3 da F6: prompts fim-a-fim (formato do APP/F4) com o
retriever COM MANUAIS + top-5 -> compressão 50% (compressor da F5 estendido aos manuais).

Formato idêntico ao e2e_build.py da F5: system BAKED com a ferramenta, user = fala crua,
assistant = chamada buscar_norma(fala crua, nr explícita) INJETADA, tool = contexto da
célula; o modelo (f4-r32-ep2) gera só a síntese. Entre células muda SÓ o turno tool.

ORÇAMENTO: em TODAS as células comprimidas o orçamento é 50% dos tokens do top-2 CRU do
CONTROLE v4app da MESMA query (= T_a da F5). Assim chunk de manual (~1700 chars) não
infla o orçamento e as células são comparáveis token a token com a d50 da F5.

Células (nome -> (variante de retrieval, modo)):
  RECORTE 344 (benchmark_mobile, normas 10/18/34/13/31/33; ouro em norma OU manual):
    ctrl_d50   thr0.7_f2                      top5->50%  (controle SEM manuais, gate novo)
    op_d50     thr0.7_f2+man1.75+margin       top5->50%  (PONTO DE OPERAÇÃO escolhido)
    mid_d50    thr0.7_f2+man1.6               top5->50%  (pareto: sem gate, manual 54%)
    hi_d50     thr0.7_f2+man2                 top5->50%  (pareto: manual 78%, trava quebrada)
  POC 151 (corpus/qa_pairs_v2.jsonl; ressalva da F5):
    a_raw2     v4app                          top2 cru   (app atual)
    d_c5_r0.5  v4app                          top5->50%  (d50 da F5, SEM manuais)
    op_d50     thr0.7_f2+man1.75+margin       top5->50%

--parity: remonta a d50 v4app do recorte e compara byte-a-byte com prompts.jsonl.gz da F5.

Saída: data/e2e/prompts_{recorte,151}.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f6 as F6  # noqa: E402
import common_f5 as C  # noqa: E402
import tool_schema_f4 as T  # noqa: E402
from compress import ntok  # noqa: E402
from compress_f6 import CompressorF6  # noqa: E402
from eval_retrieval import variant_cfg  # noqa: E402
from metrics import covered  # noqa: E402
from retriever_f5 import MANUAL_ID_OFFSET, F5Retriever, R4  # noqa: E402

SCORER = {"mode": "hybrid", "lam": 0.3, "prior": 0.05}   # = F5 (e2e_build.SCORER)
RATIO = 0.5
OP = "thr0.7_f2+man1.75+margin"
CELLS_RECORTE = {"ctrl_d50": ("thr0.7_f2", "c5"), "op_d50": (OP, "c5"),
                 "mid_d50": ("thr0.7_f2+man1.6", "c5"), "hi_d50": ("thr0.7_f2+man2", "c5")}
CELLS_151 = {"a_raw2": ("v4app", "raw2"), "d_c5_r0.5": ("v4app", "c5"), "op_d50": (OP, "c5")}
CONTEXT_TYPE = {"raw2": "retrieved_top2_raw", "c5": "retrieved_top5_compressed"}


class Builder:
    def __init__(self):
        self.rtv = F5Retriever(with_manuals=True)
        self.comp = CompressorF6()
        align = {int(k): v for k, v in json.loads((F6.F5_CACHE / "align.json").read_text()).items()}
        align.update(self.rtv.man.span_store())
        self.align = align
        self.sys_msg = T.make_baked_system_message()

    def spans_of(self, cid: int):
        return self.align[cid]["spans"]

    def cell(self, q: str, variant: str, mode: str) -> Dict[str, Any]:
        qv = self.rtv.qc.get(q)
        ctrl = self.rtv.search(q, variant_cfg("v4app"), top_k=5)["ids"][:2]
        budget = int(ntok(self.comp.render_raw(ctrl)) * RATIO)
        ids = self.rtv.search(q, variant_cfg(variant), top_k=5)["ids"]
        if mode == "raw2":
            top = ids[:2]
            return {"text": self.comp.render_raw(top), "ids": top,
                    "spans": {c: self.spans_of(c) for c in top}}
        r = self.comp.compress(qv, ids[:5], budget, q=q, **SCORER)
        return {"text": r["text"], "ids": ids[:5], "spans": r["spans"]}

    def messages(self, q: str, ctx: str) -> List[Dict[str, Any]]:
        ex = R4.detect_explicit_nr(q)
        nr_arg = f"NR-{int(ex[3:])}" if ex else None
        return [self.sys_msg, T.make_user_message(q), T.make_tool_call_message(q, nr_arg),
                T.make_tool_result_message(ctx)]

    def gold_items(self, row) -> List[str]:
        items = set()
        for cid, u in self.comp.units.items():
            cid = int(cid)
            if self.align.get(cid, {}).get("file") != row["source_file"]:
                continue
            for un in u["units"]:
                if un["item"] and covered(un["spans"], row["ev"]) > 0:
                    items.add(un["item"])
        return sorted(items)


def _ctx_type(cell_mode: str, ids: List[int]) -> str:
    base = CONTEXT_TYPE[cell_mode]
    return base + "_manuals" if any(c >= MANUAL_ID_OFFSET for c in ids) else base


def build_recorte(b: Builder, out: Path) -> None:
    rows = [r for r in C.load_bench() if C.nr_num(r["source_file"]) in C.RECORTE_NORMS]
    assert len(rows) == 344, len(rows)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            q = row["question"]
            gi = b.gold_items(row)
            for cell, (variant, mode) in CELLS_RECORTE.items():
                c = b.cell(q, variant, mode)
                rec = {"key": f"{row['id']}::{cell}", "id": row["id"], "cell": cell,
                       "variant": variant, "suite": "recorte344",
                       "base_qa_id": row["base_qa_id"], "vocabulary_level": row["vocabulary_level"],
                       "question_type": row["question_type"], "gold_kind": row["gold_kind"],
                       "source_file": row["source_file"], "ctx_ntok": ntok(c["text"]),
                       "ctx_ids": c["ids"], "n_manual_ctx": sum(1 for x in c["ids"] if x >= MANUAL_ID_OFFSET),
                       "context_type": _ctx_type(mode, c["ids"]),
                       "context_given": c["text"], "gold_items": gi,
                       "ctx_has_gold_span": any(
                           b.align.get(cid, {}).get("file") == row["source_file"]
                           and covered(ss, row["ev"]) > 0 for cid, ss in c["spans"].items()),
                       "messages": b.messages(q, c["text"])}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"[ok] {out}: {n} prompts")


def build_151(b: Builder, out: Path) -> None:
    sys.path.insert(0, str(C.REPO))
    from corpus.eval_retrieval import check_hit
    gab = {g["id"]: g for g in C.read_jsonl(C.GAB151)}
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for it in C.load_151():
            q = it["question"]
            for cell, (variant, mode) in CELLS_151.items():
                c = b.cell(q, variant, mode)
                hit = any(check_hit(b.rtv.chunk(x) | {"id": x}, it)
                          for x in (c["ids"][:2] if mode == "raw2" else c["ids"][:5]))
                rec = {"key": f"151::{it['id']}::{cell}", "id": it["id"], "cell": cell,
                       "variant": variant, "suite": "poc151", "doc": it["doc"],
                       "section": it["section"], "ctx_ntok": ntok(c["text"]),
                       "ctx_ids": c["ids"], "n_manual_ctx": sum(1 for x in c["ids"] if x >= MANUAL_ID_OFFSET),
                       "context_type": _ctx_type(mode, c["ids"]), "context_given": c["text"],
                       "retrieval_hit": hit, "gold_chunk_ids": gab[it["id"]].get("gold_chunk_ids"),
                       "messages": b.messages(q, c["text"])}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"[ok] {out}: {n} prompts")


def parity(b: Builder) -> None:
    """d50 v4app (F5) reproduzida byte-a-byte? (mesmo retriever, compressor com manuais)."""
    with gzip.open(C.HERE / "data" / "e2e" / "prompts.jsonl.gz", "rt", encoding="utf-8") as fh:
        f5 = {}
        for l in fh:
            d = json.loads(l)
            if d["cell"] == "d_c5_r0.5":
                f5[d["id"]] = d
    rows = {r["id"]: r for r in C.load_bench()}
    same = diff = 0
    for rid, d in f5.items():
        c = b.cell(rows[rid]["question"], "v4app", "c5")
        if c["text"] == d["messages"][-1]["content"] and c["ids"] == d["ctx_ids"]:
            same += 1
        else:
            diff += 1
    print(f"[parity] d50 v4app vs F5: {same} idênticos, {diff} diferentes")
    F6.DATA.joinpath("e2e").mkdir(parents=True, exist_ok=True)
    C.write_json(F6.DATA / "e2e" / "parity_d50_f5.json", {"identical": same, "different": diff})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="recorte,151")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    b = Builder()
    if args.parity:
        parity(b)
    out = F6.DATA / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    if "recorte" in args.which:
        build_recorte(b, out / "prompts_recorte.jsonl")
    if "151" in args.which:
        build_151(b, out / "prompts_151.jsonl")


if __name__ == "__main__":
    main()
