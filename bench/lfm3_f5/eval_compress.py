#!/usr/bin/env python3
"""
eval_compress.py — Frente 3 (proxy sem LLM): quanto da evidência SOBREVIVE à compressão?

Para cada query das duas réguas, monta os contextos das células da F3 e mede, no TEXTO
que o LFM receberia:
  - tokens de contexto (tokenizer do LFM3);
  - fact_recall = fração dos fatos do gabarito (151: bench/regua gabarito_151; bench:
    lfm3_f0 gabarito_384) presentes no contexto (mesmo casamento barato da régua honesta);
  - evrec (só bench) = chars de evidência cobertos pelos spans de FONTE do contexto;
  - cite_ok = algum trecho do contexto que cobre a evidência traz o código de item (sempre
    verdadeiro por construção na compressão: cada unidade carrega "(item X)").

Células (ordem do brief):
  a_raw2      topK=2 cru (como o app)                                   orçamento = T_a
  b_c2_r{x}   topK=2 comprimido a x·T_a
  c_nb_1.0    topK=2+vizinhos±1 comprimido de volta a T_a (MESMO orçamento de a)
  c_nb_r{x}   idem a x·T_a
  d_c5_r{x}   topK=5 comprimido a x·T_a
Retrieval = config `--variant` (padrão v4app = app).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "bench" / "regua"))

import common_f5 as C  # noqa: E402
from compress import Compressor, ntok  # noqa: E402
from eval_retrieval import variant_cfg  # noqa: E402
from metrics import covered, ev_chars, mean_dicts  # noqa: E402
from retriever_f5 import F5Retriever  # noqa: E402
from ruler import PRESENT_RATIO, _content_tokens, _fact_match_ratio  # noqa: E402


def fact_recall(text: str, facts: List[str]) -> float:
    if not facts:
        return 0.0
    toks = _content_tokens(text)
    return sum(1 for f in facts if _fact_match_ratio(f, toks) >= PRESENT_RATIO) / len(facts)


SCORER = {"mode": "dense", "lam": 0.3, "prior": 0.0}


def cells_for(rtv, comp, q: str, cfg, ratios) -> Dict[str, Dict[str, Any]]:
    res = rtv.search(q, cfg.with_(neighbors=1), top_k=5)
    top2, top5, nb = res["ids"][:2], res["ids"][:5], res["expanded"]
    qv = rtv.qc.get(q)
    raw = comp.render_raw(top2)
    T = ntok(raw)
    raw_spans = {c: rtv_spans(c) for c in top2}
    out = {"a_raw2": {"text": raw, "ntok": T, "spans": raw_spans, "ids": top2}}
    for x in ratios:
        r = comp.compress(qv, top2, int(T * x), q=q, **SCORER)
        out[f"b_c2_r{x}"] = r | {"ids": top2}
        r = comp.compress(qv, nb, int(T * x), q=q, **SCORER)
        out[f"c_nb_r{x}"] = r | {"ids": nb}
        r = comp.compress(qv, top5, int(T * x), q=q, **SCORER)
        out[f"d_c5_r{x}"] = r | {"ids": top5}
    return out


_ALIGN = None


def rtv_spans(cid: int):
    global _ALIGN
    if _ALIGN is None:
        _ALIGN = {int(k): v for k, v in json.loads(
            (C.CACHE / "align.json").read_text(encoding="utf-8")).items()}
    return _ALIGN[cid]["spans"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v4app")
    ap.add_argument("--ratios", default="0.3,0.5,0.7,1.0")
    ap.add_argument("--mode", default="dense")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--prior", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]
    SCORER.update(mode=args.mode, lam=args.lam, prior=args.prior)
    cfg = variant_cfg(args.variant)
    rtv = F5Retriever()
    comp = Compressor()
    gab151 = {g["id"]: g["facts"] for g in C.read_jsonl(C.GAB151)}
    gab384 = {g["base_qa_id"]: g["facts"] for g in C.read_jsonl(C.GAB384)}
    align = {int(k): v for k, v in json.loads((C.CACHE / "align.json").read_text()).items()}

    res: Dict[str, Dict[str, List[Dict[str, float]]]] = {"poc151": {}, "bench_norma": {},
                                                          "recorte344": {}}
    for it in C.load_151():
        cells = cells_for(rtv, comp, it["question"], cfg, ratios)
        for name, c in cells.items():
            res["poc151"].setdefault(name, []).append(
                {"ntok": float(c["ntok"]), "fact_recall": fact_recall(c["text"], gab151[it["id"]])})
    rows = C.load_bench()
    for row in rows:
        in_rec = C.nr_num(row["source_file"]) in C.RECORTE_NORMS
        if row["gold_kind"] != "norma" and not in_rec:
            continue
        cells = cells_for(rtv, comp, row["question"], cfg, ratios)
        evn = max(1, ev_chars(row["ev"]))
        for name, c in cells.items():
            sp = [s for cid, ss in c["spans"].items()
                  if align[cid]["file"] == row["source_file"] for s in ss]
            d = {"ntok": float(c["ntok"]),
                 "fact_recall": fact_recall(c["text"], gab384[row["base_qa_id"]]),
                 "evrec": covered(sp, row["ev"]) / evn}
            if row["gold_kind"] == "norma":
                res["bench_norma"].setdefault(name, []).append(d)
            if in_rec:
                res["recorte344"].setdefault(name, []).append(d)
    agg = {reg: {name: mean_dicts(v) | {"n": len(v)} for name, v in cells.items()}
           for reg, cells in res.items()}
    C.write_json(C.DATA / "compress" / f"proxy_{args.variant}{args.tag}.json", agg)
    for reg, cells in agg.items():
        print(f"== {reg}")
        for name, m in cells.items():
            print(f"  {name:<12} tok {m['ntok']:7.1f}  fact_recall {m['fact_recall']:.3f}"
                  + (f"  evrec {m['evrec']:.3f}" if "evrec" in m else ""))


if __name__ == "__main__":
    main()
