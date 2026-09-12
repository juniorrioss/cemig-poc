#!/usr/bin/env python3
"""
sweep_frentea.py — FRENTE A: varredura de estratégias de dados do classificador.

Treina o logreg vencedor sob várias receitas de dados-de-graça (cap por NR, só-eletricista,
peso amostral, class_weight) e, para CADA uma, reporta lado a lado (TRAVA 5):
  - OTIMISTA  : top-1/2 no test_samegen (mesmo gerador)
  - HONESTA   : top-1/2 no test_diffgen (gerador diferente) + holdout real 151+20
  - DECISIVA  : R@2/R@5 final nas 151 (fusão 3-sinais da v3) — a régua que decide.

Nenhuma receita "vence" por acurácia: só conta se a DECISIVA subir. Descartes viram
resultado registrado. Reusa train_v4 (treino) e o pipeline v3 (recall).

Uso (classifier/.venv):  ../classifier/.venv/bin/python sweep_frentea.py
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from data_utils import load_holdout  # noqa: E402
from eval_common import evaluate, load_holdout as load_ho_qa  # noqa: E402
from bm25 import HybridBm25  # noqa: E402
from rrf import rrf_fuse  # noqa: E402
from v4_common import HELD_OUT_NRS, load_jsonl  # noqa: E402
from train_v4 import (build_dataset, fit_pipeline, score, topk_acc,  # noqa: E402
                      split_by_nr)

EXP_W = (1.5, 3.0, 2.0, 1.0, 1.0)
DB = str(_ROOT / "retrieval3" / "indices" / "index_hf_36nr_exp.db")
DTEXT = _ROOT / "retrieval3" / "results" / "bin_text_rank.json"
DEXP = _ROOT / "retrieval3" / "results" / "bin_exponly_rank.json"

# Receitas a testar (nome -> kwargs de build/fit)
RECIPES: List[Dict[str, Any]] = [
    {"name": "v1_baseline", "datasets": ["v1"]},
    {"name": "v1+free_all", "datasets": ["v1", "free"]},
    {"name": "v1+free_cap200", "datasets": ["v1", "free"], "free_cap": 200},
    {"name": "v1+free_cap100", "datasets": ["v1", "free"], "free_cap": 100},
    {"name": "v1+free_cap100_w0.5", "datasets": ["v1", "free"], "free_cap": 100, "free_weight": 0.5},
    {"name": "v1+free_cap60_w0.3", "datasets": ["v1", "free"], "free_cap": 60, "free_weight": 0.3},
    {"name": "v1+free_eleconly_cap150", "datasets": ["v1", "free"], "free_elec_only": True, "free_cap": 150},
    {"name": "v1+free_eleconly_cap150_w0.5", "datasets": ["v1", "free"], "free_elec_only": True, "free_cap": 150, "free_weight": 0.5},
]


def make_clf(recipe: Dict[str, Any]):
    train, _ = build_dataset(
        recipe["datasets"], None, None,
        free_cap=recipe.get("free_cap", 0),
        free_elec_only=recipe.get("free_elec_only", False),
        seed=recipe.get("seed", 42),
    )
    texts = [r["text"] for r in train]
    y = [r["primary"] for r in train]
    fw = recipe.get("free_weight", 1.0)
    sw = None
    if fw != 1.0:
        sw = np.array([fw if r.get("kind") == "free" else 1.0 for r in train])
    cw = recipe.get("class_weight")
    char, word, clf, dim = fit_pipeline(texts, y, sample_weight=sw, class_weight=cw)
    return char, word, clf, len(texts)


def acc_on(char, word, clf, recs):
    if not recs:
        return {"n": 0}
    texts = [r["text"] for r in recs]
    golds = [r.get("gold", r.get("primary")) for r in recs]
    return {"n": len(recs), **topk_acc(char, word, clf, texts, golds)}


def decisive_recall(char, word, clf, dtext, dexp):
    """Recall final nas 151 com a fusão 3-sinais (salva pkl temporário p/ HybridBm25)."""
    import pickle, tempfile
    tmp = Path(tempfile.mkstemp(suffix=".pkl")[1])
    with open(tmp, "wb") as f:
        pickle.dump({"char": char, "word": word, "clf": clf, "name": "sweep"}, f)
    hy = HybridBm25(DB, str(tmp), weights=EXP_W)
    ho = load_ho_qa("qa_v2")

    def s1(it):
        return hy.rank(it.question, 5)

    def fusion(it):
        return rrf_fuse([hy.rank(it.question, 60), dtext.get(it.id, []), dexp.get(it.id, [])],
                        [1.0, 1.0, 1.0], k=30.0, limit=5)
    m1 = evaluate(ho, s1)
    mf = evaluate(ho, fusion)
    tmp.unlink(missing_ok=True)
    return m1, mf


def main() -> None:
    dtext = json.loads(DTEXT.read_text(encoding="utf-8"))
    dexp = json.loads(DEXP.read_text(encoding="utf-8"))
    same = load_jsonl(_HERE / "data" / "test_samegen.jsonl")
    diff = load_jsonl(_HERE / "data" / "test_diffgen.jsonl")
    ho = [{"text": h["text"], "gold": h["gold"]} for h in load_holdout()]
    ho_held = [r for r in ho if r["gold"] in set(HELD_OUT_NRS)]

    rows = []
    for rec in RECIPES:
        t0 = time.time()
        char, word, clf, n = make_clf(rec)
        m_same = acc_on(char, word, clf, same)
        m_diff = acc_on(char, word, clf, diff)
        m_ho = acc_on(char, word, clf, ho)
        m_held = acc_on(char, word, clf, ho_held)
        m1, mf = decisive_recall(char, word, clf, dtext, dexp)
        mask1 = round(m_same["top1"] - m_diff["top1"], 1)
        mask2 = round(m_same["top2"] - m_diff["top2"], 1)
        row = {
            "name": rec["name"], "n_train": n,
            "opt_same_top1": m_same["top1"], "opt_same_top2": m_same["top2"],
            "hon_diff_top1": m_diff["top1"], "hon_diff_top2": m_diff["top2"],
            "hon_holdout_top1": m_ho["top1"], "hon_holdout_top2": m_ho["top2"],
            "held_nrs_top2": m_held.get("top2", 0),
            "mask_top1_pp": mask1, "mask_top2_pp": mask2,
            "dec_s1_r2": m1.recall_at_2, "dec_s1_r5": m1.recall_at_5,
            "dec_fus_r2": mf.recall_at_2, "dec_fus_r5": mf.recall_at_5,
            "fit_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"[{rec['name']:<28}] same2={m_same['top2']:.1f} diff2={m_diff['top2']:.1f} "
              f"ho2={m_ho['top2']:.1f} | DEC fus R@2={mf.recall_at_2:.1f} R@5={mf.recall_at_5:.1f} "
              f"(mask2={mask2:+.1f}pp) {row['fit_s']}s", flush=True)

    out = {"exp_w": EXP_W, "rrf_k": 30, "recipes": rows,
           "baseline_dec_fus_r2": 51.0, "note": "só sobe se DECISIVA (fus R@2/R@5) crescer"}
    Path(_HERE / "results" / "sweep_frentea.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Tabela final
    print("\n" + "=" * 120)
    print("  FRENTE A — 3 MEDIÇÕES LADO A LADO (otimista | honesta | DECISIVA)")
    print("=" * 120)
    hdr = (f"| {'receita':<28} | {'same t2':>7} | {'diff t2':>7} | {'ho t2':>6} | "
           f"{'held t2':>7} | {'mask2':>6} | {'DEC R@2':>7} | {'DEC R@5':>7} |")
    print(hdr)
    print("|" + "-" * 30 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 8 + "|"
          + "-" * 9 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 9 + "|")
    for r in rows:
        print(f"| {r['name']:<28} | {r['opt_same_top2']:>6.1f}% | {r['hon_diff_top2']:>6.1f}% | "
              f"{r['hon_holdout_top2']:>5.1f}% | {r['held_nrs_top2']:>6.1f}% | {r['mask_top2_pp']:>+5.1f} | "
              f"{r['dec_fus_r2']:>6.1f}% | {r['dec_fus_r5']:>6.1f}% |")
    print("=" * 120)
    print("Baseline DECISIVA (v1): fus R@2=51.0 R@5=63.6. Só embarca se a DECISIVA subir.")


if __name__ == "__main__":
    main()
