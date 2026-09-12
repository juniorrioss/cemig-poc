#!/usr/bin/env python3
"""
train_v4.py — FRENTE A: classificador de NR reforçado (POC CEMIG, retrieval v4).

NÃO refaz o shootout (logreg venceu, ver classifier/README.md). Treina SÓ o logreg
vencedor, variando os DADOS de treino para medir o efeito de:
  - pares-de-graça das expansões da v3 (pergunta coloquial -> NR, milhares de graça);
  - aumento com variação de REGISTRO (gírias, forma indireta, truncada, erro de ASR).

TRAVAS ANTI-MASCARAMENTO (reporta as 3 medições lado a lado):
  - TRAVA 1: holdout 151+20 nunca entra em treino (garantido em v4_common).
  - TRAVA 2: reserva nr-33/16/26 INTEIRAS fora do treino; reporta acurácia nelas.
  - TRAVA 3: dois test sets — mesmo-gerador (otimista) e gerador-diferente (honesto).
  A distância otimista−honesta É o "tamanho da mascarada".

Mantém o formato do pipeline sklearn (char_wb 2-5 + word 1-2 + LogReg C=8), idêntico
ao que o export Kotlin (NRC2, nr_classifier.bin 4 MB) consome — sem tocar no formato.

Uso (classifier/.venv):
    ../classifier/.venv/bin/python train_v4.py --datasets v1,free --nr-split \
        --out models/v4_v1free.pkl --json-out results/train_v4_v1free.json

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_HERE))

from data_utils import load_holdout  # noqa: E402  (holdout real 151+20)
from text_utils import preprocess  # noqa: E402
from v4_common import (HELD_OUT_NRS, load_free_pairs, load_jsonl,  # noqa: E402
                       load_labels_v1, split_by_nr)


def _cap_per_nr(recs: List[Dict[str, Any]], cap: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Limita o nº de exemplos por NR (combate o desbalanceamento das expansões)."""
    if cap <= 0:
        return recs
    rng = random.Random(seed)
    by_nr: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in recs:
        by_nr[r["primary"]].append(r)
    out: List[Dict[str, Any]] = []
    for nr, group in by_nr.items():
        if len(group) > cap:
            group = rng.sample(group, cap)
        out += group
    return out


def build_dataset(names: List[str], augment_path: Optional[str],
                  samegen_path: Optional[str], free_cap: int = 0,
                  free_elec_only: bool = False, seed: int = 42
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Monta o treino a partir dos blocos pedidos. Retorna (recs, pesos-por-bloco).

    free_cap>0 limita exemplos-de-graça por NR; free_elec_only restringe às 9 do
    eletricista (o domínio que decide o recall). weights-por-bloco alimenta o
    sample_weight do LogReg (falas de campo v1 pesam mais que pares-de-graça).
    """
    from nr_taxonomy import ELECTRICIAN_NRS
    recs: List[Dict[str, Any]] = []
    if "v1" in names:
        recs += load_labels_v1()
    if "free" in names:
        free = load_free_pairs(exclude_gold=True)
        if free_elec_only:
            elec = set(ELECTRICIAN_NRS)
            free = [r for r in free if r["primary"] in elec]
        free = _cap_per_nr(free, free_cap, seed)
        recs += free
    if "augment" in names and augment_path:
        recs += [{**r, "kind": r.get("kind", "augment")} for r in load_jsonl(augment_path)]
    if "samegen" in names and samegen_path:
        # NÃO recomendado (é test set); só p/ diagnóstico de teto.
        recs += load_jsonl(samegen_path)
    return recs, {}


def fit_pipeline(train_texts: List[str], ytr: List[str],
                 sample_weight: Optional[np.ndarray] = None,
                 class_weight: Optional[str] = None) -> Tuple[Any, Any, Any, int]:
    """Ajusta o pipeline vencedor (char_wb 2-5 + word 1-2 + LogReg C=8)."""
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, preprocessor=preprocess)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                               sublinear_tf=True, preprocessor=preprocess)
    Xc = char_vec.fit_transform(train_texts)
    Xw = word_vec.fit_transform(train_texts)
    X = hstack([Xc, Xw]).tocsr()
    clf = LogisticRegression(max_iter=3000, C=8.0, class_weight=class_weight)
    clf.fit(X, ytr, sample_weight=sample_weight)
    return char_vec, word_vec, clf, X.shape[1]


def score(char_vec, word_vec, clf, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (probs, ordem topk desc). probs: (n, C); order: (n, C)."""
    Xc = char_vec.transform(texts)
    Xw = word_vec.transform(texts)
    X = hstack([Xc, Xw]).tocsr()
    probs = clf.predict_proba(X)
    order = np.argsort(-probs, axis=1)
    return probs, order


def topk_acc(char_vec, word_vec, clf, texts: List[str], golds: List[str],
             ks: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, float]:
    """Acurácia top-k (gold ∈ top-k)."""
    classes = clf.classes_
    _, order = score(char_vec, word_vec, clf, texts)
    n = len(golds)
    hits = {k: 0 for k in ks}
    for i, g in enumerate(golds):
        ranked = [classes[j] for j in order[i]]
        for k in ks:
            if g in ranked[:k]:
                hits[k] += 1
    return {f"top{k}": round(100 * hits[k] / n, 2) for k in ks}


def eval_named(char_vec, word_vec, clf, recs: List[Dict[str, Any]], label: str
               ) -> Dict[str, Any]:
    """Avalia acurácia top-k num conjunto {text, primary/gold}."""
    if not recs:
        return {"label": label, "n": 0}
    texts = [r["text"] for r in recs]
    golds = [r.get("gold", r.get("primary")) for r in recs]
    acc = topk_acc(char_vec, word_vec, clf, texts, golds)
    return {"label": label, "n": len(recs), **acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="v1", help="blocos: v1,free,augment (vírgula)")
    ap.add_argument("--augment", default=str(_HERE / "data" / "augment.jsonl"))
    ap.add_argument("--samegen", default=str(_HERE / "data" / "test_samegen.jsonl"))
    ap.add_argument("--diffgen", default=str(_HERE / "data" / "test_diffgen.jsonl"))
    ap.add_argument("--nr-split", action="store_true",
                    help="reserva HELD_OUT_NRS fora do treino (TRAVA 2)")
    ap.add_argument("--free-cap", type=int, default=0,
                    help="limite de pares-de-graça por NR (0=sem limite)")
    ap.add_argument("--free-elec-only", action="store_true",
                    help="usar pares-de-graça só das 9 NRs do eletricista")
    ap.add_argument("--free-weight", type=float, default=1.0,
                    help="sample_weight dos pares-de-graça (v1/augment=1.0)")
    ap.add_argument("--class-weight", default="", help="'balanced' ou vazio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(_HERE / "models" / "v4.pkl"))
    ap.add_argument("--json-out", default=str(_HERE / "results" / "train_v4.json"))
    args = ap.parse_args()

    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    aug = args.augment if Path(args.augment).exists() else None
    smg = args.samegen if Path(args.samegen).exists() else None
    train, _ = build_dataset(names, aug, smg, free_cap=args.free_cap,
                             free_elec_only=args.free_elec_only, seed=args.seed)

    held_recs: List[Dict[str, Any]] = []
    if args.nr_split:
        train, held_recs = split_by_nr(train, HELD_OUT_NRS)

    train_texts = [r["text"] for r in train]
    ytr = [r["primary"] for r in train]
    # sample_weight: pares-de-graça podem pesar menos que a fala de campo real (v1).
    sw = None
    if args.free_weight != 1.0:
        sw = np.array([args.free_weight if r.get("kind") == "free" else 1.0 for r in train])
    cw = args.class_weight or None
    t0 = time.time()
    char_vec, word_vec, clf, feat_dim = fit_pipeline(train_texts, ytr,
                                                     sample_weight=sw, class_weight=cw)
    fit_s = time.time() - t0

    # Holdout real (151+20) — a TRAVA 1 e a métrica âncora de acurácia.
    holdout = load_holdout()
    ho_recs = [{"text": h["text"], "gold": h["gold"]} for h in holdout]
    m_holdout = eval_named(char_vec, word_vec, clf, ho_recs, "holdout_171")

    # Holdout só nas NRs reservadas (TRAVA 2) — generalização a normas nunca vistas em treino.
    ho_heldnr = [r for r in ho_recs if r["gold"] in set(HELD_OUT_NRS)]
    m_heldnr = eval_named(char_vec, word_vec, clf, ho_heldnr, "holdout_held_nrs")

    # TRAVA 3: mesmo-gerador (otimista) vs gerador-diferente (honesto).
    m_same = m_diff = {"label": "test_samegen", "n": 0}
    if Path(args.samegen).exists() and "samegen" not in names:
        m_same = eval_named(char_vec, word_vec, clf, load_jsonl(args.samegen), "test_samegen")
    if Path(args.diffgen).exists():
        m_diff = eval_named(char_vec, word_vec, clf, load_jsonl(args.diffgen), "test_diffgen")

    # Tamanho da mascarada = otimista − honesta (em top-1 e top-2).
    mask = {}
    if m_same.get("n") and m_diff.get("n"):
        for k in ("top1", "top2"):
            mask[k] = round(m_same[k] - m_diff[k], 2)

    # Persiste o pipeline no MESMO formato consumido por export_kotlin.py.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"char": char_vec, "word": word_vec, "clf": clf, "name": "logreg_v4"}, f)

    out = {
        "datasets": names, "nr_split": args.nr_split, "held_out_nrs": HELD_OUT_NRS,
        "free_cap": args.free_cap, "free_elec_only": args.free_elec_only,
        "free_weight": args.free_weight, "class_weight": cw,
        "n_train": len(train_texts), "feature_dim": feat_dim, "fit_s": round(fit_s, 2),
        "holdout_171": m_holdout, "holdout_held_nrs": m_heldnr,
        "test_samegen": m_same, "test_diffgen": m_diff, "mask_size_pp": mask,
        "model_path": str(args.out),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"  TREINO v4  datasets={names}  nr_split={args.nr_split}  n_train={len(train_texts)}")
    print("=" * 78)
    def row(m):
        if not m.get("n"):
            return f"  {m['label']:<22} (ausente)"
        return (f"  {m['label']:<22} n={m['n']:<5} "
                f"top1={m.get('top1',0):>5.1f}%  top2={m.get('top2',0):>5.1f}%  top3={m.get('top3',0):>5.1f}%")
    print(row(m_holdout))
    print(row(m_heldnr))
    print(row(m_same))
    print(row(m_diff))
    if mask:
        print(f"  >> TAMANHO DA MASCARADA (otimista−honesta): top1={mask['top1']:+.1f}pp  top2={mask['top2']:+.1f}pp")
    print("=" * 78)
    print(f"Modelo salvo: {args.out}\nMétricas: {args.json_out}")


if __name__ == "__main__":
    main()
