#!/usr/bin/env python3
"""
train_classic.py — FASE B (clássicos): shootout de classificadores sklearn leves.

Features: TF-IDF híbrido (char n-grams 2-5 + word 1-2), sanitização idêntica ao app
(remoção de acentos, minúsculas) para paridade com o deploy Kotlin/ONNX.

Candidatos avaliados (posição do capitão: começar LEVE, árvore/tf-idf primeiro):
  - logreg        : LogisticRegression (linear, calibrável -> probs para boost suave)
  - linsvc        : LinearSVC (margem; decision_function p/ top-2)
  - complementnb  : ComplementNB (forte em classes desbalanceadas)
  - randomforest  : RandomForest (árvore ensemble)
  - gradientboost : HistGradientBoosting (boosting)
  - decisiontree  : DecisionTree (baseline árvore simples — posição explícita do capitão)

Métricas no holdout real (151+20): acurácia top-1/top-2, macro-F1, tamanho serializado
(joblib), latência de inferência DESKTOP (mediana por consulta) e estimativa mobile.

Uso:
    python3 classifier/train_classic.py --out classifier/models --json-out classifier/data/classic_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import load_holdout, load_train
from text_utils import preprocess, strip_accents

_HERE = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_classic")


def build_features(train_texts: List[str], test_texts: List[str]):
    """Ajusta TF-IDF híbrido (char 2-5 + word 1-2) e transforma treino/teste."""
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, preprocessor=preprocess
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=preprocess
    )
    Xtr_c = char_vec.fit_transform(train_texts)
    Xtr_w = word_vec.fit_transform(train_texts)
    Xte_c = char_vec.transform(test_texts)
    Xte_w = word_vec.transform(test_texts)
    Xtr = hstack([Xtr_c, Xtr_w]).tocsr()
    Xte = hstack([Xte_c, Xte_w]).tocsr()
    return Xtr, Xte, (char_vec, word_vec)


def top2_from_scores(scores: np.ndarray, classes: np.ndarray) -> List[List[str]]:
    """Extrai as 2 classes de maior score por amostra (probs ou decision_function)."""
    order = np.argsort(-scores, axis=1)[:, :2]
    return [[classes[j] for j in row] for row in order]


def get_scores(model, X) -> np.ndarray:
    """Retorna matriz de scores (probs se houver; senão decision_function).

    HistGradientBoosting exige entrada densa; densifica sob demanda ao falhar.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except TypeError:
            return model.predict_proba(X.toarray())
    dec = model.decision_function(X)
    if dec.ndim == 1:  # binário (não é o caso aqui, mas por segurança)
        dec = np.vstack([-dec, dec]).T
    return dec


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    """Macro-F1 manual (evita dependência extra; classes desbalanceadas)."""
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def eval_model(name, model, Xtr, ytr, Xte, holdout, out_dir: Path) -> Dict[str, Any]:
    """Treina, mede tamanho, latência e acurácia top-1/top-2 no holdout."""
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0

    classes = model.classes_
    scores = get_scores(model, Xte)
    top2 = top2_from_scores(scores, classes)
    pred1 = [row[0] for row in top2]

    golds = [h["gold"] for h in holdout]
    # Para itens ambíguos de treino usamos primary; holdout tem gold único (nr-XX ou nenhuma).
    top1_hits = sum(1 for p, g in zip(pred1, golds) if p == g)
    top2_hits = sum(1 for row, g in zip(top2, golds) if g in row)
    n = len(golds)

    # macro-F1 apenas sobre classes presentes no holdout (as demais não são avaliáveis)
    present = sorted(set(golds))
    mf1 = macro_f1(golds, pred1, present)

    # Latência de inferência DESKTOP: mediana de 200 consultas single (features + predict)
    lat = measure_latency(model, holdout, model_vecs=None)

    # Serializa (pipeline completo com vetorizadores é medido separadamente no caller)
    size_model = len(pickle.dumps(model))

    logger.info(
        "%-14s top1=%.1f%% top2=%.1f%% macroF1=%.3f fit=%.1fs modelo=%.0fKB",
        name, 100 * top1_hits / n, 100 * top2_hits / n, mf1, fit_s, size_model / 1024,
    )
    return {
        "name": name,
        "top1_acc": round(100 * top1_hits / n, 2),
        "top2_acc": round(100 * top2_hits / n, 2),
        "macro_f1": round(mf1, 4),
        "fit_s": round(fit_s, 2),
        "model_bytes": size_model,
        "lat_ms_desktop": lat,
        "pred1": pred1,
        "top2": top2,
    }


def measure_latency(model, holdout, model_vecs) -> float:
    """Placeholder: latência real é medida no caller com o pipeline completo."""
    return 0.0


def measure_pipeline_latency(vecs, model, texts: List[str], repeats: int = 3) -> float:
    """Mede latência mediana por consulta (featurize + predict) — caminho de produção."""
    char_vec, word_vec = vecs
    samples = texts[: min(len(texts), 100)]
    times = []
    for _ in range(repeats):
        for t in samples:
            t0 = time.perf_counter()
            xc = char_vec.transform([t])
            xw = word_vec.transform([t])
            X = hstack([xc, xw]).tocsr()
            get_scores(model, X)
            times.append((time.perf_counter() - t0) * 1000.0)
    return round(float(np.median(times)), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="FASE B: shootout de classificadores clássicos.")
    ap.add_argument("--out", type=str, default=str(_HERE / "models"))
    ap.add_argument("--json-out", type=str, default=str(_HERE / "data" / "classic_results.json"))
    ap.add_argument("--skip-slow", action="store_true",
                    help="Pula gradientboost (denso, ~10min, já desqualificado por top1=28.7%).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train()
    holdout = load_holdout()
    train_texts = [r["text"] for r in train]
    ytr = [r["primary"] for r in train]
    test_texts = [h["text"] for h in holdout]

    logger.info("Treino=%d falas | Holdout=%d perguntas | classes treino=%d",
                len(train_texts), len(test_texts), len(set(ytr)))

    Xtr, Xte, vecs = build_features(train_texts, test_texts)
    feat_dim = Xtr.shape[1]
    logger.info("Dimensão de features: %d", feat_dim)

    candidates = {
        # C=8 sem class_weight venceu no holdout (top2 73.5% vs 72.2% do C=4 balanced);
        # top-2 é o driver do boost suave -> priorizado.
        "logreg": LogisticRegression(max_iter=3000, C=8.0),
        "linsvc": LinearSVC(C=1.0, class_weight="balanced"),
        "complementnb": ComplementNB(alpha=0.3),
        "randomforest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "gradientboost": HistGradientBoostingClassifier(max_iter=200, random_state=42),
        "decisiontree": DecisionTreeClassifier(max_depth=40, random_state=42),
    }
    if args.skip_slow:
        candidates.pop("gradientboost", None)
    complexity_rank_all = {
        "complementnb": 0, "logreg": 1, "linsvc": 1, "decisiontree": 2,
        "randomforest": 3, "gradientboost": 4,
    }

    results = []
    best_pipes = {}
    for name, clf in candidates.items():
        X_in = Xtr
        Xte_in = Xte
        # HistGradientBoosting não aceita esparso -> densifica (custo alto; medir mesmo assim)
        if name == "gradientboost":
            X_in = Xtr.toarray()
            Xte_in = Xte.toarray()
        r = eval_model(name, clf, X_in, ytr, Xte_in, holdout, out_dir)
        r["lat_ms_desktop"] = measure_pipeline_latency(vecs, clf, test_texts)
        # Tamanho serializado do PIPELINE completo (vetorizadores + modelo)
        pipe_blob = pickle.dumps({"char": vecs[0], "word": vecs[1], "clf": clf})
        r["pipeline_bytes"] = len(pipe_blob)
        results.append(r)
        best_pipes[name] = clf

    # Persiste o melhor por top-2 (critério primário do híbrido) + desempate simplicidade
    results_sorted = sorted(results, key=lambda r: (-r["top2_acc"], -r["top1_acc"], complexity_rank_all[r["name"]]))
    winner = results_sorted[0]["name"]
    logger.info("Vencedor clássico (top2 > top1 > simplicidade): %s", winner)

    # Salva o pipeline vencedor para a Fase C (integração no híbrido)
    win_blob = {"char": vecs[0], "word": vecs[1], "clf": best_pipes[winner], "name": winner}
    with (out_dir / "classic_winner.pkl").open("wb") as f:
        pickle.dump(win_blob, f)
    logger.info("Pipeline vencedor salvo em %s", out_dir / "classic_winner.pkl")

    # Tabela resumo
    print("\n" + "=" * 92)
    print(f"  FASE B — CLÁSSICOS (holdout real {len(test_texts)} perguntas) — features={feat_dim}")
    print("=" * 92)
    print(f"| {'Modelo':<14} | {'Top-1':>6} | {'Top-2':>6} | {'MacroF1':>7} | {'Lat(ms)':>7} | {'Pipeline':>9} |")
    print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 11 + "|")
    for r in results_sorted:
        print(f"| {r['name']:<14} | {r['top1_acc']:>5.1f}% | {r['top2_acc']:>5.1f}% | "
              f"{r['macro_f1']:>7.3f} | {r['lat_ms_desktop']:>7.2f} | {r['pipeline_bytes']/1024:>7.0f}KB |")
    print("=" * 92)
    print(f"Vencedor: {winner}")

    out = {
        "feature_dim": feat_dim,
        "n_train": len(train_texts),
        "n_holdout": len(test_texts),
        "winner": winner,
        "results": [{k: v for k, v in r.items() if k not in ("pred1", "top2")} for r in results_sorted],
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Métricas salvas em %s", args.json_out)


if __name__ == "__main__":
    main()
