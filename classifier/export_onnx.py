#!/usr/bin/env python3
"""
export_onnx.py — FASE C (deploy): TENTATIVA de export ONNX (NÃO usada em produção).

ATENÇÃO: skl2onnx NÃO converte char n-grams (só tokenizer='word'), e o char_wb é o que
dá robustez a ruído de ASR (+2.6 p.p. top-2 sob typos). Este script FALHA com
"NotImplementedError: CountVectorizer ... only tokenizer='word' is fully supported".
Por isso o deploy real é Kotlin puro (export_kotlin.py / NrClassifier.kt). Mantido como
registro da investigação do caminho ONNX; não reintroduzir sem resolver o char n-gram.

Reconstrói o pipeline vencedor como um Pipeline sklearn único
(FeatureUnion[char TF-IDF, word TF-IDF] -> LogisticRegression) e o serializa em ONNX,
que roda no Android via ONNX Runtime Mobile SEM Python. A entrada é uma string (a fala
crua); a saída são os rótulos e as probabilidades por classe (para o boost suave/gated).

Valida a PARIDADE numérica entre o pipeline sklearn e o grafo ONNX no holdout real, e
mede a latência de inferência ONNX no desktop (proxy do caminho mobile).

Uso:
    python3 classifier/export_onnx.py --out classifier/models/nr_classifier.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from data_utils import load_holdout, load_train
from text_utils import preprocess

_HERE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export_onnx")


def main() -> None:
    ap = argparse.ArgumentParser(description="FASE C: exporta o classificador para ONNX.")
    ap.add_argument("--out", type=str, default=str(_HERE / "models" / "nr_classifier.onnx"))
    ap.add_argument("--labels-out", type=str, default=str(_HERE / "models" / "nr_classes.json"))
    args = ap.parse_args()

    train = load_train()
    holdout = load_holdout()
    Xtr = [r["text"] for r in train]
    ytr = [r["primary"] for r in train]

    # skl2onnx NÃO serializa preprocessor Python custom; por isso sanitizamos ANTES
    # (strip_accents+lower) e treinamos com lowercase=False. O app Android deve aplicar
    # a mesma sanitização (text_utils.preprocess) na fala antes de invocar o grafo ONNX.
    logger.info("Treinando Pipeline único p/ ONNX sobre texto pré-sanitizado (%d falas)...", len(Xtr))
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=False)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
    union = FeatureUnion([("char", char_vec), ("word", word_vec)])
    clf = LogisticRegression(max_iter=3000, C=8.0)
    pipe2 = Pipeline([("features", union), ("clf", clf)])
    Xtr_s = [preprocess(t) for t in Xtr]
    pipe2.fit(Xtr_s, ytr)

    onnx_model = convert_sklearn(
        pipe2,
        "nr_classifier",
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(clf): {"zipmap": False}},
        target_opset=17,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    size_kb = out_path.stat().st_size / 1024
    logger.info("ONNX salvo: %s (%.0f KB)", out_path, size_kb)

    # Salva a ordem canônica das classes (índice de saída ONNX -> nr-XX)
    classes = list(pipe2.named_steps["clf"].classes_)
    Path(args.labels_out).write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    logger.info("Classes salvas: %s (%d)", args.labels_out, len(classes))

    # --- Validação de paridade sklearn vs ONNX no holdout ---
    import onnxruntime as ort

    ho_texts = [preprocess(h["text"]) for h in holdout]
    golds = [h["gold"] for h in holdout]

    P_sk = pipe2.predict_proba(ho_texts)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    arr = np.array(ho_texts, dtype=object).reshape(-1, 1)
    outputs = sess.run(None, {in_name: arr})
    # saída [labels, probabilities] com zipmap=False
    P_onnx = np.array(outputs[1])

    max_diff = float(np.max(np.abs(P_sk - P_onnx)))
    logger.info("Paridade sklearn vs ONNX: max |Δprob| = %.2e", max_diff)

    def top2_acc(P, C):
        order = np.argsort(-P, axis=1)[:, :2]
        return 100 * sum(1 for i, g in enumerate(golds) if g in [C[j] for j in order[i]]) / len(golds)

    C = np.array(classes)
    logger.info("Top-2 acc: sklearn=%.1f%% | ONNX=%.1f%%", top2_acc(P_sk, C), top2_acc(P_onnx, C))

    # Latência ONNX (mediana por consulta) — proxy do caminho mobile
    lat = []
    for t in ho_texts[:100]:
        a = np.array([[t]], dtype=object)
        t0 = time.perf_counter()
        sess.run(None, {in_name: a})
        lat.append((time.perf_counter() - t0) * 1000)
    import statistics
    logger.info("Latência ONNX desktop: mediana %.2f ms | p90 %.2f ms",
                statistics.median(lat), statistics.quantiles(lat, n=10)[-1])

    meta = {
        "onnx_kb": round(size_kb, 1),
        "max_prob_diff": max_diff,
        "top2_sklearn": round(top2_acc(P_sk, C), 2),
        "top2_onnx": round(top2_acc(P_onnx, C), 2),
        "lat_ms_median": round(statistics.median(lat), 3),
        "n_classes": len(classes),
        "note": "app deve pré-sanitizar (strip_accents+lower) idêntico a text_utils.preprocess",
    }
    (_HERE / "data" / "onnx_export.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Meta export salva em classifier/data/onnx_export.json")


if __name__ == "__main__":
    main()
