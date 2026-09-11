#!/usr/bin/env python3
"""
export_kotlin.py — FASE C (deploy): exporta o classificador vencedor p/ Kotlin puro.

skl2onnx NÃO converte char n-grams (só tokenizer='word'), e o char_wb é o que dá
robustez a ruído de ASR (+2.6 p.p. top-2 sob typos). Portanto o caminho de deploy é
Kotlin PURO (sem runtime ONNX): reimplementamos TF-IDF (char_wb 2-5 + word 1-2,
sublinear_tf, norma L2) + produto interno linear + softmax.

Este script:
  1. Serializa vocabulário (ngram->col), IDF e a matriz de coeficientes/intercepto num
     binário compacto (`nr_classifier.bin`) + metadados JSON (`nr_classifier_meta.json`).
  2. Fornece uma reimplementação Python `predict_pure()` idêntica ao algoritmo Kotlin
     e VALIDA a paridade com o sklearn no holdout real (deve bater top-1/top-2).

O formato binário é little-endian (NRC2, auto-contido — sem meta JSON em runtime):
  [magic 4B 'NRC2'][n_classes u16][n_feat u32][n_char u32]
  classes: n_classes x [len u16][utf8]  (rótulos nr-XX na ordem das colunas de coef)
  vocab:   n_feat  x [len u16][ngram utf8][idf f32]  (char em col [0,n_char), depois word)
  coef:    n_classes * n_feat f32 (linha-maior)
  intercept: n_classes f32

Uso:
    python3 classifier/export_kotlin.py
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data_utils import load_holdout
from text_utils import preprocess

_HERE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export_kotlin")


def char_wb_ngrams(text: str, nmin: int, nmax: int) -> List[str]:
    """Reproduz FIELMENTE o analyzer='char_wb' do sklearn (_char_wb_ngrams).

    Para cada palavra padded ' '+w+' ' e cada n: adiciona w[0:n], desliza; se a palavra
    for mais curta que n (offset permanece 0), conta a palavra inteira UMA vez e para
    de aumentar n (break) — evita duplicar a palavra curta em n maiores.
    """
    ngrams: List[str] = []
    for token in text.split():
        w = " " + token + " "
        w_len = len(w)
        for n in range(nmin, nmax + 1):
            offset = 0
            ngrams.append(w[offset : offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams.append(w[offset : offset + n])
            if offset == 0:  # palavra curta (w_len <= n): conta 1x e para
                break
    return ngrams


def word_ngrams(text: str, nmin: int, nmax: int) -> List[str]:
    """Reproduz analyzer='word' do sklearn com o token_pattern padrão (\\b\\w\\w+\\b)."""
    import re
    tokens = re.findall(r"(?u)\b\w\w+\b", text)
    ngrams: List[str] = []
    for n in range(nmin, nmax + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def build_vocab_maps(cv, wv) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray, int, int]:
    """Extrai vocabulários e IDF concatenados (char primeiro, depois word)."""
    char_vocab = cv.vocabulary_       # ngram -> local idx
    word_vocab = wv.vocabulary_
    n_char = len(char_vocab)
    n_word = len(word_vocab)
    idf = np.concatenate([cv.idf_, wv.idf_]).astype(np.float32)
    # coluna global: char em [0, n_char), word em [n_char, n_char+n_word)
    char_map = {ng: idx for ng, idx in char_vocab.items()}
    word_map = {ng: (idx + n_char) for ng, idx in word_vocab.items()}
    return char_map, word_map, idf, n_char, n_word


def featurize_pure(text: str, char_map, word_map, idf: np.ndarray, n_char: int) -> np.ndarray:
    """TF-IDF sublinear + L2, reimplementado (paridade com o algoritmo Kotlin).

    CRÍTICO: o sklearn L2-normaliza os blocos char e word SEPARADAMENTE (cada
    TfidfVectorizer aplica norm='l2') e só depois o hstack concatena. Portanto
    normalizamos o bloco char [0:n_char) e o bloco word [n_char:) de forma independente.
    """
    n_feat = idf.shape[0]
    vec = np.zeros(n_feat, dtype=np.float64)
    for ng in char_wb_ngrams(text, 2, 5):
        col = char_map.get(ng)
        if col is not None:
            vec[col] += 1.0
    for ng in word_ngrams(text, 1, 2):
        col = word_map.get(ng)
        if col is not None:
            vec[col] += 1.0
    nz = vec > 0
    vec[nz] = 1.0 + np.log(vec[nz])  # sublinear_tf
    vec *= idf                        # tf-idf
    # L2 por bloco (char e word), replicando os dois vetorizadores + hstack
    nc = np.linalg.norm(vec[:n_char])
    if nc > 0:
        vec[:n_char] /= nc
    nw = np.linalg.norm(vec[n_char:])
    if nw > 0:
        vec[n_char:] /= nw
    return vec


def main() -> None:
    blob = pickle.load((_HERE / "models" / "classic_winner.pkl").open("rb"))
    cv, wv, clf = blob["char"], blob["word"], blob["clf"]
    classes = list(clf.classes_)
    n_classes = len(classes)

    char_map, word_map, idf, n_char, n_word = build_vocab_maps(cv, wv)
    n_feat = n_char + n_word
    logger.info("Features: char=%d word=%d total=%d | classes=%d", n_char, n_word, n_feat, n_classes)

    coef = clf.coef_.astype(np.float32)          # (n_classes, n_feat)
    intercept = clf.intercept_.astype(np.float32)  # (n_classes,)

    # --- Validação de paridade: featurize_pure + coef vs sklearn.predict_proba ---
    holdout = load_holdout()
    ho_qa = [h for h in holdout if h["source"] == "qa_v2"]
    golds = [h["gold"] for h in ho_qa]

    # sklearn de referência (usa os vetorizadores originais com preprocessor)
    from scipy.sparse import hstack
    texts_raw = [h["text"] for h in ho_qa]
    Xc = cv.transform(texts_raw)
    Xw = wv.transform(texts_raw)
    X = hstack([Xc, Xw]).tocsr()
    P_sk = clf.predict_proba(X)

    # pure
    P_pure = np.zeros((len(ho_qa), n_classes))
    for i, h in enumerate(ho_qa):
        v = featurize_pure(preprocess(h["text"]), char_map, word_map, idf, n_char)
        logits = coef @ v + intercept
        # softmax
        m = logits.max()
        e = np.exp(logits - m)
        P_pure[i] = e / e.sum()

    max_diff = float(np.max(np.abs(P_sk - P_pure)))

    def acc(P):
        order = np.argsort(-P, axis=1)
        t1 = 100 * sum(1 for i, g in enumerate(golds) if classes[order[i, 0]] == g) / len(golds)
        t2 = 100 * sum(1 for i, g in enumerate(golds) if g in [classes[order[i, j]] for j in range(2)]) / len(golds)
        return t1, t2

    t1s, t2s = acc(P_sk)
    t1p, t2p = acc(P_pure)
    logger.info("Paridade sklearn vs pure: max |Δprob| = %.2e", max_diff)
    logger.info("sklearn top1=%.1f%% top2=%.1f%% | pure top1=%.1f%% top2=%.1f%%", t1s, t2s, t1p, t2p)

    # --- Serialização binária compacta ---
    out_bin = _HERE / "models" / "nr_classifier.bin"
    with out_bin.open("wb") as f:
        f.write(b"NRC2")
        f.write(struct.pack("<H", n_classes))
        f.write(struct.pack("<I", n_feat))
        f.write(struct.pack("<I", n_char))
        # rótulos de classe (auto-contido)
        for c in classes:
            enc = c.encode("utf-8")
            f.write(struct.pack("<H", len(enc)))
            f.write(enc)
        # vocab: char primeiro (col 0..n_char-1) depois word
        inv = [None] * n_feat
        for ng, col in char_map.items():
            inv[col] = ng
        for ng, col in word_map.items():
            inv[col] = ng
        for col in range(n_feat):
            ng = inv[col]
            enc = ng.encode("utf-8")
            f.write(struct.pack("<H", len(enc)))  # ngram pode ter multibyte utf-8
            f.write(enc)
            f.write(struct.pack("<f", float(idf[col])))
        f.write(coef.tobytes(order="C"))
        f.write(intercept.tobytes())
    size_mb = out_bin.stat().st_size / 1e6
    logger.info("Binário Kotlin salvo: %s (%.2f MB)", out_bin, size_mb)

    meta = {
        "format": "NRC2",
        "n_classes": n_classes,
        "n_features": n_feat,
        "n_char": n_char,
        "n_word": n_word,
        "classes": classes,
        "char_ngram_range": [2, 5],
        "word_ngram_range": [1, 2],
        "analyzer_char": "char_wb",
        "sublinear_tf": True,
        "norm": "l2",
        "bin_mb": round(size_mb, 3),
        "parity_max_prob_diff": max_diff,
        "top1_sklearn": round(t1s, 2),
        "top2_sklearn": round(t2s, 2),
        "top1_pure": round(t1p, 2),
        "top2_pure": round(t2p, 2),
        "preprocess": "strip_accents(NFKD) + lowercase + trim (idêntico a text_utils.preprocess)",
    }
    (_HERE / "models" / "nr_classifier_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (_HERE / "data" / "kotlin_export.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metadados salvos em models/nr_classifier_meta.json")


if __name__ == "__main__":
    main()
