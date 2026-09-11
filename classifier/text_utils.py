#!/usr/bin/env python3
"""
text_utils.py — Pré-processamento de texto compartilhado (treino e inferência).

Mantido em módulo próprio para que os TfidfVectorizer serializados (pickle) consigam
resolver `preprocess` ao serem recarregados fora do processo de treino (hybrid.py, app).
A sanitização espelha o Fts5Retriever.kt (remoção de acentos + minúsculas).
"""

from __future__ import annotations

import unicodedata


def strip_accents(text: str) -> str:
    """Remove acentos (paridade com a sanitização do Fts5Retriever.kt)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def preprocess(text: str) -> str:
    """Normaliza a fala para featurização TF-IDF (acentos + caixa baixa)."""
    return strip_accents(text).lower().strip()
