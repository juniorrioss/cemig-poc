#!/usr/bin/env python3
"""
trim.py — TRIM determinístico e barato de chunk (sem LLM), p/ a bancada JANELA×TRECHOS.

Ordem 1 do capitão: chunks têm ~370 tok; 5 chunks estouram o orçamento atual (~1000 tok
de contexto). Uma alternativa a "menos trechos" é "trechos mais curtos": recortar cada
chunk para a JANELA RELEVANTE em torno do melhor casamento com a query, cabendo mais
trechos na janela sem inflar o prompt.

Algoritmo (determinístico, O(n), sem modelo):
  1. Tokeniza o chunk em sentenças (por pontuação e por itens normativos "10.2.8.1").
  2. Pontua cada sentença por sobreposição de termos com a query (BM25-lite: soma de
     idf-proxy por termo presente; idf-proxy = 1/log(1+df_no_chunk), favorece termos raros).
  3. Escolhe a sentença de maior score como âncora e cresce uma janela contígua ao redor
     dela até `max_chars` (adicionando alternadamente a vizinha de maior score).
  4. Preserva SEMPRE o cabeçalho normativo (doc/section/título) se estiver na 1ª sentença,
     para não perder a citação. Concatena com " (...) " quando corta o miolo.

Se o chunk já couber em `max_chars`, devolve o texto original (trim é no-op).

Barato: só contagem de termos e ordenação. Determinístico: sem sampling, sem rede.
Comentários PT-BR, código em inglês. Procedência: task poc-ctx-topk.
"""

from __future__ import annotations

import math
import re
from typing import List

# Stopwords PT-BR mínimas p/ não pontuar preposição/artigo (alinhado ao APP_STOPWORDS).
_STOP = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "que", "em", "no", "na", "nos", "nas",
    "um", "uma", "para", "por", "com", "se", "os", "as", "ao", "aos", "à", "às", "ou",
    "the", "is", "and", "of", "to", "in", "é", "ser", "está", "são", "como", "mais",
    "pra", "pro", "tá", "aqui", "lá", "isso", "esse", "essa", "meu", "minha", "ele", "ela",
}

_WORD_RE = re.compile(r"[0-9a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]+")
# Divisor de sentenças: quebra em ponto final/;/: seguidos de espaço, e antes de itens
# normativos do tipo "10.2.8.1 " no meio do texto.
_ITEM_RE = re.compile(r"(?<=\s)(\d{1,2}(?:\.\d+){1,4})\s")


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _split_sentences(text: str) -> List[str]:
    """Divide em sentenças por pontuação e por marcadores de item normativo."""
    # Insere quebra antes de cada item normativo embutido (ex.: "... coletiva 10.2.8.2 ...").
    marked = _ITEM_RE.sub(lambda m: "\n" + m.group(1) + " ", " " + text)
    parts: List[str] = []
    for block in marked.split("\n"):
        # Quebra também em ponto/;/: de fim de frase.
        for s in re.split(r"(?<=[.;:!?])\s+", block.strip()):
            s = s.strip()
            if s:
                parts.append(s)
    return parts or [text.strip()]


def _score_sentence(sent_tokens: List[str], q_terms: set[str], idf: dict[str, float]) -> float:
    """BM25-lite: soma de idf-proxy dos termos da query presentes na sentença (sem repetir)."""
    seen: set[str] = set()
    score = 0.0
    for t in sent_tokens:
        if t in q_terms and t not in seen and t not in _STOP:
            score += idf.get(t, 1.0)
            seen.add(t)
    return score


def trim_chunk(text: str, query: str, max_chars: int) -> str:
    """Recorta `text` para ~max_chars em torno do melhor casamento com `query`.

    Determinístico e sem LLM. No-op se o texto já couber. Preserva o cabeçalho (1ª frase)
    quando corta o miolo, para não perder a citação normativa (doc/section/título).
    """
    if len(text) <= max_chars:
        return text

    sents = _split_sentences(text)
    if len(sents) <= 1:
        return text[:max_chars].rsplit(" ", 1)[0] + " (...)"

    q_terms = {t for t in _tokens(query) if t not in _STOP and len(t) > 2}
    # idf-proxy por df dentro do próprio chunk: termos raros no chunk pesam mais.
    df: dict[str, int] = {}
    sent_toks = [_tokens(s) for s in sents]
    for toks in sent_toks:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n_sent = len(sents)
    idf = {t: math.log(1 + n_sent / (1 + c)) + 1.0 for t, c in df.items()}

    scores = [_score_sentence(sent_toks[i], q_terms, idf) for i in range(n_sent)]
    anchor = max(range(n_sent), key=lambda i: scores[i])

    # Cresce janela contígua ao redor da âncora, alternando p/ a vizinha de maior score.
    lo = hi = anchor
    total = len(sents[anchor])
    while total < max_chars:
        left_ok = lo > 0
        right_ok = hi < n_sent - 1
        if not left_ok and not right_ok:
            break
        left_s = scores[lo - 1] if left_ok else -1.0
        right_s = scores[hi + 1] if right_ok else -1.0
        if right_ok and (right_s >= left_s or not left_ok):
            if total + len(sents[hi + 1]) + 1 > max_chars:
                break
            hi += 1
            total += len(sents[hi]) + 1
        elif left_ok:
            if total + len(sents[lo - 1]) + 1 > max_chars:
                break
            lo -= 1
            total += len(sents[lo]) + 1
        else:
            break

    window = " ".join(sents[lo:hi + 1])
    # Preserva o cabeçalho normativo (1ª sentença) se não estiver na janela.
    if lo > 0:
        head = sents[0]
        if len(head) + len(window) + 8 <= max_chars + len(head):
            window = head + " (...) " + window
        else:
            window = head + " (...) " + window  # cabeçalho é barato e crítico p/ citação
    prefix = "(...) " if lo > 0 and not window.startswith(sents[0]) else ""
    suffix = " (...)" if hi < n_sent - 1 else ""
    return prefix + window + suffix
