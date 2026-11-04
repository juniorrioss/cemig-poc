#!/usr/bin/env python3
"""
compress.py — Frente 3: COMPRESSÃO de contexto por relevância (estilo SCR do MobileRAG+).

Hipótese do capitão: o LFM é sensível ao topK/tamanho do contexto; entregar MENOS tokens,
mais densos, deve manter ou melhorar a resposta.

Desenho (sem LLM, "modelinho de relevância" = o MESMO EmbeddingGemma-300M do app):
  1. SEGMENTA cada chunk em UNIDADES normativas: quebra antes de cada código de item
     (10.2.8.1), alínea (a), b)...) ou fronteira de frase. Cada unidade carrega o
     CÓDIGO DO ITEM a que pertence (o último código visto antes dela, ou o do cabeçalho)
     -> a frase selecionada NUNCA perde o id citável (lição do trim.py, que cortava o
     miolo e piorou a citação).
  2. JANELA de 3 unidades (anterior+atual+próxima) — igual ao SCR (granite-107m, janela
     de 3 frases) — embedada com o prompt de documento oficial do Gemma. As janelas não
     dependem da query: no app viram um índice pré-computado (DVEC), e a compressão no
     aparelho custa só produtos internos (zero inferência extra).
  3. score(unidade) = cos(query, janela). Seleção GULOSA global pelo score até o
     orçamento de tokens (tokenizer do LFM3); render na ORDEM DO DOCUMENTO por chunk,
     com cabeçalho de fonte do chunk e "[…]" nas lacunas; unidades que começam no meio
     de um item recebem o prefixo "(item X)".

Saídas auxiliares: spans de FONTE de cada unidade (via tokmap do align.py) -> evidence
recall pós-compressão na régua bench.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common_f5 as C  # noqa: E402
from align import _src_index, align_chunk  # noqa: E402

UNITS_JSON = C.CACHE / "units.json"
UNITS_NPZ = C.CACHE / "units_win3.npz"

# código de item normativo: 10.2.8.1 / 12.1 / 1.5.6 / Anexo-style "4.40"
ITEM_RE = re.compile(r"(?<![\w.,])(\d{1,2}(?:\.\d{1,3}){1,5})(?![\w,]|\.\d)")
_SPLIT_RE = re.compile(
    r"(?<=[.;:])\s+(?=(?:\d{1,2}(?:\.\d{1,3}){1,5}\s)|(?:[a-z]\)\s)|(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ]))"
    r"|\s+(?=\d{1,2}(?:\.\d{1,3}){1,5}\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ])"
    r"|(?<=;)\s+(?=[a-z]\)\s)"
    r"|\s+(?=\|\s*\d)")   # linhas de tabela
MAX_UNIT = 600            # chars; unidades maiores são quebradas em frases/vírgulas


def _hard_split(s: str, a: int, b: int) -> List[Tuple[int, int]]:
    out, cur = [], a
    while b - cur > MAX_UNIT:
        cut = s.rfind(" ", cur + MAX_UNIT // 2, cur + MAX_UNIT)
        cut = cut if cut > cur else cur + MAX_UNIT
        out.append((cur, cut))
        cur = cut
    out.append((cur, b))
    return out


def segment(body: str) -> List[Tuple[int, int]]:
    """Unidades (ini, fim) no corpo do chunk."""
    cuts = [0] + [m.end() for m in _SPLIT_RE.finditer(body)] + [len(body)]
    units = []
    for a, b in zip(cuts, cuts[1:]):
        while a < b and body[a].isspace():
            a += 1
        e = b
        while e > a and body[e - 1].isspace():
            e -= 1
        if e - a < 2:
            continue
        units.extend(_hard_split(body, a, e))
    # funde unidades muito curtas ("a)", "Comentário") na seguinte
    merged: List[List[int]] = []
    for a, b in units:
        if merged and (merged[-1][1] - merged[-1][0]) < 25:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _header_item(chunk: Dict[str, Any]) -> str:
    m = ITEM_RE.search(chunk.get("section") or "")
    return m.group(1) if m else (chunk.get("section") or "")


def build_units() -> None:
    """Segmenta os 2202 chunks, atribui item, mapeia spans de fonte e embeda as janelas."""
    S = C.load_sources()
    chunks = C.load_chunks()
    idx_cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    win_texts: List[str] = []
    for cid in sorted(chunks):
        ch = chunks[cid]
        prefix, body, off = C.chunk_body(ch["text"])
        f = S["norm_file_of_doc"][ch["doc"]]
        if f not in idx_cache:
            idx_cache[f] = _src_index(S["by_file"][f])
        _, _, _, tokmap = align_chunk(body, idx_cache[f], want_map=True)
        segs = segment(body)
        item = _header_item(ch)
        units = []
        for a, b in segs:
            text = body[a:b]
            m = ITEM_RE.match(text)
            starts_item = bool(m)
            if m:
                item = m.group(1)
            # spans de fonte dos tokens casados dentro da unidade
            src = sorted((s2, e2) for (s1, e1, s2, e2) in tokmap if s1 >= a and e1 <= b)
            spans: List[List[int]] = []
            for s2, e2 in src:
                if spans and s2 - spans[-1][1] <= 40:
                    spans[-1][1] = max(spans[-1][1], e2)
                else:
                    spans.append([s2, e2])
            units.append({"a": a, "b": b, "item": item, "starts_item": starts_item,
                          "spans": spans})
        for i in range(len(units)):
            lo, hi = max(0, i - 1), min(len(units), i + 2)
            win = body[units[lo]["a"]:units[hi - 1]["b"]]
            win_texts.append(f"{prefix}{win}")
        out[str(cid)] = {"prefix": prefix, "units": units}
    C.write_json(UNITS_JSON, out)
    n = len(win_texts)
    print(f"[units] {n} unidades em {len(out)} chunks (média {n/len(out):.1f}/chunk)")
    vecs = []
    for i in range(0, n, 32):
        batch = [C.DOC_PROMPT + t for t in win_texts[i:i + 32]]
        r = requests.post(f"{C.EMB_URL}/v1/embeddings", json={"input": batch, "model": "emb"},
                          timeout=600)
        r.raise_for_status()
        d = sorted(r.json()["data"], key=lambda x: x["index"])
        e = np.array([x["embedding"] for x in d], dtype=np.float32)
        vecs.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9))
        if (i // 32) % 100 == 0:
            print(f"  ... {i}/{n}", flush=True)
    np.savez(UNITS_NPZ, vecs=np.vstack(vecs))
    print(f"[ok] {UNITS_JSON} + {UNITS_NPZ}")


@lru_cache(maxsize=1)
def _tok():
    from tokenizers import Tokenizer
    import os
    path = os.environ.get("LFM3_TOKENIZER", "/tmp/f4_base/lfm3-1.7b-data2x-ep4-merged/tokenizer.json")
    return Tokenizer.from_file(path)


_STOP = None


def _lex_tokens(text: str) -> List[str]:
    global _STOP
    if _STOP is None:
        sys.path.insert(0, str(C.REPO / "retrieval3"))
        from eval_common import APP_STOPWORDS
        _STOP = set(APP_STOPWORDS)
    return [t for t, _, _ in C.tokens_with_offsets(text) if len(t) > 2 and t not in _STOP]


def ntok(text: str) -> int:
    return len(_tok().encode(text, add_special_tokens=False).ids)


class Compressor:
    def __init__(self):
        self.units = json.loads(UNITS_JSON.read_text(encoding="utf-8"))
        self.vecs = np.load(UNITS_NPZ)["vecs"]
        self.chunks = C.load_chunks()
        # deslocamento da 1ª janela de cada chunk na matriz
        self.base: Dict[int, int] = {}
        k = 0
        for cid in sorted(self.chunks):
            self.base[cid] = k
            k += len(self.units[str(cid)]["units"])

    # ---------------------------------------------------------------- render
    @staticmethod
    def source_label(ch: Dict[str, Any]) -> str:
        """== format_tool_result do F4: 'doc - seção - título'."""
        src = f"{ch['doc']} - {ch['section']}"
        if ch.get("title"):
            src += f" - {ch['title']}"
        return src

    def render_raw(self, cids: List[int]) -> str:
        parts = []
        for i, cid in enumerate(cids, 1):
            ch = self.chunks[cid]
            parts.append(f"[{i}] ({self.source_label(ch)}):\n{ch['text']}")
        return "\n\n".join(parts)

    def _unit_text(self, cid: int, j: int) -> str:
        u = self.units[str(cid)]["units"][j]
        _, body, _ = C.chunk_body(self.chunks[cid]["text"])
        t = body[u["a"]:u["b"]]
        if not u["starts_item"] and u["item"]:
            t = f"(item {u['item']}) {t}"
        return t

    def render_selected(self, sel: Dict[int, List[int]], order: List[int]) -> str:
        parts = []
        i = 0
        for cid in order:
            js = sorted(sel.get(cid, []))
            if not js:
                continue
            i += 1
            ch = self.chunks[cid]
            prefix = self.units[str(cid)]["prefix"].rstrip(" —").strip()
            pieces, prev = [], None
            for j in js:
                if prev is not None and j != prev + 1:
                    pieces.append("[…]")
                pieces.append(self._unit_text(cid, j) if (prev is None or j != prev + 1)
                              else self._unit_text_plain(cid, j))
                prev = j
            body = " ".join(pieces)
            parts.append(f"[{i}] ({self.source_label(ch)}):\n{prefix} — {body}")
        return "\n\n".join(parts)

    def _unit_text_plain(self, cid: int, j: int) -> str:
        u = self.units[str(cid)]["units"][j]
        _, body, _ = C.chunk_body(self.chunks[cid]["text"])
        return body[u["a"]:u["b"]]

    # ---------------------------------------------------------------- seleção
    def scores(self, qv: np.ndarray, cid: int) -> np.ndarray:
        b = self.base[cid]
        n = len(self.units[str(cid)]["units"])
        return self.vecs[b:b + n] @ qv

    def lex(self, q: str, cid: int) -> np.ndarray:
        """Sobreposição lexical query↔janela (radical-6 de tokens de conteúdo, como o app)."""
        qs = {t[:6] for t in _lex_tokens(q)}
        if not qs:
            return np.zeros(len(self.units[str(cid)]["units"]))
        _, body, _ = C.chunk_body(self.chunks[cid]["text"])
        us = self.units[str(cid)]["units"]
        toks = [{t[:6] for t in _lex_tokens(body[u["a"]:u["b"]])} for u in us]
        out = []
        for i in range(len(us)):
            w = set().union(*toks[max(0, i - 1):i + 2])
            out.append(len(qs & w) / len(qs))
        return np.array(out)

    def compress(self, qv: np.ndarray, cids: List[int], budget: int,
                 q: str = "", mode: str = "dense", lam: float = 0.3,
                 prior: float = 0.0) -> Dict[str, Any]:
        """Seleção gulosa por score até `budget` tokens (contando o render final).

        mode: dense = cos(query, janela-3); hybrid = cos + lam·lexical;
        prior: bônus por posição do chunk no ranking (rank r ganha prior/(r+1))."""
        cand = []
        for r, cid in enumerate(cids):
            sc = self.scores(qv, cid)
            if mode == "hybrid" and q:
                sc = sc + lam * self.lex(q, cid)
            for j, s in enumerate(sc):
                cand.append((float(s) + prior / (r + 1), r, cid, j))
        cand.sort(key=lambda x: -x[0])
        # custo incremental estimado por unidade (tokens da unidade + marcadores) e do
        # cabeçalho de cada chunk; confere o render real no fim e remove o pior se estourar.
        hdr = {cid: ntok(f"[9] ({self.source_label(self.chunks[cid])}):\n"
                         f"{self.units[str(cid)]['prefix'].rstrip(' —').strip()} — ") + 2
               for cid in cids}
        sel: Dict[int, List[int]] = {}
        used = 0
        picked: List[tuple] = []
        for s, r, cid, j in cand:
            cost = ntok(self._unit_text(cid, j)) + 3 + (hdr[cid] if cid not in sel else 0)
            if used + cost > budget:
                continue
            sel.setdefault(cid, []).append(j)
            used += cost
            picked.append((s, cid, j))
        if not sel:   # garante ao menos a melhor unidade
            s, r, cid, j = cand[0]
            sel = {cid: [j]}
            picked = [(s, cid, j)]
        text = self.render_selected(sel, cids)
        while ntok(text) > budget and len(picked) > 1:
            s, cid, j = picked.pop()            # remove a de menor score
            sel[cid].remove(j)
            if not sel[cid]:
                del sel[cid]
            text = self.render_selected(sel, cids)
        spans = {cid: [sp for j in js for sp in self.units[str(cid)]["units"][j]["spans"]]
                 for cid, js in sel.items()}
        return {"text": text, "sel": sel, "spans": spans, "ntok": ntok(text)}


if __name__ == "__main__":
    build_units()
