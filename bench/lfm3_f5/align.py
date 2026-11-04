#!/usr/bin/env python3
"""
align.py — realinha cada chunk do índice v4 (sem offsets) ao texto-fonte do nrs.parquet.

Receita do smoke (apêndice do report): tokens \\w+ sem acento minúsculos; âncora de
6-grama; difflib.SequenceMatcher(chunk_tokens, janela_fonte[c0-30 : c0+2.5*len+60]);
spans = blocos casados (size>=3), lacunas <=8 tokens fundidas; corpo = texto após
" — " (remove o prefixo "NR-XX · seção título").

Saída: cache/align.json = {chunk_id: {"file": filename, "spans": [[s,e],...],
"cov": fração dos tokens do corpo casados}}. Também produz, por chunk, o mapa
token-do-chunk -> char da fonte (usado na compressão para dar spans às frases).

Uso: classifier/.venv/bin/python bench/lfm3_f5/align.py
"""
from __future__ import annotations

import bisect
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_f5 as C  # noqa: E402

OUT = C.CACHE / "align.json"
N_ANCHOR = 6
MIN_BLOCK = 3
MAX_GAP = 8


def _src_index(text: str):
    toks = C.tokens_with_offsets(text)
    words = [t[0] for t in toks]
    grams: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(len(words) - N_ANCHOR + 1):
        grams[tuple(words[i:i + N_ANCHOR])].append(i)
    return toks, words, grams


def align_chunk(body: str, src, want_map: bool = False):
    """Retorna (spans_char, cobertura, posição_token_inicial_na_fonte[, tokmap]).

    tokmap (want_map): lista [(char_ini_no_corpo, char_fim_no_corpo, src_ini, src_fim)] dos
    tokens CASADOS do corpo -> usado para dar spans de fonte a sub-trechos (compressão)."""
    stoks, swords, grams = src
    ctoks = C.tokens_with_offsets(body)
    cw = [t[0] for t in ctoks]
    if not cw:
        return ([], 0.0, -1, []) if want_map else ([], 0.0, -1)
    # votos de âncora: cada 6-grama do chunk que existe na fonte vota em c0 = pos - i
    votes: Dict[int, int] = defaultdict(int)
    for i in range(0, max(1, len(cw) - N_ANCHOR + 1)):
        g = tuple(cw[i:i + N_ANCHOR])
        pos = grams.get(g)
        if pos and len(pos) <= 20:
            for p in pos:
                votes[(p - i) // 20] += 1      # bucket de 20 tokens tolera inserções
    if not votes:
        return ([], 0.0, -1, []) if want_map else ([], 0.0, -1)
    best_bucket = max(votes.items(), key=lambda kv: kv[1])[0]
    c0 = best_bucket * 20
    lo = max(0, c0 - 30)
    hi = min(len(swords), c0 + int(2.5 * len(cw)) + 60)
    sm = difflib.SequenceMatcher(None, cw, swords[lo:hi], autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= MIN_BLOCK]
    if not blocks:
        return ([], 0.0, -1, []) if want_map else ([], 0.0, -1)
    matched = sum(b.size for b in blocks)
    # spans em posições de token da fonte, fundindo lacunas <= MAX_GAP
    tok_spans: List[List[int]] = []
    for b in sorted(blocks, key=lambda b: b.b):
        s, e = lo + b.b, lo + b.b + b.size      # [s, e)
        if tok_spans and s - tok_spans[-1][1] <= MAX_GAP:
            tok_spans[-1][1] = max(tok_spans[-1][1], e)
        else:
            tok_spans.append([s, e])
    spans = [[stoks[s][1], stoks[e - 1][2]] for s, e in tok_spans]
    if want_map:
        tokmap = []
        for b in blocks:
            for k in range(b.size):
                ct, st = ctoks[b.a + k], stoks[lo + b.b + k]
                tokmap.append((ct[1], ct[2], st[1], st[2]))
        return spans, matched / len(cw), tok_spans[0][0], tokmap
    return spans, matched / len(cw), tok_spans[0][0]


def main() -> None:
    S = C.load_sources()
    chunks = C.load_chunks()
    idx_cache = {}
    out = {}
    low = 0
    for cid in sorted(chunks):
        ch = chunks[cid]
        f = S["norm_file_of_doc"][ch["doc"]]
        if f not in idx_cache:
            idx_cache[f] = _src_index(S["by_file"][f])
        _, body, _ = C.chunk_body(ch["text"])
        spans, cov, _ = align_chunk(body, idx_cache[f])
        out[cid] = {"file": f, "spans": spans, "cov": round(cov, 4)}
        if cov < 0.9:
            low += 1
    covs = sorted(v["cov"] for v in out.values())
    print(f"[align] {len(out)} chunks; cov p5={covs[len(covs)//20]:.3f} "
          f"mediana={covs[len(covs)//2]:.3f}; <0.9: {low}; vazios: "
          f"{sum(1 for v in out.values() if not v['spans'])}")
    C.write_json(OUT, out)
    print(f"[ok] {OUT}")


if __name__ == "__main__":
    main()
