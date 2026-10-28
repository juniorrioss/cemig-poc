"""
prep_corpus.py — prepara o corpus das 3 normas TRATADAS no formato "chunked"
do graphgen-mask (lista de documentos; documento = lista de {"content": ...}).

Reusa o janelamento do funil (split_windows) p/ chunks ~6000 chars.

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/k2v/prep_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synth"))
from common_synth import load_norm_corpus, treated_norms  # noqa: E402
from s1_decompose import split_windows  # noqa: E402

OUT = HERE / "data"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    docs = []
    for nr in treated_norms():
        corpus = load_norm_corpus(nr)
        windows = split_windows(corpus["conteudo"], window=3000)
        docs.append([{"content": w} for w in windows])
        print(f"NR-{nr}: {len(windows)} chunks")
    out = OUT / "treated_corpus.json"
    out.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {out} ({sum(len(d) for d in docs)} chunks em {len(docs)} docs)")


if __name__ == "__main__":
    main()
