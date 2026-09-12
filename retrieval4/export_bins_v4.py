#!/usr/bin/env python3
"""
export_bins_v4.py — Exporta os índices densos v4 em formato DVEC1 para os assets do app.

Reencoda os documentos com o encoder GGUF on-device (:8399) usando a expansão escolhida e
grava o binário DVEC1 (idêntico ao retrieval3/export_dense_bin.py, mas a partir dos vetores
numpy em vez de sqlite-vec). Formato:
  magic b"DVEC1" | int32 dim | int32 count | por chunk: int32 id + dim×float32 (L2-norm).

Combo v4 embarcado (do sweep_frenteb + calibrate_v4):
  dense_text.bin    = MANTÉM v3 (a expansão v4 no texto piora o denso query↔passagem);
  dense_exponly.bin = v4 (expansão v4-only ajuda query↔query e robustez a ASR).

Uso (classifier/.venv, servidor GGUF em :8399):
  ../classifier/.venv/bin/python export_bins_v4.py --which exp --out dense_exponly.bin
Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))
sys.path.insert(0, str(_ROOT / "retrieval3"))
sys.path.insert(0, str(_HERE))

from dense_v4 import DOC_PROMPT, embed_batch, load_chunks, load_exp  # noqa: E402

MAGIC = b"DVEC1"


def build_texts(which: str):
    e3 = load_exp(_ROOT / "retrieval3" / "data" / "expansions.jsonl", "v3")
    e4 = load_exp(_HERE / "data" / "expansions_v4.jsonl", "v4")
    rows = load_chunks(str(_ROOT / "corpus" / "index_hf_36nr.db"))
    texts = []
    for r in rows:
        if which == "exp":  # expansão v4-only (query↔query)
            ex = (e3.get(r["id"], "") + " " + e4.get(r["id"], "")).strip()
            texts.append(ex or r["text"])
        else:  # text = v3 (texto + expansão v3), NÃO usa v4 no texto
            ev3 = e3.get(r["id"], "")
            texts.append(r["text"] + " " + ev3 if ev3 else r["text"])
    return rows, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8399")
    ap.add_argument("--which", choices=["text", "exp"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows, texts = build_texts(args.which)
    parts = []
    for i in range(0, len(texts), args.batch):
        parts.append(embed_batch(args.url, [DOC_PROMPT + t for t in texts[i:i + args.batch]]))
        if (i // args.batch) % 15 == 0:
            print(f"  {i + min(args.batch, len(texts) - i)}/{len(texts)}", file=sys.stderr, flush=True)
    emb = np.vstack(parts).astype(np.float32)
    dim = emb.shape[1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", dim, len(rows)))
        for j, r in enumerate(rows):
            v = emb[j]
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            f.write(struct.pack("<i", int(r["id"])))
            f.write(v.astype(np.float32).tobytes())
    print(f"{args.which}: {out} count={len(rows)} dim={dim} size={out.stat().st_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
