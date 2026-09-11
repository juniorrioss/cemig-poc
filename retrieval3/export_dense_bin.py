#!/usr/bin/env python3
"""
export_dense_bin.py — Exporta os vetores densos (do encoder GGUF on-device) num formato
binário compacto para os assets do app Android (busca vetorial em memória, brute-force
exato em Kotlin, sem dependência nativa nova).

Formato binário "DVEC1" (little-endian):
  magic   : 5 bytes  = b"DVEC1"
  dim     : int32     (768)
  count   : int32     (2202)
  então, para cada chunk (ordenado por id crescente):
    id    : int32
    vec   : dim × float32   (L2-normalizado)

Os vetores são L2-normalizados -> cosseno = produto interno. A metadata (doc/section/
title/text) NÃO vai aqui: o app já a tem no índice FTS5 expandido (join por id). Assim
cada índice ocupa ~6.8 MB (2202 × (4 + 768×4)).

Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

MAGIC = b"DVEC1"


def export(vec_db: str, out_bin: str) -> dict:
    import sqlite_vec
    con = sqlite3.connect(vec_db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    rows = con.execute("SELECT rowid, embedding FROM vec ORDER BY rowid").fetchall()
    con.close()

    dim = None
    packed = []
    for rowid, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if dim is None:
            dim = len(v)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        packed.append((int(rowid), v.astype(np.float32)))

    out = Path(out_bin)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", dim, len(packed)))
        for rowid, v in packed:
            f.write(struct.pack("<i", rowid))
            f.write(v.tobytes())
    return {"out": out_bin, "count": len(packed), "dim": dim,
            "size_mb": round(out.stat().st_size / 1e6, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-db", required=True)
    ap.add_argument("--out-bin", required=True)
    args = ap.parse_args()
    import json
    print(json.dumps(export(args.vec_db, args.out_bin), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
