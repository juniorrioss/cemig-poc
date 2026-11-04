#!/usr/bin/env python3
"""
export_assets.py — F7: gera os assets do app (Android) para o ponto de operação da F6
(thr0.7_f2+man1.75+margin -> top-5 -> compressão 50%) a partir dos caches do ESPELHO
Python (bench/lfm3_f5 + bench/lfm3_f6). Nada é recalculado: só reempacotado.

Saídas (android/app/src/main/assets/):
  manuals.db            cópia de bench/lfm3_f5/cache/manuals.db (1092 chunks, FTS5 próprio,
                        mesmo tokenizer unicode61 remove_diacritics 2) — schema inalterado.
  manuals_dense.bin     DVEC1 (fp32) de manuals_dense.npz (ids 1_000_000+idx).
  units.bin             UNIT1: unidades do compressor (2202 normas + 1092 manuais, ordem = ordem
                        das linhas de units_win3.f16) com texto, item, starts_item, RADICAIS
                        lexicais (= compress._lex_tokens -> t[:6]), custo em tokens da unidade
                        (ntok(_unit_text)+3) e do cabeçalho de cada chunk (hdr) — os mesmos
                        valores que o compress.Compressor calcula; o app não reimplementa
                        offsets/regex Unicode nem precisa do tokenizer para a seleção gulosa.
  units_win3.f16        HVEC1: magic "HVEC1" + dim(i32 LE) + count(i32 LE) + count*dim fp16 LE.
                        fp16 (não int8): simulado no espelho, fp16 reproduz 344/344 contextos
                        op_d50 byte a byte; int8 só 332/344 (ver RESULTS.md).
  lfm3_tools_system_prompt.txt  system BAKED do f4-r32-ep2 (tool_schema_f4.baked_system_content).

Também grava data/assets_md5.json.

Uso: classifier/.venv/bin/python bench/lfm3_f7/export_assets.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f6"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f5"))

import common_f5 as C  # noqa: E402
import tool_schema_f4 as T  # noqa: E402
from compress import Compressor, _lex_tokens, ntok  # noqa: E402

ASSETS = REPO / "android" / "app" / "src" / "main" / "assets"
F5C = REPO / "bench" / "lfm3_f5" / "cache"
F6C = REPO / "bench" / "lfm3_f6" / "cache"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_dvec(path: Path, ids: np.ndarray, mat: np.ndarray) -> None:
    with path.open("wb") as fh:
        fh.write(b"DVEC1")
        fh.write(struct.pack("<ii", mat.shape[1], mat.shape[0]))
        for i, v in zip(ids, mat.astype("<f4")):
            fh.write(struct.pack("<i", int(i)))
            fh.write(v.tobytes())


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    # 1) manuals.db (schema da F5 já serve: mchunks + mchunks_fts com 4 campos)
    shutil.copyfile(F5C / "manuals.db", ASSETS / "manuals.db")
    # 2) denso dos manuais (fp32: o gate por margem compara com o denso das normas)
    z = np.load(F5C / "manuals_dense.npz")
    write_dvec(ASSETS / "manuals_dense.bin", z["ids"], z["vecs"])

    # 3) unidades + janelas, na MESMA ordem do CompressorF6 (normas ordenadas, depois manuais)
    units_n = json.loads((F5C / "units.json").read_text(encoding="utf-8"))
    units_m = json.loads((F6C / "units_manual.json").read_text(encoding="utf-8"))
    norm_chunks = C.load_chunks()
    man = sqlite3.connect(str(F5C / "manuals.db"))
    man.row_factory = sqlite3.Row
    man_meta = {r["id"]: dict(r) for r in man.execute("SELECT * FROM mchunks")}
    meta = dict(norm_chunks)
    meta.update(man_meta)
    order = [(cid, units_n[str(cid)]) for cid in sorted(norm_chunks)]
    order += [(cid, units_m[str(cid)]) for cid in sorted(man_meta)]
    vecs = np.vstack([np.load(F5C / "units_win3.npz")["vecs"], np.load(F6C / "units_manual_win3.npz")["vecs"]])

    def bstr(fh, s: str) -> None:
        b = s.encode("utf-8")
        fh.write(struct.pack("<i", len(b)))
        fh.write(b)

    # UNIT1: magic + nChunks; por chunk: cid(i64) label prefixClean hdrTok(i32) nUnits(i32);
    # por unidade: item starts(i8) tokFull(i32) text lex(stems separados por espaço).
    # Strings = len(i32 LE) + bytes UTF-8. Valores = os do compress.Compressor (espelho).
    row = 0
    with (ASSETS / "units.bin").open("wb") as fh:
        fh.write(b"UNIT1")
        fh.write(struct.pack("<i", len(order)))
        for cid, u in order:
            ch = meta[cid]
            _, body, _ = C.chunk_body(ch["text"])
            label = Compressor.source_label(ch)
            prefix_clean = u["prefix"].rstrip(" —").strip()
            hdr = ntok(f"[9] ({label}):\n{prefix_clean} — ") + 2
            fh.write(struct.pack("<q", int(cid)))
            bstr(fh, label)
            bstr(fh, prefix_clean)
            fh.write(struct.pack("<ii", hdr, len(u["units"])))
            for un in u["units"]:
                t = body[un["a"]:un["b"]]
                full = t if (un["starts_item"] or not un["item"]) else f"(item {un['item']}) {t}"
                lex = " ".join(sorted({x[:6] for x in _lex_tokens(t)}))
                bstr(fh, un["item"] or "")
                fh.write(struct.pack("<bi", int(bool(un["starts_item"])), ntok(full) + 3))
                bstr(fh, t)
                bstr(fh, lex)
                row += 1
    assert row == len(vecs), (row, len(vecs))
    with (ASSETS / "units_win3.f16").open("wb") as fh:
        fh.write(b"HVEC1")
        fh.write(struct.pack("<ii", vecs.shape[1], vecs.shape[0]))
        fh.write(vecs.astype("<f2").tobytes())

    # 4) system BAKED do f4-r32-ep2
    (ASSETS / "lfm3_tools_system_prompt.txt").write_text(T.baked_system_content(), encoding="utf-8")

    names = ["manuals.db", "manuals_dense.bin", "units.bin", "units_win3.f16",
             "lfm3_tools_system_prompt.txt", "index.db", "nr_classifier.bin"]
    rec = {n: {"md5": md5(ASSETS / n), "bytes": (ASSETS / n).stat().st_size} for n in names}
    rec["_units"] = {"rows": int(row), "dim": int(vecs.shape[1]), "chunks": len(order)}
    rec["system_sha256"] = hashlib.sha256(T.baked_system_content().encode()).hexdigest()
    C.write_json(HERE / "data" / "assets_md5.json", rec)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
