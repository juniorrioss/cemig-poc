#!/usr/bin/env python3
"""
build_assets.py — empacota os artefatos de BUSCA do ponto de operação F6 para o HF dataset
`juniorrios/cemig-nr-retrieval-assets`. Nada é recalculado: só reempacotado a partir do
repo e dos caches (gitignored) da F5/F6.

Saídas (em --out):
  index.db               índice FTS5 v4 das 36 NRs (2202 chunks)       = android/.../assets/index.db
  manuals.db             FTS5 próprio dos 9 manuais (1092 chunks)     = bench/lfm3_f5/cache/manuals.db
  dense_text.bin         DVEC1 v4 (texto+expansão)                    = bench/ctx_topk/data/dense_text.bin
  dense_exponly.bin      DVEC1 v4 (expansão-only)                     = bench/ctx_topk/data/dense_exponly.bin
  manuals_dense.npz      denso dos manuais (ids 1_000_000+, fp32)     = bench/lfm3_f5/cache/manuals_dense.npz
  units.json             unidades do compressor (normas + manuais; só a/b/item/starts_item)
  units_win3.f16.npy     janelas-3 EmbeddingGemma em fp16 (ordem: normas por id, depois manuais)
  nr_classifier.bin      classificador NRC2 (numpy puro, sem pickle)  = classifier/models/nr_classifier.bin
  classic_winner.pkl     o mesmo classificador em pickle sklearn 1.9.1 (referência)
  config.json            ponto de operação (cópia de release/config.json)
  MANIFEST.json          md5 + bytes de cada arquivo

Uso (repo, classifier/.venv):
  classifier/.venv/bin/python release/tools/build_assets.py --out /tmp/cemig-assets \
      --f5-cache <wt>/bench/lfm3_f5/cache --f6-cache <wt>/bench/lfm3_f6/cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--f5-cache", default=str(REPO / "bench/lfm3_f5/cache"))
    ap.add_argument("--f6-cache", default=str(REPO / "bench/lfm3_f6/cache"))
    a = ap.parse_args()
    out, f5, f6 = Path(a.out), Path(a.f5_cache), Path(a.f6_cache)
    out.mkdir(parents=True, exist_ok=True)

    copies = {
        "index.db": REPO / "android/app/src/main/assets/index.db",
        "manuals.db": f5 / "manuals.db",
        "dense_text.bin": REPO / "bench/ctx_topk/data/dense_text.bin",
        "dense_exponly.bin": REPO / "bench/ctx_topk/data/dense_exponly.bin",
        "manuals_dense.npz": f5 / "manuals_dense.npz",
        "nr_classifier.bin": REPO / "classifier/models/nr_classifier.bin",
        "classic_winner.pkl": REPO / "classifier/models/classic_winner.pkl",
        "config.json": REPO / "release/config.json",
    }
    for name, src in copies.items():
        shutil.copyfile(src, out / name)

    # unidades: normas (ordem de id) + manuais (ordem de id) — mesma ordem das janelas
    un = json.loads((f5 / "units.json").read_text(encoding="utf-8"))
    um = json.loads((f6 / "units_manual.json").read_text(encoding="utf-8"))
    merged = {}
    n_units = 0
    for src in (un, um):
        for cid in sorted(src, key=int):
            u = src[cid]
            merged[cid] = {"prefix": u["prefix"],
                           "units": [[x["a"], x["b"], x["item"], int(x["starts_item"])]
                                     for x in u["units"]]}
            n_units += len(u["units"])
    (out / "units.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    vn = np.load(f5 / "units_win3.npz")["vecs"]
    vm = np.load(f6 / "units_manual_win3.npz")["vecs"]
    vecs = np.vstack([vn, vm]).astype(np.float16)
    assert vecs.shape[0] == n_units, (vecs.shape, n_units)
    np.save(out / "units_win3.f16.npy", vecs)

    manifest = {p.name: {"md5": md5(p), "bytes": p.stat().st_size}
                for p in sorted(out.iterdir()) if p.is_file() and p.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    print(f"[ok] {n_units} unidades ({len(un)} chunks de norma + {len(um)} de manual) -> {out}")


if __name__ == "__main__":
    main()
