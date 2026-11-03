#!/usr/bin/env python3
"""
build_variant_mhx3.py — ponto 2 da curva dado×métrica do MULTIHOP (brief:
"se o multihop não decolar com o dado inicial, reporte a curva dado×métrica
antes de dobrar o dataset").

Diagnóstico do ponto 1 (ep2/ep3, dose multihop 9% vs recusa 23%): no hop1
parcial o modelo RECUSA (28/41) ou responde com o parcial (13/41) — o prior
da recusa vence a distinção "parcial→re-buscar" vs "errado→recusar". Esta
variante NÃO gera dado novo; só muda a DOSE (teste limpo da hipótese de
prior): multihop replicado 3x (411→1233, ~24%) e recusa cortada 1048→700
(~14%). Resto idêntico (mesmo seed de shuffle, mesmo val).

Saída: data/train/mhx3/{train,val}.jsonl
Uso: classifier/.venv/bin/python bench/lfm3_f4/build_variant_mhx3.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_f4 as C  # noqa: E402

OUT = C.TRAIN / "mhx3"
MH_FACTOR = 3
RECUSA_KEEP = 700


def main() -> None:
    rng = random.Random(C.SEED + 99)
    train = C.read_jsonl(C.TRAIN / "train.jsonl")
    val = C.read_jsonl(C.TRAIN / "val.jsonl")

    out_rows = []
    recusa = [r for r in train if r["family"] == "recusa"]
    rng.shuffle(recusa)
    keep_recusa = set(id(r) for r in recusa[:RECUSA_KEEP])
    for r in train:
        fam = r["family"]
        if fam == "recusa":
            if id(r) in keep_recusa:
                out_rows.append(r)
        elif fam == "multihop":
            out_rows.extend([r] * MH_FACTOR)
        else:
            out_rows.append(r)
    rng.shuffle(out_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(OUT / "train.jsonl", out_rows)
    C.write_jsonl(OUT / "val.jsonl", val)  # val INALTERADO (comparável)

    stats = {"n_train": len(out_rows),
             "by_family": dict(collections.Counter(r["family"] for r in out_rows)),
             "mh_factor": MH_FACTOR, "recusa_keep": RECUSA_KEEP}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
