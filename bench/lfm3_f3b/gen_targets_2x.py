#!/usr/bin/env python3
"""
gen_targets_2x.py — alvos do 27B para a variante data2x: reusa o
gen_targets.py da F3 com TRAIN apontado para data/train/data2x (os alvos do
pool antigo são reaproveitados via ckpt por qa_id:case — só os QAs novos do
funil f3b custam tokens).

Uso: classifier/.venv/bin/python bench/lfm3_f3b/gen_targets_2x.py --workers 12
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3b
REPO = HERE.parents[1]
F3 = REPO / "bench" / "lfm3_f3"
OUT = HERE / "data" / "train" / "data2x"

sys.path.insert(0, str(F3 / "synth"))

spec = importlib.util.spec_from_file_location(
    "f3_gen_targets", F3 / "synth" / "gen_targets.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

# semeia o ckpt da variante com o da F3 (reaproveita alvos já pagos)
OUT.mkdir(parents=True, exist_ok=True)
ck_new = OUT / "targets.ckpt.json"
ck_old = F3 / "data" / "train" / "targets.ckpt.json"
if not ck_new.exists() and ck_old.exists():
    shutil.copy(ck_old, ck_new)
    print(f"[seed] ckpt da F3 copiado ({len(json.loads(ck_new.read_text()))} entradas)")

gt.TRAIN = OUT

if __name__ == "__main__":
    gt.main()
