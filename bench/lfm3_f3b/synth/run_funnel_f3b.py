"""
run_funnel_f3b.py — funil do eixo 2 do sweep (DATASET-tamanho ~2x): roda o
MESMO funil da F3 (s1 -> subsample -> s2 -> subsample -> s4; s3 pulado) nas
NRs RESTANTES do corpus, com saída em bench/lfm3_f3b/data/synth.

NRs novas (critério: fora das já usadas 6/10/12/17/18/22/34/35 E fora das
normas de CONTROLE do eval 13/31/33 — treinar nelas sujaria a leitura do
controle): 4, 7, 9, 11, 15, 19, 20, 29, 30, 32, 36, 37.

Caps IGUAIS aos da F3 (180 unidades/NR, 4 QAs/unidade) — mesma mistura.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3b/synth/run_funnel_f3b.py \
        [--nr 4] [--stage all|s1|s2|s4] [--workers 16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3b/synth
F3B = HERE.parent
REPO = F3B.parents[1]
F3_SYNTH_DIR = REPO / "bench" / "lfm3_f3" / "synth"
F3B_SYNTH = F3B / "data" / "synth"

NEW_NRS = [4, 7, 9, 11, 15, 19, 20, 29, 30, 32, 36, 37]

sys.path.insert(0, str(F3_SYNTH_DIR))
import common_f3  # noqa: E402

# Redireciona a saída do funil ANTES de importar os estágios (o patch do
# common_synth.SYNTH dentro de f2_synth_modules usa o valor de F3_SYNTH).
common_f3.F3_SYNTH = F3B_SYNTH
common_f3.EXTRA_NRS = NEW_NRS

import run_funnel_f3  # noqa: E402

# o run_funnel_f3 congelou as refs no import — repatcha as dele também
run_funnel_f3.F3_SYNTH = F3B_SYNTH
run_funnel_f3.EXTRA_NRS = NEW_NRS
run_funnel_f3.common_synth.SYNTH = F3B_SYNTH
F3B_SYNTH.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    run_funnel_f3.main()
