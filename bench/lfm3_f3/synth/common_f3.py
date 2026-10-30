"""
common_f3.py — utilitários da F3 (robustez de uso de contexto do LFM3-1.7B).

Reusa o funil sintético da F2 (bench/lfm3_f2/synth) SEM copiá-lo: importa os
módulos de estágio com os caminhos de saída redirecionados para
bench/lfm3_f3/data/synth. As NRs extras da F3 (variedade, ordem do brief) são
definidas aqui — NÃO mexem no norm_selection.json da F2.

NRs extras (critério: presença forte no benchmark_mobile + diversidade de
domínio + fora das tratadas da F2 10/18/34):
  NR-6 (EPI, 48 queries), NR-12 (máquinas, 108), NR-17 (ergonomia, 44),
  NR-22 (mineração, 132), NR-35 (altura, 92).

Comentários em português; identificadores em inglês (padrão do projeto).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3/synth
F3 = HERE.parent                                   # bench/lfm3_f3
REPO = F3.parents[1]                               # repo root
F2_SYNTH_DIR = REPO / "bench" / "lfm3_f2" / "synth"
F3_DATA = F3 / "data"
F3_SYNTH = F3_DATA / "synth"

EXTRA_NRS = [6, 12, 17, 22, 35]

# Caps de custo (27B): subamostragem seedada ANTES do s2/s4 — a F3 precisa de
# VARIEDADE de perguntas (âncora do retriever), não de cobertura exaustiva da
# norma (que era o objetivo da F2).
MAX_UNITS_PER_NR = 180      # unidades sorteadas por NR antes do s2
MAX_QA_PER_UNIT = 4         # QAs sorteados por unidade antes do s4 (crítico)
SEED = 20260206


def f2_synth_modules():
    """Importa common_synth da F2 com SYNTH redirecionado p/ a F3."""
    sys.path.insert(0, str(F2_SYNTH_DIR))
    import common_synth  # noqa: PLC0415
    common_synth.SYNTH = F3_SYNTH
    F3_SYNTH.mkdir(parents=True, exist_ok=True)
    return common_synth
