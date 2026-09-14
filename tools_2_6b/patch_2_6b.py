#!/usr/bin/env python3
"""
patch_2_6b.py — injeta o renderer do 2.6B (thinking-OFF) nos módulos de eval do tools_v1.

O valor desta task é a COMPARABILIDADE: rodar EXATAMENTE a mesma bateria do 1.2B
(tools_v1/run_eval.py, score_e2e.py; tools_oraculo/harness_oraculo.py) contra o 2.6B, sem
reescrever nem tocar aquele código. A única diferença de FORMATO entre 1.2B e 2.6B é o
generation prompt (o 2.6B é reasoning: termina em `<|im_start|>assistant\n<think>`).

Os módulos de eval fazem `import render_jinja as RJ` e chamam `RJ.render(...)`. Como a busca do
atributo `render` acontece em tempo de CHAMADA, basta reatribuir `render_jinja.render` para o
renderer do 2.6B (thinking-OFF por default, replicando o `enable_thinking=False` do sft_v3) e
todos os consumidores passam a renderizar no formato do 2.6B — o corpo dos turnos é byte-a-byte
idêntico ao 1.2B (provado em verify_2_6b.py), então só muda o generation prompt.

Importe este módulo ANTES de qualquer módulo de eval do tools_v1/tools_oraculo.
Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tools_v1"))

import render_jinja  # noqa: E402  (template 1.2B — vamos sobrescrever .render)
import render_jinja_2_6b as RJ26  # noqa: E402


def _render_2_6b(messages, tools=None, add_generation_prompt=False):
    """Adapter: sempre thinking-OFF (replica sft_v3; comparável ao 1.2B)."""
    return RJ26.render(messages, tools=tools,
                       add_generation_prompt=add_generation_prompt, thinking_off=True)


# aplica o patch no import (idempotente).
render_jinja.render = _render_2_6b
PATCHED = True
