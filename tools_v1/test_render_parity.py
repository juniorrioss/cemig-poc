#!/usr/bin/env python3
"""
test_render_parity.py — prova que render_jinja (classifier/.venv) == render (transformers).

Roda em DOIS venvs: gera a saída jinja no classifier/.venv e a saída transformers no
.venv-train, compara byte-a-byte. Sem paridade, os exemplos filtrados no classifier/.venv
poderiam divergir do que o treino renderiza. Uso:

  ../classifier/.venv/bin/python test_render_parity.py --emit jinja  > /tmp/j.txt
  ../.venv-train/bin/python     test_render_parity.py --emit hf     > /tmp/h.txt
  diff /tmp/j.txt /tmp/h.txt && echo PARIDADE_OK

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_schema import (TOOLS, format_tool_result, make_assistant_message,
                         make_system_message, make_tool_call_message,
                         make_tool_result_message, make_user_message)

CHUNK = {"doc": "NR-10", "section": "10.2.8.2", "title": "Proteção coletiva",
         "text": "Distância de segurança para 13,8 kV conforme o Anexo II."}


def dialogs():
    return [
        [make_system_message(), make_user_message("perto de 13,8 kV, distância?"),
         make_tool_call_message("distancia seguranca 13,8 kV media tensao", "NR-10"),
         make_tool_result_message(format_tool_result([CHUNK])),
         make_assistant_message("Conforme a NR-10, item 10.2.8.2, respeite o Anexo II.")],
        [make_system_message(), make_user_message("bom dia!"),
         make_assistant_message("Bom dia! Como posso ajudar com as normas?")],
        [make_system_message(), make_user_message("aspas 'x', vírgula e çãé"),
         make_tool_call_message("consulta com aspas 'x', vírgula e çãé", None),
         make_tool_result_message(format_tool_result([CHUNK])),
         make_assistant_message("Resposta final de teste.")],
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["jinja", "hf"], required=True)
    args = ap.parse_args()
    if args.emit == "jinja":
        import render_jinja as R
        render = R.render
    else:
        import render as R
        render = R.render
    for i, msgs in enumerate(dialogs()):
        text = render(msgs, tools=TOOLS)
        print(f"===DIALOG {i}===")
        print(text)


if __name__ == "__main__":
    main()
