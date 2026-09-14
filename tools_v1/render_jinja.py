#!/usr/bin/env python3
"""
render_jinja.py — Renderização via o chat_template.jinja OFICIAL (sem transformers).

Motivo: a geração/validação por-exemplo roda no classifier/.venv (tem sklearn p/ o
retrieval v4 e a régua), que não tem transformers. Este módulo carrega o MESMO
chat_template.jinja do LFM2.5-1.2B-Instruct e o renderiza com jinja2 puro, reproduzindo
byte-a-byte o apply_chat_template (validado em test_render_parity contra .venv-train).

A renderização continua sendo do TEMPLATE OFICIAL (emenda 2): não montamos string à mão.
O parser round-trip é o mesmo de render.py (reusado).

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment
from jinja2.exceptions import TemplateError

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# reusa o parser round-trip (puro python, sem transformers)
from render import parse_tool_calls  # noqa: E402

TEMPLATE_PATH = (Path.home() / ".cache" / "huggingface" / "hub"
                 / "models--LiquidAI--LFM2.5-1.2B-Instruct" / "snapshots"
                 / "0f604ada3f766f9f257460c4c9f0b5d6f69d431b" / "chat_template.jinja")
BOS_TOKEN = "<|startoftext|>"  # confirmado no special_tokens_map do LFM2.5


def _raise_exception(msg: str):
    raise TemplateError(msg)


def _strip_generation_tags(src: str) -> str:
    """Remove os tags {% generation %}/{% endgeneration %} (extensão do transformers).

    São no-ops para o TEXTO renderizado (só marcam a região da máscara de treino, que aqui
    não precisamos — a máscara é validada em .venv-train por verify_format.py). Removê-los
    deixa o template compilável em jinja2 puro sem alterar 1 byte da saída de texto.
    """
    import re
    return re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """Réplica do filtro tojson do transformers (ensure_ascii=False, sem sort_keys)."""
    import json
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


@lru_cache(maxsize=1)
def _template():
    # autoescape=False + tojson estilo transformers: paridade byte-a-byte com apply_chat_template.
    env = Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True,
                      autoescape=False)
    env.globals["raise_exception"] = _raise_exception
    env.filters["tojson"] = _tojson
    src = _strip_generation_tags(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return env.from_string(src)


def render(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
           add_generation_prompt: bool = False) -> str:
    """Renderiza a conversa no formato nativo do LFM2.5 usando o template oficial."""
    return _template().render(messages=messages, tools=tools, bos_token=BOS_TOKEN,
                              add_generation_prompt=add_generation_prompt)


def roundtrip_ok(consulta: str, nr: Optional[str]) -> bool:
    """Renderiza uma tool_call e confere que consulta/nr voltam idênticos (PASSO 2)."""
    from tool_schema import make_system_message, make_tool_call_message, make_user_message, TOOLS
    msgs = [make_system_message(), make_user_message("(rt)"),
            make_tool_call_message(consulta, nr)]
    try:
        text = render(msgs, tools=TOOLS)
        calls = parse_tool_calls(text)
    except Exception:
        return False
    if len(calls) != 1 or calls[0]["name"] != "buscar_norma":
        return False
    a = calls[0]["arguments"]
    return a.get("consulta") == consulta and a.get("nr", None) == (nr if nr else None)


if __name__ == "__main__":
    for consulta, nr in [("distancia seguranca zona risco 13,8 kV media tensao", "NR-10"),
                         ("estabilidade estrutural poste inspecao antes de subir", None),
                         ("aspas 'simples', vírgula e acento çãé", None)]:
        print(f"[{'OK' if roundtrip_ok(consulta, nr) else 'FAIL'}] {nr} {consulta!r}")
