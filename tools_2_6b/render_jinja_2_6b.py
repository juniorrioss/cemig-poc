#!/usr/bin/env python3
"""
render_jinja_2_6b.py — Renderização via o chat_template.jinja OFICIAL do LFM2.5-2.6B.

O 2.6B é um modelo de RACIOCÍNIO: o único ponto em que o template dele difere do 1.2B é o
GENERATION PROMPT, que termina em `<|im_start|>assistant\n<think>` (o 1.2B termina em
`<|im_start|>assistant\n`). Todo o resto — system, tools (`List of tools: [...]`), turno
`tool`, e a renderização da tool_call `<|tool_call_start|>[buscar_norma(...)]<|tool_call_end|>`
— é BYTE-A-BYTE idêntico ao 1.2B (provado em `verify_2_6b.py`).

Para a bateria de tool-calling do 2.6B ser MAÇÃ-COM-MAÇÃ com a do 1.2B, o corpo dos turnos
(system/user/tool/assistant) tem de ser o mesmo — o que muda é só o SUFIXO do generation
prompt. Este módulo carrega o template do 2.6B e o renderiza em jinja2 puro (paridade
byte-a-byte com apply_chat_template, mesma técnica de tools_v1/render_jinja).

`thinking_off=True` (default): remove o sufixo `<think>` do generation prompt — replica o
`--reasoning-budget 0` / `enable_thinking=False` que o `sft_v3` usou p/ o 2.6B (o modelo
responde direto, sem gastar tokens de raciocínio; obrigatório p/ latência de voz e p/ a
síntese/recusa ser comparável ao 1.2B). Com `thinking_off=False`, deixa o `<think>` do template
oficial (para medir o custo do raciocínio, se pedido).

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment
from jinja2.exceptions import TemplateError

_HERE = Path(__file__).resolve().parent
# reusa o parser round-trip do tools_v1 (puro python, sem transformers)
sys.path.insert(0, str(_HERE.parent / "tools_v1"))
from render import parse_tool_calls  # noqa: E402

TEMPLATE_PATH = _HERE / "data" / "chat_template_2_6b.jinja"
BOS_TOKEN = "<|startoftext|>"  # mesmo do 1.2B (tokenizer.json idêntico em special tokens)


def _raise_exception(msg: str):
    raise TemplateError(msg)


def _strip_generation_tags(src: str) -> str:
    """Remove os tags {% generation %}/{% endgeneration %} (extensão do transformers)."""
    return re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    import json
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


@lru_cache(maxsize=1)
def _template():
    env = Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True,
                      autoescape=False)
    env.globals["raise_exception"] = _raise_exception
    env.filters["tojson"] = _tojson
    src = _strip_generation_tags(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return env.from_string(src)


def render(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
           add_generation_prompt: bool = False, thinking_off: bool = True) -> str:
    """Renderiza no formato nativo do LFM2.5-2.6B.

    thinking_off=True remove o sufixo `<think>` do generation prompt (replica enable_thinking
    =False do sft_v3). Sem generation prompt (add_generation_prompt=False) o parâmetro é no-op.
    """
    text = _template().render(messages=messages, tools=tools, bos_token=BOS_TOKEN,
                              add_generation_prompt=add_generation_prompt)
    if add_generation_prompt and thinking_off and text.endswith("<think>"):
        text = text[: -len("<think>")]
    return text
