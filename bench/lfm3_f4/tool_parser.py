#!/usr/bin/env python3
"""
tool_parser.py — parser do formato XML nativo de tool-call do LFM3.

O template oficial emite:
    <tool_call>
    <function=buscar_norma>
    <parameter=consulta>
    VALOR
    </parameter>
    <parameter=nr>
    VALOR|null
    </parameter>
    </function>
    </tool_call>

Este parser é o INVERSO determinístico desse render (round-trip provado em
verify_format.py). Valores multi-linha são preservados; 'null' literal vira
None (o render_value do template emite 'null' para None).

Comentários PT-BR; identificadores em inglês.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n(.*?)\n</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extrai todas as tool calls do texto gerado. Retorna
    [{"name": str, "arguments": {k: v|None}}, ...]."""
    calls: List[Dict[str, Any]] = []
    for call_block in _CALL_RE.findall(text or ""):
        m = _FUNC_RE.search(call_block)
        if not m:
            continue
        name, body = m.group(1), m.group(2)
        args: Dict[str, Any] = {}
        for pname, pval in _PARAM_RE.findall(body + "\n"):
            args[pname] = None if pval == "null" else pval
        calls.append({"name": name, "arguments": args})
    return calls


def strip_tool_calls(text: str) -> str:
    """Conteúdo textual fora dos blocos <tool_call> (resposta direta)."""
    return _CALL_RE.sub("", text or "").strip()


def split_thinking(text: str) -> Tuple[str, str]:
    """Separa (thinking, resto) quando houver bloco <think>...</think>."""
    m = re.match(r"\s*<think>(.*?)</think>", text or "", re.DOTALL)
    if m:
        return m.group(1), (text[m.end():]).strip()
    return "", (text or "").strip()


def first_call(text: str) -> Optional[Dict[str, Any]]:
    calls = parse_tool_calls(text)
    return calls[0] if calls else None
