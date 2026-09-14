#!/usr/bin/env python3
"""
render.py — Renderização (via tokenizer oficial) + PARSER de round-trip da chamada.

Renderização: usa tokenizer.apply_chat_template do LFM2.5-1.2B-Instruct com tools=TOOLS.
Toda a sintaxe nativa (<|tool_call_start|>[buscar_norma(consulta='...', nr='NR-10')]
<|tool_call_end|>, escape de aspas, role 'tool' entre <|im_start|>tool ...) é emitida pelo
PRÓPRIO template — não montamos string à mão (emenda 2 do brief).

Parser de round-trip (PASSO 2): re-extrai (consulta, nr) do TEXTO renderizado e confere que
voltam IDÊNTICOS ao JSON de origem. Isso valida escape de aspas, acento, vírgula na consulta
e número com vírgula decimal ('13,8 kV'). A gramática da chamada é a do template:
  <|tool_call_start|>[func(arg=valor, arg2=valor2)]<|tool_call_end|>
  strings entre aspas simples com escape \\', \\n, \\r, \\\\.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MODEL = str(Path.home() / ".cache" / "huggingface" / "hub"
                    / "models--LiquidAI--LFM2.5-1.2B-Instruct" / "snapshots"
                    / "0f604ada3f766f9f257460c4c9f0b5d6f69d431b")

_TOOL_CALL_RE = re.compile(r"<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>", re.DOTALL)


@lru_cache(maxsize=2)
def get_tokenizer(model_path: str = DEFAULT_MODEL):
    """Carrega o tokenizer oficial (cacheado). Requer transformers (.venv-train)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def render(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
           add_generation_prompt: bool = False, model_path: str = DEFAULT_MODEL) -> str:
    """Renderiza a conversa completa no formato nativo do LFM2.5 (texto)."""
    tok = get_tokenizer(model_path)
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def render_and_mask(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
                    model_path: str = DEFAULT_MODEL) -> Tuple[List[int], List[int]]:
    """Renderiza tokenizando E retorna (input_ids, assistant_masks).

    assistant_masks[i]=1 nos tokens da região {% generation %} (turnos assistant),
    0 caso contrário — prova de que a máscara de treino cobre SÓ os turnos assistant
    (inclusive mascarando o turno 'tool'). Requer return_assistant_tokens_mask=True.
    """
    tok = get_tokenizer(model_path)
    enc = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True, add_generation_prompt=False)
    return enc["input_ids"], enc["assistant_masks"]


# ---------------------------------------------------------------------------
# PARSER da chamada renderizada (round-trip).
# ---------------------------------------------------------------------------
def _unescape_single_quoted(s: str) -> str:
    """Inverte o escape do format_arg_value do template (\\\\, \\', \\n, \\r)."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append({"\\": "\\", "'": "'", "n": "\n", "r": "\r"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _split_top_level_args(inner: str) -> List[str]:
    """Divide 'a=1, b=2' em ['a=1','b=2'] respeitando vírgulas dentro de strings/aspas."""
    parts: List[str] = []
    buf: List[str] = []
    in_str = False
    esc = False
    for ch in inner:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
        else:
            if ch == "'":
                in_str = True
                buf.append(ch)
            elif ch == ",":
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_arg_value(raw: str) -> Any:
    """Converte o valor renderizado de volta ao tipo Python."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return _unescape_single_quoted(raw[1:-1])
    if raw == "None" or raw == "null":
        return None
    # números / listas / dicts via literal_eval (o template usa tojson p/ não-strings)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def parse_tool_calls(rendered_text: str) -> List[Dict[str, Any]]:
    """Extrai as chamadas do texto renderizado como [{'name':..., 'arguments':{...}}].

    Levanta ValueError se a sintaxe não parseia (validade sintática do PASSO 2/4b).
    """
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(rendered_text):
        body = m.group(1).strip()
        if not body:
            continue
        # o corpo é uma lista de chamadas func(args) separadas por vírgula de TOPO.
        # localiza cada 'nome(' ... ')' balanceado.
        for call_str in _split_calls(body):
            name, inner = _split_name_args(call_str)
            args: Dict[str, Any] = {}
            for piece in _split_top_level_args(inner):
                if "=" not in piece:
                    raise ValueError(f"arg sem nome: {piece!r} em {call_str!r}")
                k, v = piece.split("=", 1)
                args[k.strip()] = _parse_arg_value(v)
            calls.append({"name": name, "arguments": args})
    return calls


def _split_calls(body: str) -> List[str]:
    """Separa múltiplas chamadas func(...) no topo da lista, respeitando parênteses/strings."""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str = False
    esc = False
    for ch in body:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _split_name_args(call_str: str) -> Tuple[str, str]:
    """'func(a=1, b=2)' -> ('func', 'a=1, b=2')."""
    call_str = call_str.strip()
    lp = call_str.find("(")
    if lp < 0 or not call_str.endswith(")"):
        raise ValueError(f"chamada malformada: {call_str!r}")
    return call_str[:lp].strip(), call_str[lp + 1:-1]


def roundtrip_check(consulta: str, nr: Optional[str],
                    model_path: str = DEFAULT_MODEL) -> Tuple[bool, str, Dict[str, Any]]:
    """Renderiza uma tool_call e re-parseia; confere consulta/nr IDÊNTICOS ao original.

    Retorna (ok, texto_renderizado, args_parseados). Prova o escape de aspas/acento/vírgula
    decimal ('13,8 kV') sem regex de conversão — a renderização é do template.
    """
    from tool_schema import make_system_message, make_tool_call_message, make_user_message, TOOLS
    msgs = [make_system_message(), make_user_message("(teste de round-trip)"),
            make_tool_call_message(consulta, nr)]
    text = render(msgs, tools=TOOLS, model_path=model_path)
    calls = parse_tool_calls(text)
    if len(calls) != 1:
        return False, text, {}
    args = calls[0]["arguments"]
    ok = (calls[0]["name"] == "buscar_norma"
          and args.get("consulta") == consulta
          and args.get("nr", None) == (nr if nr else None))
    return ok, text, args


if __name__ == "__main__":
    # smoke local: round-trip dos casos difíceis do capitão.
    cases = [
        ("distancia seguranca zona risco 13,8 kV 13.8 kV media tensao", "NR-10"),
        ("estabilidade estrutural poste inspecao antes de subir", None),
        ("uso de camiseta de algodao rasgada vestimenta trabalho", "NR-10"),
        ("aspas 'simples' e vírgula, acento çãé no meio", None),
    ]
    for consulta, nr in cases:
        ok, text, args = roundtrip_check(consulta, nr)
        print(f"[{'OK ' if ok else 'FAIL'}] nr={nr!r} args={args}")
        if not ok:
            print(text)
