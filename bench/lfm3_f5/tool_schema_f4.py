#!/usr/bin/env python3
# Cópia VERBATIM de bench/lfm3_f4/tool_schema_f4.py (branch fm/lfm3-f4-tools, commit d26ec60) — contrato
# do f4-r32-ep2; copiado p/ a F5 ser auto-contida enquanto a F4 não está em main.
"""
tool_schema_f4.py — contrato ÚNICO da ferramenta do F4 (LFM3-1.7B) e
construtores de mensagens. O formato nativo do LFM3 é XML <tool_call>
emitido pelo PRÓPRIO chat_template (apply_chat_template com tools=...);
nós nunca montamos string de formato à mão (receita tools_v1: o 27B gera
CONTEÚDO, a renderização é determinística do template).

Decisões (brief F4):
  - buscar_norma(consulta: str, nr: str|null). 'consulta' = a FALA CRUA do
    usuário (evidência tools_v1: reescrita pelo modelo PIORA a busca).
  - 'nr' = a norma quando citada/inequívoca; null caso contrário.
  - arguments SEMPRE com as duas chaves (consulta e nr) — uniformidade de
    schema exigida pelo Arrow/datasets (chave ausente viraria null silencioso
    no treino e divergiria do render de geração).
  - NÃO chamar é comportamento válido; re-chamar (multihop) com argumento
    DIFERENTE quando a 1ª busca não basta é a família nova.

Comentários PT-BR; identificadores em inglês.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

TOOL_NAME = "buscar_norma"

TOOL_BUSCAR_NORMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Busca trechos das Normas Regulamentadoras (NRs) brasileiras de segurança do "
            "trabalho para fundamentar a resposta. Chame quando precisar de um dado "
            "normativo que ainda não está na conversa; chame DE NOVO com outros termos se "
            "os trechos retornados não bastarem para responder. NÃO chame para saudações, "
            "agradecimentos, conversa geral, nem quando os trechos já consultados nesta "
            "conversa respondem à nova pergunta."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": (
                        "Os termos de busca. Use a própria pergunta do usuário como ela é "
                        "(fala crua); numa SEGUNDA busca da mesma pergunta, use termos "
                        "diferentes, focados no que faltou."
                    ),
                },
                "nr": {
                    "type": ["string", "null"],
                    "description": (
                        "A norma no formato 'NR-10' quando citada ou inequívoca; senão null."
                    ),
                },
            },
            "required": ["consulta", "nr"],
        },
    },
}

TOOLS: List[Dict[str, Any]] = [TOOL_BUSCAR_NORMA]

# ---------------------------------------------------------------------------
# System prompt do F4 (treino == avaliação de tool == app futuro).
# Baseia-se no SYSTEM_PROMPT do F0 (mesma persona/regras de resposta) e
# ACRESCENTA a política de uso da ferramenta. As células estáticas de
# regressão continuam usando o SYSTEM_PROMPT do F0 SEM tools (teste de que o
# modo sem ferramenta não regrediu).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_F4 = (
    "Você é um assistente técnico de segurança do trabalho, especialista em "
    "Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do "
    "Brasil, de forma objetiva e curta.\n"
    "Você tem a ferramenta buscar_norma(consulta, nr) para consultar as NRs. "
    "Decida sozinho quando usá-la:\n"
    "- CHAME quando precisar de um dado normativo que ainda não está na conversa. "
    "No campo 'consulta', use a pergunta do usuário como ela é; preencha 'nr' só "
    "quando a norma for citada ou inequívoca, senão null.\n"
    "- Se os trechos retornados responderem só PARTE da pergunta, CHAME DE NOVO "
    "com uma consulta diferente, focada no que faltou, antes de responder.\n"
    "- NÃO CHAME para saudações, agradecimentos, conversa geral, nem quando os "
    "trechos já consultados nesta conversa respondem à nova pergunta (reuse o "
    "contexto).\n"
    "Na resposta final: baseie-se exclusivamente nos trechos consultados e cite a "
    "fonte (norma e item). Se os trechos não contiverem a informação, diga isso "
    "claramente em vez de inventar. Quando não houver contexto, responda com seu "
    "conhecimento das normas, citando a norma quando souber."
)


# ---------------------------------------------------------------------------
# Construtores de mensagens (dicts). O tokenizer emite os tokens.
# ---------------------------------------------------------------------------
def make_system_message() -> Dict[str, Any]:
    return {"role": "system", "content": SYSTEM_PROMPT_F4}


def make_user_message(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def make_tool_call_message(consulta: str, nr: Optional[str]) -> Dict[str, Any]:
    """Turno assistant que CHAMA a ferramenta (content vazio, sem thinking)."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                # SEMPRE as duas chaves (ver docstring do módulo).
                "arguments": {"consulta": consulta, "nr": nr},
            },
        }],
    }


def make_tool_result_message(content: str) -> Dict[str, Any]:
    return {"role": "tool", "content": content}


def make_assistant_message(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


def format_tool_result(chunks: List[Dict[str, Any]]) -> str:
    """Formata o retorno da busca (== render_user do F0: '[i] (fonte):\ntexto')."""
    parts = []
    for i, c in enumerate(chunks, 1):
        src = f"{c['doc']} - {c['section']}"
        if c.get("title"):
            src += f" - {c['title']}"
        parts.append(f"[{i}] ({src}):\n{c['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tools BAKED no system (o TRL aplica o chat template SEM tools=; a paridade
# byte a byte entre 'system simples + tools=' e 'system baked sem tools=' é
# provada em verify_format.py). Replica o template: ns.system_prompt +
# "\n\n<tools>\n" + tool_to_json(fn) por linha + "\n</tools>".
# ---------------------------------------------------------------------------
def _tool_to_json(fn: Dict[str, Any]) -> str:
    # mesmo comportamento do macro tool_to_json do template (ordem de inserção,
    # sem defer_loading), com json.dumps estilo tojson do jinja (', '/': ').
    items = [f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
             for k, v in fn.items() if k != "defer_loading"]
    return "{" + ", ".join(items) + "}"


def baked_system_content() -> str:
    lines = [_tool_to_json(t["function"]) for t in TOOLS]
    return SYSTEM_PROMPT_F4 + "\n\n<tools>\n" + "\n".join(lines) + "\n</tools>"


def make_baked_system_message() -> Dict[str, Any]:
    return {"role": "system", "content": baked_system_content()}


if __name__ == "__main__":
    print(json.dumps(TOOL_BUSCAR_NORMA, ensure_ascii=False, indent=2))
    print("\n--- SYSTEM F4 ---\n" + SYSTEM_PROMPT_F4)
    print("\n--- BAKED ---\n" + baked_system_content())
