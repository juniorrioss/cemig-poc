#!/usr/bin/env python3
"""
tool_schema.py — Contrato ÚNICO da ferramenta e renderização NATIVA do LFM2.5.

Decisões do capitão (fechadas, não reabrir):
  - Assinatura ENXUTA: buscar_norma(consulta: str, nr: str|None). Sem top_k, sem tipo.
  - 'consulta' = a REESCRITA da fala crua do operário em termos de busca (fala de campo ->
    termos técnicos + números normalizados). É o PRODUTO PRINCIPAL deste treino.
  - 'nr' = a norma quando o usuário cita ou quando é inequívoca; None caso contrário.
  - NÃO chamar a ferramenta é comportamento válido (não existe tool 'nenhuma').

Emenda 2 do brief (crítica): a RENDERIZAÇÃO NO FORMATO LFM É NOSSA e DETERMINÍSTICA.
Montamos o dict de messages (role assistant com tool_calls; role tool com content) e
deixamos o PRÓPRIO tokenizer.apply_chat_template do LFM2.5 emitir os tokens. Zero regex,
zero string concatenada à mão — o template oficial já escapa aspas, monta a lista e marca
a região de geração ({% generation %} -> assistant_only_loss nativo). Se o template mudar
numa versão futura, o código continua correto.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# O contrato JSON-Schema da ferramenta, EXATAMENTE como vai para o system em produção.
# A lista de tools entra no system como JSON em texto (o template faz: "List of tools: [...]").
# ATENÇÃO (emenda 2): o system + tools usados no TREINO têm de ser IDÊNTICOS aos de produção.
# ---------------------------------------------------------------------------
TOOL_BUSCAR_NORMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "buscar_norma",
        "description": (
            "Busca trechos das Normas Regulamentadoras (NRs) de segurança do trabalho para "
            "fundamentar a resposta ao eletricista. Chame SOMENTE quando precisar consultar "
            "uma norma para responder; NÃO chame para saudações, agradecimentos, pedidos de "
            "repetição, bom senso geral ou quando o contexto já consultado nesta conversa "
            "responde à nova pergunta."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": (
                        "A fala do eletricista REESCRITA em termos técnicos de busca: converta "
                        "a linguagem de campo em termos das NRs e normalize números (ex.: "
                        "'13,8 kV' e '13.8 kV'). Ex.: 'o poste não parece firme pra subir' -> "
                        "'estabilidade estrutural poste inspeção antes de subir'."
                    ),
                },
                "nr": {
                    "type": ["string", "null"],
                    "description": (
                        "A norma no formato 'NR-10' quando o eletricista a cita ou quando é "
                        "inequívoca pela pergunta; caso contrário null."
                    ),
                },
            },
            "required": ["consulta"],
        },
    },
}

TOOLS: List[Dict[str, Any]] = [TOOL_BUSCAR_NORMA]

# ---------------------------------------------------------------------------
# SYSTEM PROMPT de produção/treino (IDÊNTICO nos dois — exigência da emenda 2).
#
# Baseia-se no SYNTHESIS_SYSTEM_PROMPT_V2 do sft_v2 (voz, <=4 frases, sem markdown, cita
# norma, recusa útil) e ACRESCENTA a política de uso da ferramenta (quando chamar / quando
# não). A lista de tools NÃO é escrita aqui: o template a anexa a partir de tools=[...].
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TOOLS = (
    "Você é o assistente técnico de campo da CEMIG. O eletricista OUVE sua resposta por voz, "
    "então seja curto e direto.\n"
    "FERRAMENTA: você tem a ferramenta buscar_norma(consulta, nr) para consultar as Normas "
    "Regulamentadoras. Use-a para decidir sozinho quando buscar:\n"
    "- CHAME quando precisar de um dado normativo que ainda não está na conversa. No campo "
    "'consulta', REESCREVA a fala do eletricista em termos técnicos de busca e normalize "
    "números. Preencha 'nr' só quando a norma for citada ou inequívoca; senão deixe null.\n"
    "- NÃO CHAME para saudações, agradecimentos, pedidos de repetição, bom senso geral, "
    "assuntos fora do escopo das NRs, nem quando um trecho já consultado nesta conversa "
    "responde à nova pergunta (reuse o que está no contexto).\n"
    "REGRAS DA RESPOSTA FINAL (nunca viole):\n"
    "- Responda em NO MÁXIMO 4 frases curtas.\n"
    "- PROIBIDO markdown, títulos, negrito, listas, bullets ou numeração. Escreva em prosa corrida.\n"
    "- Cite a norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...').\n"
    "- Baseie-se EXCLUSIVAMENTE no contexto normativo consultado; não invente procedimentos, "
    "valores nem itens.\n"
    "- Se o contexto trouxer só PARTE da resposta, responda o que der e diga com franqueza o "
    "que não dá para afirmar com o que foi consultado.\n"
    "- Se o contexto NÃO responder à pergunta, não invente: diga que isso não está nas normas "
    "consultadas e, em uma frase, o que seria preciso verificar."
)

TOOL_NAME = "buscar_norma"


# ---------------------------------------------------------------------------
# Construtores de mensagens (dicts) — o tokenizer é quem emite os tokens.
# ---------------------------------------------------------------------------
def make_tool_call_message(consulta: str, nr: Optional[str],
                           preface: str = "") -> Dict[str, Any]:
    """Turno assistant que CHAMA a ferramenta.

    'preface' é conteúdo textual opcional antes da chamada (normalmente vazio — o LFM
    emite a chamada direto). 'arguments' é um MAPPING (o template exige mapping, não string).
    Só inclui 'nr' quando não-nulo (assinatura enxuta; None = campo ausente).
    """
    arguments: Dict[str, Any] = {"consulta": consulta}
    if nr:
        arguments["nr"] = nr
    return {
        "role": "assistant",
        "content": preface,
        "tool_calls": [{
            "type": "function",
            "function": {"name": TOOL_NAME, "arguments": arguments},
        }],
    }


def make_tool_result_message(content: str) -> Dict[str, Any]:
    """Turno role='tool' com o conteúdo dos trechos recuperados (o retorno da busca)."""
    return {"role": "tool", "content": content}


def make_assistant_message(content: str) -> Dict[str, Any]:
    """Turno assistant SEM tool call (resposta direta ou síntese final)."""
    return {"role": "assistant", "content": content}


def make_user_message(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def make_system_message() -> Dict[str, Any]:
    return {"role": "system", "content": SYSTEM_PROMPT_TOOLS}


def format_tool_result(chunks: List[Dict[str, Any]]) -> str:
    """Formata os trechos recuperados como conteúdo do turno 'tool' (== AskPipeline)."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    print(json.dumps(TOOL_BUSCAR_NORMA, ensure_ascii=False, indent=2))
    print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT_TOOLS)
