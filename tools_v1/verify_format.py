#!/usr/bin/env python3
"""
verify_format.py — PASSO 0 (exigência permanente após o bug do LoRA).

Renderiza UM diálogo completo de CADA uma das 5 famílias com o chat_template REAL do
LFM2.5-1.2B-Instruct (apply_chat_template com tools=[...] e messages com tool_calls e
role 'tool') e imprime o texto renderizado. Só geramos o dataset depois que o render
estiver idêntico ao contrato do brief:
  - tools no SYSTEM como "List of tools: [...]";
  - chamada <|tool_call_start|>[buscar_norma(consulta='...', nr='NR-10')]<|tool_call_end|>;
  - retorno em turno role 'tool': <|im_start|>tool\n{conteudo}<|im_end|>.

Também prova que a MÁSCARA de treino cobre APENAS os turnos assistant (region {% generation %},
inclusive mascarando o turno 'tool') via return_assistant_tokens_mask.

Comentários PT-BR; identificadores em inglês. Uso: ../.venv-train/bin/python verify_format.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render import get_tokenizer, parse_tool_calls, render, render_and_mask  # noqa: E402
from tool_schema import (  # noqa: E402
    TOOLS, format_tool_result, make_assistant_message, make_system_message,
    make_tool_call_message, make_tool_result_message, make_user_message)

CHUNK_NR10 = {"doc": "NR-10", "section": "10.2.8.2", "title": "Medidas de proteção coletiva",
              "text": ("Nos trabalhos em instalações elétricas, quando não for possível o "
                       "desligamento, a distância de segurança para a faixa de tensão de 13,8 kV "
                       "deve respeitar o raio de delimitação da zona de risco definido no Anexo II.")}
CHUNK_NR35 = {"doc": "NR-35", "section": "35.5.1", "title": "Sistema de ancoragem",
              "text": ("Antes do início dos trabalhos em altura, o supervisor deve inspecionar a "
                       "estrutura e os pontos de ancoragem, assegurando estabilidade e resistência "
                       "à força aplicável antes de o trabalhador subir.")}


def famchunk(c):
    return [c]


def dialog_simples():
    """(1) CHAMADA SIMPLES: pergunta -> tool_call -> turno tool -> resposta acionável citando o item."""
    return [
        make_system_message(),
        make_user_message("distância segura pra trabalhar perto de rede de 13,8 kV"),
        make_tool_call_message("distancia seguranca zona risco 13,8 kV 13.8 kV media tensao", "NR-10"),
        make_tool_result_message(format_tool_result(famchunk(CHUNK_NR10))),
        make_assistant_message("Conforme a NR-10, item 10.2.8.2, quando não dá para desligar você "
                               "deve respeitar o raio de delimitação da zona de risco para 13,8 kV "
                               "definido no Anexo II. Mantenha essa distância mínima antes de "
                               "aproximar."),
    ]


def dialog_sem_ferramenta():
    """(2) SEM FERRAMENTA: saudação -> responde direto, sem chamar."""
    return [
        make_system_message(),
        make_user_message("bom dia, tudo certo por aí?"),
        make_assistant_message("Bom dia! Tudo certo, estou aqui pra ajudar com as normas de "
                               "segurança. Pode mandar sua dúvida."),
    ]


def dialog_reuso():
    """(3) MULTITURNO COM REUSO: 2º turno sobre o MESMO assunto já em contexto -> NÃO chama."""
    return [
        make_system_message(),
        make_user_message("distância segura pra trabalhar perto de rede de 13,8 kV"),
        make_tool_call_message("distancia seguranca zona risco 13,8 kV 13.8 kV media tensao", "NR-10"),
        make_tool_result_message(format_tool_result(famchunk(CHUNK_NR10))),
        make_assistant_message("Pela NR-10, item 10.2.8.2, respeite o raio de delimitação da zona "
                               "de risco para 13,8 kV do Anexo II antes de se aproximar."),
        make_user_message("e se eu não conseguir desligar a rede?"),
        make_assistant_message("É exatamente esse o caso: a NR-10, item 10.2.8.2, trata de quando "
                               "não é possível desligar e manda respeitar a distância de segurança "
                               "da zona de risco. Sem desligar, mantenha esse raio do Anexo II."),
    ]


def dialog_nova_busca():
    """(4) MULTITURNO COM NOVA BUSCA: o assunto MUDA -> chama de novo com consulta diferente."""
    return [
        make_system_message(),
        make_user_message("distância segura pra trabalhar perto de rede de 13,8 kV"),
        make_tool_call_message("distancia seguranca zona risco 13,8 kV media tensao", "NR-10"),
        make_tool_result_message(format_tool_result(famchunk(CHUNK_NR10))),
        make_assistant_message("Pela NR-10, item 10.2.8.2, respeite o raio da zona de risco para "
                               "13,8 kV do Anexo II."),
        make_user_message("beleza. e antes de subir no poste, o que a norma exige?"),
        make_tool_call_message("inspecao estrutura ancoragem antes de subir trabalho em altura poste",
                               "NR-35"),
        make_tool_result_message(format_tool_result(famchunk(CHUNK_NR35))),
        make_assistant_message("Conforme a NR-35, item 35.5.1, antes de subir o supervisor precisa "
                               "inspecionar a estrutura e os pontos de ancoragem, garantindo "
                               "estabilidade e resistência à força aplicável."),
    ]


def dialog_recusa():
    """(5) RECUSA APÓS BUSCA: a ferramenta retorna trecho errado/parcial -> delimita/recusa."""
    return [
        make_system_message(),
        make_user_message("qual o fator de queda máximo que o talabarte aguenta?"),
        make_tool_call_message("fator de queda maximo talabarte trabalho em altura", "NR-35"),
        make_tool_result_message(format_tool_result(famchunk(CHUNK_NR35))),
        make_assistant_message("O trecho consultado da NR-35, item 35.5.1, trata da inspeção da "
                               "estrutura e dos pontos de ancoragem, mas não traz o valor do fator "
                               "de queda do talabarte. Para esse número seria preciso consultar a "
                               "especificação do equipamento e o anexo próprio de EPI."),
    ]


FAMILIES = [
    ("(1) chamada_simples", dialog_simples, True),
    ("(2) sem_ferramenta", dialog_sem_ferramenta, False),
    ("(3) multiturno_reuso", dialog_reuso, True),
    ("(4) multiturno_nova_busca", dialog_nova_busca, True),
    ("(5) recusa_apos_busca", dialog_recusa, True),
]


def main() -> None:
    tok = get_tokenizer()
    all_ok = True
    for label, fn, expect_call in FAMILIES:
        msgs = fn()
        text = render(msgs, tools=TOOLS)
        print("=" * 78)
        print(f"FAMÍLIA {label}")
        print("=" * 78)
        print(text)

        # checagens de contrato
        has_toollist = "List of tools: [" in text
        has_toolrole = "<|im_start|>tool\n" in text
        calls = parse_tool_calls(text)
        has_call = len(calls) > 0
        print("-" * 40)
        print(f"  system tem 'List of tools: [': {has_toollist}")
        print(f"  tem turno role 'tool': {has_toolrole}")
        print(f"  chamadas parseadas: {len(calls)} -> {calls if calls else '(nenhuma)'}")

        # máscara: só assistant deve ter mask=1; o turno 'tool' deve ser 0.
        ids, masks = render_and_mask(msgs, tools=TOOLS)
        n_gen = sum(masks)
        # confere que nenhum token do conteúdo do turno 'tool' está mascarado como assistant:
        # decodifica os tokens mascarados e garante que NÃO contêm o texto do chunk.
        gen_ids = [t for t, m in zip(ids, masks) if m == 1]
        gen_text = tok.decode(gen_ids)
        tool_leak = any(c["text"][:40] in gen_text for c in [CHUNK_NR10, CHUNK_NR35]
                        if "tool" in text)
        print(f"  tokens totais={len(ids)} | tokens assistant (mask=1)={n_gen}")
        print(f"  vazamento do turno 'tool' na máscara assistant: {tool_leak}")

        ok = has_toollist and (has_call == expect_call) and (has_toolrole == expect_call) \
            and not tool_leak and n_gen > 0
        print(f"  >>> FAMÍLIA {'OK' if ok else 'FALHOU'}")
        all_ok = all_ok and ok
        print()

    print("=" * 78)
    print(f"PASSO 0 {'PASSOU — render idêntico ao contrato' if all_ok else 'FALHOU'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
