#!/usr/bin/env python3
"""
verify_2_6b.py — PASSO 0 (verificação prévia, exigência permanente do brief).

Prova, ANTES de treinar/avaliar, que a bateria de tool-calling do 2.6B é MAÇÃ-COM-MAÇÃ com a
do 1.2B:

  (1) o template do 2.6B renderiza system/user/tool/assistant e a tool_call
      `<|tool_call_start|>[buscar_norma(...)]<|tool_call_end|>` BYTE-A-BYTE idêntico ao 1.2B —
      a ÚNICA diferença é o sufixo `<think>` do generation prompt (o 2.6B é modelo de
      raciocínio). Com thinking_off, o generation prompt também fica idêntico.
  (2) round-trip da chamada (consulta/nr voltam idênticos, incluindo `13,8 kV`).
  (3) renderiza 1 exemplo de CADA família do dataset (tools_v1) com o template do 2.6B e
      cola no relatório (o brief pede: "renderize e cole um exemplo de cada família").

A confirmação dos NOMES DE MÓDULO do 2.6B (self_attn.{q,k,v,out}_proj 8x, feed_forward.
{w1,w2,w3} 30x, conv.* 22x) e o print de targeted_module_names/trainable% rodam na Spark
(o base_hf mora lá) via `verify_modules_2_6b.py`; este script cobre a parte de RENDER (local,
no classifier/.venv).

Uso: ../classifier/.venv/bin/python verify_2_6b.py
Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tools_v1"))

import render_jinja as RJ_12B  # noqa: E402 (template 1.2B)
import render_jinja_2_6b as RJ_26B  # noqa: E402 (template 2.6B)
from render import parse_tool_calls  # noqa: E402 (parser round-trip do tools_v1)
from tool_schema import (TOOLS, format_tool_result, make_assistant_message,  # noqa: E402
                         make_system_message, make_tool_call_message,
                         make_tool_result_message, make_user_message)

OUT = _HERE / "data" / "passo0_render.txt"
DATASET = _HERE.parent / "tools_v1" / "data" / "dialogs.jsonl"


def _rows(lines: List[str]) -> List[str]:
    return [f"  {l}" for l in lines]


def prove_byte_identity() -> List[str]:
    """(1) Compara 1.2B vs 2.6B nos corpos que importam para o treino/eval."""
    log: List[str] = ["=" * 78, "(1) BYTE-IDENTITY 1.2B vs 2.6B (corpos de treino/eval)", "=" * 78]

    # Um diálogo tool-calling completo (system+tools, user, tool_call, tool, assistant).
    msgs = [
        make_system_message(),
        make_user_message("qual a distância segura pra trabalhar perto de rede de 13,8 kV?"),
        make_tool_call_message("distância segura zona risco 13,8 kV 13.8 kV média tensão", "NR-10"),
        make_tool_result_message(format_tool_result([
            {"id": 1, "doc": "NR-10", "section": "Anexo II", "title": "Zona de risco",
             "text": "Faixa de tensão e raio de delimitação da zona de risco..."}])),
        make_assistant_message("A zona de risco para 13,8 kV exige distância conforme a tabela "
                               "do Anexo II da NR-10."),
    ]

    # (a) render COMPLETO (add_generation_prompt=False) — corpo dos turnos de treino/eval.
    body_12 = RJ_12B.render(msgs, tools=TOOLS, add_generation_prompt=False)
    body_26 = RJ_26B.render(msgs, tools=TOOLS, add_generation_prompt=False)
    ident_body = body_12 == body_26
    log.append(f"corpo completo (turnos) idêntico? {ident_body}  "
               f"(len 1.2B={len(body_12)} 2.6B={len(body_26)})")

    # (b) generation prompt COM thinking (default do 2.6B) — DEVE diferir (o 2.6B tem <think>).
    gp_12 = RJ_12B.render([make_system_message(), make_user_message("oi")], tools=TOOLS,
                          add_generation_prompt=True)
    gp_26_think = RJ_26B.render([make_system_message(), make_user_message("oi")], tools=TOOLS,
                                add_generation_prompt=True, thinking_off=False)
    gp_26_off = RJ_26B.render([make_system_message(), make_user_message("oi")], tools=TOOLS,
                              add_generation_prompt=True, thinking_off=True)
    log.append(f"generation prompt 2.6B COM <think> difere do 1.2B? {gp_12 != gp_26_think} "
               f"(esperado True — 2.6B é reasoning)")
    log.append(f"generation prompt 2.6B thinking_off == 1.2B? {gp_12 == gp_26_off} "
               f"(esperado True — thinking-OFF replica o 1.2B)")
    log.append(f"sufixo do 2.6B COM think: ...{gp_26_think[-40:]!r}")
    log.append(f"sufixo do 2.6B thinking_off: ...{gp_26_off[-40:]!r}")

    ok = ident_body and (gp_12 != gp_26_think) and (gp_12 == gp_26_off)
    log.append(f"CONCLUSÃO (1): corpos idênticos + só o generation prompt difere (thinking) => "
               f"maçã-com-maçã OK = {ok}")
    assert ident_body, "corpo dos turnos NÃO é idêntico entre 1.2B e 2.6B — pare!"
    assert gp_12 == gp_26_off, "generation prompt thinking_off NÃO bate o 1.2B — pare!"
    return log


def prove_roundtrip() -> List[str]:
    """(2) round-trip: consulta/nr voltam idênticos (escape, acento, vírgula decimal)."""
    log: List[str] = ["", "=" * 78, "(2) ROUND-TRIP da tool_call (template 2.6B)", "=" * 78]
    cases = [
        ("distância segurança zona risco 13,8 kV 13.8 kV média tensão", "NR-10"),
        ("estabilidade estrutural poste inspeção antes de subir", None),
        ("uso de camiseta de algodão rasgada vestimenta trabalho", "NR-10"),
        ("aspas 'simples' e vírgula, acento çãé no meio", None),
    ]
    all_ok = True
    for consulta, nr in cases:
        msgs = [make_system_message(), make_user_message("(rt)"),
                make_tool_call_message(consulta, nr)]
        text = RJ_26B.render(msgs, tools=TOOLS)
        calls = parse_tool_calls(text)
        ok = (len(calls) == 1 and calls[0]["name"] == "buscar_norma"
              and calls[0]["arguments"].get("consulta") == consulta
              and calls[0]["arguments"].get("nr", None) == (nr if nr else None))
        all_ok = all_ok and ok
        log.append(f"[{'OK ' if ok else 'FAIL'}] nr={nr!r} consulta={consulta!r}")
    log.append(f"CONCLUSÃO (2): round-trip 100% = {all_ok}")
    assert all_ok, "round-trip falhou — pare!"
    return log


def render_one_per_family() -> List[str]:
    """(3) 1 exemplo de cada família renderizado com o template do 2.6B (brief)."""
    from render import parse_tool_calls as _p  # noqa: F401
    log: List[str] = ["", "=" * 78, "(3) 1 EXEMPLO POR FAMÍLIA (template 2.6B, thinking_off)",
                      "=" * 78]
    if not DATASET.exists():
        log.append(f"[aviso] {DATASET} ausente (dataset gitignored); pule com `make data` no "
                   f"tools_v1 se quiser as amostras renderizadas.")
        return log
    seen: Dict[str, Dict[str, Any]] = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        d = json.loads(line)
        fam = d["meta"]["family"]
        if fam not in seen:
            seen[fam] = d
        if len(seen) == 5:
            break
    for fam, d in seen.items():
        # o dataset já tem tools no system (baked); renderiza com tools=None (idempotente).
        text = RJ_26B.render(d["messages"], tools=None, add_generation_prompt=False)
        log.append("")
        log.append("-" * 78)
        log.append(f"FAMÍLIA: {fam}   (src_doc={d['meta'].get('src_doc')})")
        log.append("-" * 78)
        log.append(text)
    return log


def main() -> None:
    log: List[str] = ["PASSO 0 — verificação do formato nativo do 2.6B (tool-calling)", ""]
    log += prove_byte_identity()
    log += prove_roundtrip()
    log += render_one_per_family()
    OUT.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log[:40]))
    print(f"\n[ok] relatório completo em {OUT}")


if __name__ == "__main__":
    main()
