#!/usr/bin/env python3
"""
verify_format.py — PASSO 0 do F4: prova, ANTES de gerar dado, que:

  (1) o formato nativo de tool-calling do LFM3 é o XML <tool_call> emitido
      pelo apply_chat_template com tools=... (render de referência salvo em
      data/passo0_render.txt);
  (2) ROUND-TRIP: render da tool_call -> tool_parser.parse_tool_calls ->
      argumentos idênticos (acentos, vírgula decimal '13,8 kV', aspas,
      multi-linha, nr=None -> 'null');
  (3) PARIDADE BAKED: system BAKED (make_baked_system_message, sem tools=) ==
      system simples + tools=[...] BYTE A BYTE — é assim que o TRL treina;
  (4) assistant_only_loss: a máscara {% generation %} cobre a DECISÃO +
      ARGUMENTOS + SÍNTESE e NÃO cobre o turno tool (chunks fora da loss).

Uso: classifier/.venv/bin/python bench/lfm3_f4/verify_format.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tool_parser import parse_tool_calls  # noqa: E402
from tool_schema_f4 import (TOOLS, SYSTEM_PROMPT_F4, format_tool_result,  # noqa: E402
                            make_assistant_message, make_baked_system_message,
                            make_system_message, make_tool_call_message,
                            make_tool_result_message, make_user_message)

MODEL = "LiquidAI/LFM3-1.7B"


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)


def main() -> None:
    tok = get_tok()

    # --- (1)+(2) round-trip em casos difíceis -------------------------------
    hard_cases = [
        ("Qual a distância mínima para rede de 13,8 kV energizada?", "NR-10"),
        ("o poste tá meio bambo, posso subir assim mesmo?", None),
        ("EPI pra trabalho em altura acima de 2 m — o que diz a norma?", "NR-35"),
        ("preciso d'água no canteiro: quantos bebedouros por trabalhador?", None),
        ("sinalização de \"perigo\" em área classificada", "NR-26"),
        ("linha1\nlinha2 com quebra", None),
    ]
    renders = []
    for consulta, nr in hard_cases:
        msgs = [make_system_message(),
                make_user_message(consulta),
                make_tool_call_message(consulta, nr)]
        s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                    enable_thinking=False)
        calls = parse_tool_calls(s)
        assert len(calls) == 1, f"esperava 1 call, veio {len(calls)}: {s[-400:]}"
        got = calls[0]
        assert got["name"] == "buscar_norma", got
        assert got["arguments"].get("consulta") == consulta, (
            f"consulta divergiu:\n  in : {consulta!r}\n  out: {got['arguments'].get('consulta')!r}")
        assert got["arguments"].get("nr") == nr, (
            f"nr divergiu: in={nr!r} out={got['arguments'].get('nr')!r}")
        renders.append(s)
    print(f"[ok] round-trip em {len(hard_cases)} casos difíceis")

    # --- (3) paridade BAKED byte a byte -------------------------------------
    convo = [
        make_user_message("Qual a distância para 13,8 kV?"),
        make_tool_call_message("Qual a distância para 13,8 kV?", "NR-10"),
        make_tool_result_message(format_tool_result([
            {"doc": "nr-10", "section": "Anexo II", "title": "Zona de risco",
             "text": "Tabela de raios de zona de risco..."}])),
        make_assistant_message("Conforme a NR-10, Anexo II, o raio é 0,70 m."),
    ]
    with_tools = tok.apply_chat_template(
        [make_system_message()] + convo, tools=TOOLS, tokenize=False,
        enable_thinking=False)
    baked = tok.apply_chat_template(
        [make_baked_system_message()] + convo, tokenize=False,
        enable_thinking=False)
    if with_tools != baked:
        for i, (a, b) in enumerate(zip(with_tools, baked)):
            if a != b:
                print(f"DIVERGE no byte {i}:\n  tools=: ...{with_tools[max(0,i-80):i+80]!r}\n  baked : ...{baked[max(0,i-80):i+80]!r}")
                break
        raise SystemExit("PARIDADE BAKED FALHOU")
    print(f"[ok] paridade BAKED byte a byte ({len(baked)} chars)")

    # --- (4) máscara assistant_only -----------------------------------------
    out = tok.apply_chat_template(
        [make_baked_system_message()] + convo, tokenize=True,
        return_assistant_tokens_mask=True, return_dict=True,
        enable_thinking=False)
    ids, mask = out["input_ids"], out["assistant_masks"]
    spans, cur = [], None
    for i, m in enumerate(mask):
        if m and cur is None:
            cur = i
        if not m and cur is not None:
            spans.append((cur, i)); cur = None
    if cur is not None:
        spans.append((cur, len(mask)))
    masked_text = " ".join(tok.decode(ids[a:b]) for a, b in spans)
    unmasked_text = tok.decode([t for t, m in zip(ids, mask) if not m])
    assert "<tool_call>" in masked_text and "0,70 m" in masked_text, masked_text
    assert "Tabela de raios" not in masked_text, "turno tool DENTRO da máscara!"
    assert "Tabela de raios" in unmasked_text
    assert SYSTEM_PROMPT_F4[:40] not in masked_text, "system dentro da máscara!"
    print(f"[ok] máscara generation: {len(spans)} spans assistant; turno tool FORA da loss")

    # --- (5) multihop: duas calls na mesma conversa rendem/parseiam ---------
    convo2 = [
        make_user_message("A CIPA de uma obra com 120 trabalhadores precisa de quantos membros e qual o treinamento deles?"),
        make_tool_call_message("quantos membros CIPA 120 trabalhadores", "NR-5"),
        make_tool_result_message("[1] (nr-05 - 5.6):\ndimensionamento..."),
        make_tool_call_message("treinamento CIPA carga horária conteúdo", "NR-5"),
        make_tool_result_message("[1] (nr-05 - 5.7):\ntreinamento..."),
        make_assistant_message("Conforme a NR-5, itens 5.6 e 5.7, ..."),
    ]
    s2 = tok.apply_chat_template([make_baked_system_message()] + convo2,
                                 tokenize=False, enable_thinking=False)
    calls2 = parse_tool_calls(s2)
    assert len(calls2) == 2 and calls2[0]["arguments"]["consulta"] != calls2[1]["arguments"]["consulta"]
    print("[ok] multihop: 2 calls renderizam e re-parseiam, argumentos distintos")

    # --- registro ------------------------------------------------------------
    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "passo0_render.txt").write_text(
        "=== conversa 1 tool (baked) ===\n" + baked +
        "\n\n=== conversa multihop (baked) ===\n" + s2 +
        "\n\n=== exemplos round-trip (com tools=) ===\n" + "\n\n".join(renders),
        encoding="utf-8")
    print(f"[ok] render de referência salvo em {out_dir / 'passo0_render.txt'}")
    print("PASSO 0: TUDO OK")


if __name__ == "__main__":
    main()
