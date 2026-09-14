#!/usr/bin/env python3
"""
infer_tools.py — Loop de inferência TOOL-CALLING (decisão -> chamada -> busca -> síntese).

Serve um GGUF via llama-server e conduz o diálogo EXATAMENTE como o app faria:
  1. Renderiza system(+tools) + histórico + add_generation_prompt (nosso render).
  2. Gera com o /completion RAW (controle total do prompt = igual ao treino).
  3. Se a saída tem <|tool_call_start|>[buscar_norma(...)]<|tool_call_end|>:
        - parseia (consulta, nr);
        - EXECUTA a busca no índice v4 (BM25 top-2; nr vira filtro-duro quando presente,
          espelhando o gate do app);
        - injeta o turno role 'tool' com os trechos e gera de novo -> resposta final.
     Senão: a saída é a resposta direta (sem ferramenta).

Usa /completion (texto cru) e NÃO /chat/completions: assim o prompt é byte-a-byte o do
treino (render_jinja), sem o llama-server reaplicar um template. Sampling Liquid
(temp 0.1 / top_k 50 / repeat 1.05), determinístico opcional.

Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import render_jinja as RJ
import retrieval_check as RC
from render import parse_tool_calls
from tool_schema import (TOOLS, format_tool_result, make_assistant_message,
                         make_system_message, make_tool_call_message,
                         make_tool_result_message, make_user_message)

LFM_SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}
DET_SAMPLING = {"temperature": 0.0}
STOP = ["<|im_end|>"]


def _complete(url: str, prompt: str, max_tokens: int, sampling: Dict[str, Any],
              timeout: int = 240) -> Dict[str, Any]:
    """Chama /completion (texto cru) do llama-server. Retorna content + usage."""
    payload = {"prompt": prompt, "n_predict": max_tokens, "stop": STOP,
               "cache_prompt": True, **sampling}
    r = requests.post(f"{url}/completion", json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return {"content": d.get("content", ""),
            "tokens_predicted": d.get("tokens_predicted", 0),
            "tokens_evaluated": d.get("tokens_evaluated", 0)}


def _search_v4(consulta: str, nr: Optional[str], topk: int = 2) -> List[Dict[str, Any]]:
    """Executa a busca do app: BM25 v4 top-K; nr (quando presente) vira filtro-duro de doc."""
    ret = RC._retriever()
    from corpus.eval_fino import app_fts_query  # via retrieval3 path já inserido em RC
    q = app_fts_query(consulta)
    doc_filter = None
    if nr:
        d = nr.strip().lower()
        doc_filter = [d]
    res = ret.search(q, topk, doc_filter=doc_filter)
    if not res and doc_filter:  # fallback sem filtro (gate suave)
        res = ret.search(q, topk)
    return res


def run_turn(url: str, history: List[Dict[str, Any]], user_text: str,
             sampling: Dict[str, Any], max_tokens: int = 384, topk: int = 2,
             timeout: int = 240) -> Dict[str, Any]:
    """Roda UM turno do usuário. Retorna dict com decisão, chamada, trechos e resposta final.

    history = lista de mensagens dos turnos ANTERIORES (sem system; o system é injetado aqui).
    Retorna também o history ATUALIZADO (com o turno completo) para o próximo turno.
    """
    msgs: List[Dict[str, Any]] = [make_system_message()] + history + [make_user_message(user_text)]
    prompt = RJ.render(msgs, tools=TOOLS, add_generation_prompt=True)
    first = _complete(url, prompt, max_tokens, sampling, timeout)
    raw = first["content"]

    result: Dict[str, Any] = {
        "user": user_text, "called_tool": False, "syntactic_ok": False,
        "consulta": None, "nr": None, "retrieved": [], "answer": "",
        "raw_first": raw, "tokens": first["tokens_predicted"],
    }

    calls: List[Dict[str, Any]] = []
    if "<|tool_call_start|>" in raw:
        try:
            calls = parse_tool_calls(raw)
            result["syntactic_ok"] = len(calls) >= 1 and calls[0]["name"] == "buscar_norma"
        except Exception:
            result["syntactic_ok"] = False
        result["called_tool"] = True

    new_history = list(history) + [make_user_message(user_text)]

    if result["called_tool"] and result["syntactic_ok"]:
        args = calls[0]["arguments"]
        consulta = str(args.get("consulta", "")).strip()
        nr = args.get("nr")
        nr = str(nr).strip() if nr else None
        result["consulta"], result["nr"] = consulta, nr
        chunks = _search_v4(consulta, nr, topk)
        result["retrieved"] = [{"id": c["id"], "doc": c["doc"], "section": c["section"]}
                               for c in chunks]
        # 2ª passada: injeta tool_call + turno tool e gera a resposta final
        new_history.append(make_tool_call_message(consulta, nr))
        new_history.append(make_tool_result_message(format_tool_result(chunks)))
        msgs2 = [make_system_message()] + new_history
        prompt2 = RJ.render(msgs2, tools=TOOLS, add_generation_prompt=True)
        second = _complete(url, prompt2, max_tokens, sampling, timeout)
        answer = second["content"].strip()
        result["answer"] = answer
        result["tokens"] += second["tokens_predicted"]
        new_history.append(make_assistant_message(answer))
    else:
        # sem ferramenta (resposta direta) — remove qualquer resíduo de tool tokens
        answer = raw.split("<|tool_call_start|>")[0].strip() if result["called_tool"] else raw.strip()
        result["answer"] = answer
        new_history.append(make_assistant_message(answer))

    result["history"] = new_history
    return result


def run_dialog(url: str, user_turns: List[str], sampling: Dict[str, Any],
               max_tokens: int = 384, topk: int = 2) -> List[Dict[str, Any]]:
    """Roda um diálogo multiturno completo. Retorna a lista de resultados por turno."""
    history: List[Dict[str, Any]] = []
    out = []
    for ut in user_turns:
        r = run_turn(url, history, ut, sampling, max_tokens, topk)
        history = r.pop("history")
        out.append(r)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--turn", action="append", required=True, help="fala do usuário (repetível)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    sampling = DET_SAMPLING if args.deterministic else LFM_SAMPLING
    for i, r in enumerate(run_dialog(args.url, args.turn, sampling)):
        print(f"=== turno {i+1}: {r['user']!r}")
        print(f"  chamou={r['called_tool']} sintaxe_ok={r['syntactic_ok']} "
              f"consulta={r['consulta']!r} nr={r['nr']!r}")
        print(f"  trechos={r['retrieved']}")
        print(f"  resposta: {r['answer']}")
