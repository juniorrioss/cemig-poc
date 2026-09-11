#!/usr/bin/env python3
"""
sanity_ptbr.py — Sanity de PT-BR do MiniCPM5 (2B e 1B) ANTES da bateria completa (ordem do
capitão). Faz 5 perguntas técnicas de domínio elétrico/NR e imprime a resposta para leitura
humana. Se o PT-BR for ruim a ponto de invalidar (mistura com inglês/chinês, texto
ininteligível), cortamos cedo e registramos.

Roda contra um llama-server já no ar (RTX 5070). Testa thinking-OFF (enable_thinking=false),
que é o modo candidato ao embarque (thinking-ON raciocina em inglês -> destrói o PT).

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

# 5 perguntas coloquiais de eletricista (fora do holdout — genéricas de sanity).
SANITY_QUESTIONS = [
    "O que eu preciso vestir pra trabalhar em rede energizada com segurança?",
    "Quando é obrigatório usar o cinto de segurança tipo paraquedista no poste?",
    "Como faço pra desligar a energia de um circuito antes de mexer nele?",
    "Preciso de treinamento pra trabalhar em altura? Quantas horas?",
    "O que é aterramento temporário e por que ele é importante?",
]


def ask(url: str, q: str, thinking: bool, max_tokens: int = 400) -> dict:
    payload = {
        "messages": [{"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    d = r.json()
    m = d["choices"][0]["message"]
    return {
        "content": (m.get("content") or "").strip(),
        "reasoning": (m.get("reasoning_content") or "").strip(),
        "tokens": d.get("usage", {}).get("completion_tokens", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8792")
    ap.add_argument("--label", default="MiniCPM5-2B")
    ap.add_argument("--thinking", action="store_true", help="Liga o thinking (default OFF).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    print(f"===== SANITY PT-BR: {args.label} (thinking={'ON' if args.thinking else 'OFF'}) =====")
    for i, q in enumerate(SANITY_QUESTIONS, 1):
        r = ask(args.url, q, args.thinking)
        results.append({"q": q, **r})
        print(f"\n[{i}] PERGUNTA: {q}")
        if r["reasoning"]:
            print(f"    [THINKING {len(r['reasoning'].split())}w]: {r['reasoning'][:180]}...")
        print(f"    RESPOSTA ({r['tokens']} tok): {r['content']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "thinking": args.thinking, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
