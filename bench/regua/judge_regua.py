"""
judge_regua.py — Desempate do juiz 27B para a régua honesta (Passo 2, parte LLM).

Uma ÚNICA chamada do juiz por resposta que o passe determinístico marcou como
`needs_judge` (candidata a aprovação com fatos ambíguos) OU que possa alucinar.
O juiz recebe: fatos AMBÍGUOS (para dizer quais estão realmente presentes) e o
TRECHO-OURO (para dizer se a resposta CONTRADIZ a norma = alucinação).

Retorna JSON estrito: {"fatos_presentes": [...], "alucinacao": bool, "justificativa": "..."}.

Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from common import judge_call

JUDGE_PROMPT = """Você é um auditor técnico de Normas Regulamentadoras brasileiras (NRs).
Sua tarefa tem DOIS objetivos, avaliando a RESPOSTA DO MODELO contra a NORMA:

OBJETIVO 1 — Para cada FATO da lista abaixo, decida se ele está SEMANTICAMENTE PRESENTE
na resposta do modelo (mesmo com outras palavras/sinônimos). Seja rigoroso: só conte como
presente se a resposta realmente comunica aquele fato ao trabalhador.

OBJETIVO 2 — Diga se a resposta contém ALUCINAÇÃO: alguma afirmação que CONTRADIZ o trecho
oficial da norma ou inventa procedimento perigoso não amparado por ela. Omitir não é alucinar;
alucinar é afirmar algo ERRADO/contraditório.

[PERGUNTA DO TRABALHADOR]
{question}

[TRECHO OFICIAL DA NORMA]
{gold_text}

[FATOS A VERIFICAR]
{facts}

[RESPOSTA DO MODELO]
{answer}

Retorne EXCLUSIVAMENTE um objeto JSON estrito:
{{"fatos_presentes": ["fato exatamente como escrito na lista, se presente"], "alucinacao": false, "justificativa": "curta"}}"""


def _parse(raw: str) -> Dict[str, Any]:
    """Extrai o JSON do juiz, de trás para frente (evita rascunho de thinking)."""
    text = raw.strip()
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    cands += re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", text, re.DOTALL)
    for m in reversed(cands):
        try:
            d = json.loads(m)
            if "fatos_presentes" in d or "alucinacao" in d:
                return {
                    "fatos_presentes": d.get("fatos_presentes", []) or [],
                    "alucinacao": bool(d.get("alucinacao", False)),
                    "justificativa": str(d.get("justificativa", ""))[:300],
                }
        except Exception:
            pass
    return {"fatos_presentes": [], "alucinacao": False, "justificativa": "parse_fail"}


def judge_disambiguate(question: str, answer: str, ambiguous_facts: List[str],
                       gold_text: str) -> Dict[str, Any]:
    """Desempate único: quais fatos ambíguos estão presentes + houve alucinação?"""
    facts_block = "\n".join(f"- {f}" for f in ambiguous_facts) or "- (nenhum)"
    prompt = JUDGE_PROMPT.format(
        question=question, gold_text=gold_text[:3000],
        facts=facts_block, answer=answer[:2500])
    raw = judge_call(prompt, max_tokens=768, temperature=0.0)
    return _parse(raw)
