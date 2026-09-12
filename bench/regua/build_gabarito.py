"""
build_gabarito.py — PASSO 1 da régua honesta.

Para cada uma das 151 do holdout, extrai do TRECHO-OURO (índice canônico) uma
lista curta de FATOS OBRIGATÓRIOS: itens concretos que qualquer resposta correta
precisa conter. Uma passada PARALELA do juiz 27B sobre os chunks-ouro (nunca sobre
o holdout como treino), com revisão automática de formato (JSON estrito, 2-6 fatos
curtos e atômicos). Salva bench/regua/data/gabarito_151.jsonl com procedência.

REGRA CRÍTICA (ordem do capitão): não inventar fatos — extrair SOMENTE do texto
da norma (trecho-ouro). A golden_answer é fornecida ao juiz apenas como pista de
foco (o que a pergunta cobra), mas os fatos devem estar ancorados no trecho-ouro.

Uso:
    python3 bench/regua/build_gabarito.py [--workers 16] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import (GoldResolver, JUDGE_MODEL, JUDGE_URL, load_holdout,
                    judge_call, parallel_map)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "gabarito_151.jsonl"
CKPT_PATH = DATA_DIR / "gabarito_151.ckpt.json"

# Prompt de extração de fatos. Ancorado no TRECHO-OURO; a golden serve de foco.
EXTRACT_PROMPT = """Você é um auditor técnico de Normas Regulamentadoras brasileiras (NRs).
Sua tarefa: a partir do TRECHO OFICIAL DA NORMA abaixo, extrair a lista dos FATOS OBRIGATÓRIOS que
QUALQUER resposta correta à pergunta do trabalhador precisa conter.

Um FATO OBRIGATÓRIO é um item concreto e verificável extraído LITERALMENTE do trecho da norma:
um equipamento, uma etapa de procedimento, uma condição, um valor/limite, uma exigência, um prazo.
Exemplo (pergunta sobre EPI de trabalho em altura): ["cinturão tipo paraquedista", "talabarte", "trava-quedas", "ponto de ancoragem"].

REGRAS ESTRITAS:
- Extraia SOMENTE fatos presentes no TRECHO DA NORMA. NÃO invente, NÃO use conhecimento externo.
- Cada fato deve ser CURTO (1 a 6 palavras), ATÔMICO (um item por entrada) e no VOCABULÁRIO da norma.
- Foque nos fatos que respondem à PERGUNTA. Ignore texto irrelevante do trecho.
- Produza de 2 a 6 fatos. Se o trecho só define um conceito, 1 ou 2 fatos bastam.
- Não inclua o número da norma/item como fato (a citação não conta).

[PERGUNTA DO TRABALHADOR]
{question}

[FOCO DA RESPOSTA CORRETA (referência, NÃO é fonte de fatos)]
{golden}

[TRECHO OFICIAL DA NORMA (ÚNICA FONTE DE FATOS)]
{gold_text}

Retorne EXCLUSIVAMENTE um objeto JSON estrito:
{{"fatos": ["fato 1", "fato 2", "..."]}}"""

# Fallback: quando o trecho-ouro do índice não cobre a subseção cobrada (o índice de
# 36 NRs é grosso e não tem todas as subseções), a fonte autorizada de fatos passa a
# ser a RESPOSTA-OURO oficial (referência derivada da norma na criação do dataset,
# NUNCA uma saída de modelo). Procedencia registrada como 'golden_answer'.
EXTRACT_PROMPT_GOLDEN = """Você é um auditor técnico de Normas Regulamentadoras brasileiras (NRs).
Sua tarefa: a partir da RESPOSTA-OURO OFICIAL abaixo, extrair a lista dos FATOS OBRIGATÓRIOS que
QUALQUER resposta correta à pergunta do trabalhador precisa conter.

Um FATO OBRIGATÓRIO é um item concreto e verificável: um equipamento, uma etapa de procedimento,
uma condição, um valor/limite, uma exigência, um prazo.

REGRAS ESTRITAS:
- Extraia SOMENTE fatos presentes na RESPOSTA-OURO. NÃO invente, NÃO use conhecimento externo.
- Cada fato deve ser CURTO (1 a 6 palavras), ATÔMICO e no VOCABULÁRIO da norma.
- Foque nos fatos que respondem à PERGUNTA. Produza de 2 a 6 fatos.
- Não inclua o número da norma/item como fato (a citação não conta).

[PERGUNTA DO TRABALHADOR]
{question}

[RESPOSTA-OURO OFICIAL (Única fonte de fatos)]
{golden}

Retorne EXCLUSIVAMENTE um objeto JSON estrito:
{{"fatos": ["fato 1", "fato 2", "..."]}}"""


def _extract_list(d: dict):
    """Aceita a chave em PT/EN/PT-PT ('fatos'/'facts'/'factos') e, em último caso,
    qualquer chave cujo valor seja uma lista de strings — o juiz oscila o nome."""
    for k in ("fatos", "facts", "factos"):
        if isinstance(d.get(k), list):
            return d[k]
    for v in d.values():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    return None


def _parse_facts(raw: str) -> list:
    """Extrai a lista de fatos do JSON do juiz, de trás para frente (evita thinking)."""
    text = raw.strip()
    # blocos ```json ... ```
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for b in reversed(blocks):
        try:
            lst = _extract_list(json.loads(b))
            if lst is not None:
                return _clean(lst)
        except Exception:
            pass
    # qualquer {...}
    for m in reversed(re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", text, re.DOTALL)):
        try:
            lst = _extract_list(json.loads(m))
            if lst is not None:
                return _clean(lst)
        except Exception:
            pass
    return []


def _clean(facts: list) -> list:
    """Normaliza: strings curtas, sem duplicatas, sem vazios."""
    out, seen = [], set()
    for f in facts:
        if not isinstance(f, str):
            continue
        s = re.sub(r"\s+", " ", f).strip().strip('."')
        if not s or len(s) > 80:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:6]


def build_one(resolver: GoldResolver, q: dict) -> dict:
    """Extrai fatos de uma pergunta a partir do seu trecho-ouro."""
    gold_chunks = resolver.resolve(q["doc"], q["section"])
    gold_text = "\n\n".join(c["text"] for c in gold_chunks)
    prompt = EXTRACT_PROMPT.format(
        question=q["question"], golden=q.get("golden_answer", ""), gold_text=gold_text)
    raw = judge_call(prompt, max_tokens=768, temperature=0.0)
    facts = _parse_facts(raw)
    return {
        "id": q["id"],
        "doc": q["doc"],
        "section": q["section"],
        "question": q["question"],
        "facts": facts,
        "n_facts": len(facts),
        "gold_chunk_ids": [c["id"] for c in gold_chunks],
        "gold_sections": [c["section"] for c in gold_chunks],
        "provenance": {
            "source": "gold_chunk_text(index_hf_36nr.db)",
            "judge_model": JUDGE_MODEL,
            "judge_url": JUDGE_URL,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    holdout = load_holdout()
    if args.limit:
        holdout = holdout[:args.limit]

    resolver = GoldResolver()
    # Resolvedor NÃO é thread-safe (uma conexão sqlite); pré-resolvo os trechos
    # em série (rápido) e a chamada ao juiz roda em paralelo.
    prepared = []
    for q in holdout:
        gc = resolver.resolve(q["doc"], q["section"])
        prepared.append({**q, "_gold_chunks": gc})

    existing = {}
    if CKPT_PATH.exists():
        existing = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
        print(f"[ckpt] retomando com {len(existing)} já extraídos")

    def worker(q: dict) -> dict:
        gold_text = "\n\n".join(c["text"] for c in q["_gold_chunks"])
        prompt = EXTRACT_PROMPT.format(
            question=q["question"], golden=q.get("golden_answer", ""), gold_text=gold_text)
        raw = judge_call(prompt, max_tokens=768, temperature=0.0)
        facts = _parse_facts(raw)
        return {
            "id": q["id"], "doc": q["doc"], "section": q["section"],
            "question": q["question"], "facts": facts, "n_facts": len(facts),
            "gold_chunk_ids": [c["id"] for c in q["_gold_chunks"]],
            "gold_sections": [c["section"] for c in q["_gold_chunks"]],
            "provenance": {"source": "gold_chunk_text(index_hf_36nr.db)",
                           "judge_model": JUDGE_MODEL, "judge_url": JUDGE_URL},
        }

    results = parallel_map(worker, prepared, max_workers=args.workers,
                           checkpoint_path=CKPT_PATH, key_fn=lambda q: q["id"],
                           existing=existing)

    # Revisão automática de formato: reprocessa itens que voltaram vazios.
    empties = [q for q in prepared if not results.get(q["id"], {}).get("facts")]
    if empties:
        print(f"[revisão] {len(empties)} itens sem fatos; retry serial (trecho-ouro)")
        for q in empties:
            try:
                results[q["id"]] = worker(q)
            except Exception as e:  # noqa: BLE001
                print(f"  falha {q['id']}: {e}")

    # Fallback golden_answer: se o trecho-ouro do índice grosso não cobre a subseção,
    # extrai da resposta-ouro oficial (procedencia distinta, registrada).
    still = [q for q in prepared if not results.get(q["id"], {}).get("facts")]
    if still:
        print(f"[fallback] {len(still)} itens sem fatos; extraindo da resposta-ouro")
        for q in still:
            try:
                prompt = EXTRACT_PROMPT_GOLDEN.format(
                    question=q["question"], golden=q.get("golden_answer", ""))
                raw = judge_call(prompt, max_tokens=768, temperature=0.0)
                facts = _parse_facts(raw)
                results[q["id"]] = {
                    "id": q["id"], "doc": q["doc"], "section": q["section"],
                    "question": q["question"], "facts": facts, "n_facts": len(facts),
                    "gold_chunk_ids": [c["id"] for c in q["_gold_chunks"]],
                    "gold_sections": [c["section"] for c in q["_gold_chunks"]],
                    "provenance": {"source": "golden_answer(qa_pairs_v2.jsonl)",
                                   "reason": "gold chunk grosso nao cobre subsecao",
                                   "judge_model": JUDGE_MODEL, "judge_url": JUDGE_URL},
                }
            except Exception as e:  # noqa: BLE001
                print(f"  falha fallback {q['id']}: {e}")

    # Ordena por id e grava JSONL final com procedência.
    ordered = [results[q["id"]] for q in prepared if q["id"] in results]
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_facts = [r["n_facts"] for r in ordered]
    print(f"[ok] gabarito salvo em {OUT_PATH}")
    print(f"     {len(ordered)} itens | média {sum(n_facts)/max(1,len(n_facts)):.2f} fatos/item "
          f"| vazios={sum(1 for n in n_facts if n==0)}")


if __name__ == "__main__":
    main()
