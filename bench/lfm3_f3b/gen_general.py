#!/usr/bin/env python3
"""
gen_general.py — pares de instrução GERAL PT-BR (não-domínio) para o eixo 6
do sweep F3b (âncora anti-esquecimento 25%). Gerados pelo 27B em 2 passos:
(1) prompts diversos por categoria (mesmas 4 famílias do probe congelado:
instruções genéricas, resumo/reescrita, QA factual, conversa banal) e
(2) respostas curtas e corretas.

ANTICONTAMINAÇÃO: o probe (bench/lfm3_f2/probe/ood_general.json +
ood_behavioral.json) é régua CONGELADA — todo prompt gerado com Jaccard >=0.5
de tokens vs qualquer prompt do probe é DESCARTADO. Sem isso o eixo 6 viraria
treino-no-teste.

Saída: data/general/general_pairs.jsonl  {id, category, prompt, answer}
Uso:   classifier/.venv/bin/python bench/lfm3_f3b/gen_general.py --n 1600
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
from common_synth import gen_json, parallel_map, write_jsonl  # noqa: E402

PROBE_DIR = REPO / "bench" / "lfm3_f2" / "probe"
OUT = HERE / "data" / "general"
SEED = 20260213

CATEGORIES = {
    "instrucoes_genericas": (
        "instruções genéricas do dia a dia: explicar conceitos simples "
        "(ciência, tecnologia, finanças pessoais, culinária, esportes), dar "
        "passos de uma tarefa, comparar duas coisas"),
    "resumo_reescrita": (
        "resumo ou reescrita: o prompt TRAZ um parágrafo curto (3-6 frases, "
        "invente o conteúdo: notícia, e-mail, texto escolar) e pede resumo em "
        "1 frase, reescrita formal/informal, ou correção"),
    "qa_factual": (
        "perguntas factuais de conhecimento geral (geografia, história, "
        "biologia, matemática básica, cultura brasileira) com resposta curta "
        "e verificável"),
    "conversa_banal": (
        "conversa cotidiana: pedidos de opinião leve, small talk, conselhos "
        "práticos simples, mensagens de felicitação"),
}

PROMPT_GEN = """Gere {n} prompts de usuário DIVERSOS em português do Brasil na categoria:
{desc}

Regras:
- NADA de segurança do trabalho, normas regulamentadoras, eletricidade ou mineração.
- Variedade real de tema, comprimento e registro; nada de repetir estrutura.
- Prompts auto-contidos (se a tarefa precisa de um texto, inclua o texto).

Pense primeiro e emita APENAS o JSON final no formato
{{"prompts": ["...", "..."]}}"""

ANSWER_SYSTEM = (
    "Você é um assistente prestativo em português do Brasil. Responda de "
    "forma correta, natural, completa e concisa (2 a 6 frases, ou o formato "
    "que a tarefa pedir). Responda SOMENTE com um JSON: "
    "{\"resposta\": \"...\"}")


def norm_tokens(s: str) -> set[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return {t for t in "".join(c if c.isalnum() else " " for c in s).split()
            if len(t) > 2}


def load_probe_tokens() -> list[set[str]]:
    out = []
    for name in ("ood_general.json", "ood_behavioral.json"):
        data = json.loads((PROBE_DIR / name).read_text(encoding="utf-8"))
        for it in data["items"]:
            out.append(norm_tokens(it["prompt"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    per_cat = args.n // len(CATEGORIES) + 40      # margem p/ dedup
    batch = 25

    # passo 1: prompts (com sobra), em lotes por categoria
    tasks = []
    for cat, desc in CATEGORIES.items():
        for b in range(per_cat // batch + 1):
            tasks.append({"cat": cat, "desc": desc, "batch": b})

    def gen_prompts(t: dict) -> dict:
        v = gen_json(PROMPT_GEN.format(n=batch, desc=t["desc"]),
                     temperature=1.0, max_tokens=4096)
        # o 27B às vezes emite a lista crua em vez de {"prompts": [...]}
        if isinstance(v, list):
            v = {"prompts": v}
        return {"prompts": [p for p in v.get("prompts", []) if isinstance(p, str)]}

    res = parallel_map(gen_prompts, tasks, max_workers=args.workers,
                       checkpoint_path=OUT / "prompts.ckpt.json",
                       key_fn=lambda t: f"{t['cat']}:{t['batch']}")

    probe_toks = load_probe_tokens()
    rng = random.Random(SEED)
    seen: set[str] = set()
    pool: list[dict] = []
    n_contam = 0
    for t in tasks:
        r = res.get(f"{t['cat']}:{t['batch']}")
        if not r or "__error__" in r:
            continue
        for p in r["prompts"]:
            key = " ".join(p.lower().split())
            if key in seen or len(p) < 10:
                continue
            seen.add(key)
            pt = norm_tokens(p)
            if any(len(pt & bt) / max(1, len(pt | bt)) >= 0.5 for bt in probe_toks):
                n_contam += 1
                continue
            pool.append({"category": t["cat"], "prompt": p})
    rng.shuffle(pool)
    pool = pool[: args.n]
    print(f"[prompts] {len(pool)} únicos ({n_contam} contaminados descartados)")

    # passo 2: respostas
    def answer(it: dict) -> dict:
        v = gen_json(it["prompt"], system=ANSWER_SYSTEM,
                     temperature=0.4, max_tokens=2048)
        if isinstance(v, list):
            v = v[0] if v and isinstance(v[0], dict) else {}
        resp = str(v.get("resposta", "")).strip() if isinstance(v, dict) else ""
        if len(resp) < 15:
            raise ValueError("resposta curta demais")
        return {"answer": resp}

    for i, it in enumerate(pool):
        it["id"] = f"f3b-gen-{i:04d}"
    res2 = parallel_map(answer, pool, max_workers=args.workers,
                        checkpoint_path=OUT / "answers.ckpt.json",
                        key_fn=lambda it: it["id"])
    rows = []
    for it in pool:
        v = res2.get(it["id"])
        if v and "__error__" not in v:
            rows.append({**it, **v})
    write_jsonl(OUT / "general_pairs.jsonl", rows)
    import collections
    print(f"[ok] general_pairs.jsonl: {len(rows)}",
          dict(collections.Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
