"""
s2_expand.py — estágio (b) do funil: expansão multi-persona × multi-cenário.

Para cada unidade de conhecimento, o 27B gera perguntas variando persona ×
cenário × estilo. ESTILO é a exigência central do capitão contra a mesmice:
cada pergunta nasce em DUAS variantes — COM referência explícita à norma e SEM
(remoção/adição de referência) — e o prompt proíbe começar perguntas do mesmo
jeito (lista de aberturas proibidas amostrada por chamada p/ forçar entropia
dos primeiros tokens).

Cada chamada sorteia (seed derivada do unit_id) um subconjunto de 3 personas ×
2 cenários, e pede 1 pergunta por combinação (6) × 2 variantes de referência
= até 12 QAs/unidade. A resposta é escrita pelo 27B APOIADA no fato (fiel).

Saída: data/synth/qa_raw_nr{N}.jsonl

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/synth/s2_expand.py [--nr 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, gen_json, parallel_map, read_jsonl,
                          treated_norms, write_jsonl)

PERSONAS = [
    ("eletricista", "eletricista de campo, fala coloquial, direto ao ponto"),
    ("tecnico_seg", "técnico de segurança do trabalho, vocabulário técnico"),
    ("engenheiro", "engenheiro responsável, linguagem formal e precisa"),
    ("auditor", "auditor/fiscal verificando conformidade, linguagem jurídica"),
    ("trabalhador_leigo", "trabalhador leigo, não conhece jargão, pergunta como fala na obra"),
    ("supervisor", "supervisor de equipe, preocupado com procedimento e prazo"),
    ("cipeiro", "membro da CIPA, meio leigo meio técnico, foco no coletivo"),
    ("comprador", "comprador/almoxarife especificando material ou EPI"),
]

CENARIOS = [
    ("duvida_direta", "dúvida direta sobre o que a norma exige"),
    ("situacao_campo", "situação concreta de campo descrita antes da pergunta"),
    ("incidente", "relato de incidente/quase-acidente que motiva a pergunta"),
    ("conformidade", "verificação de conformidade/checklist antes de liberar serviço"),
    ("treinamento", "preparação de treinamento/DDS, quer explicar para outros"),
    ("discussao", "discordância entre colegas sobre o que vale, pede o desempate"),
]

# Aberturas proibidas rotativas: o anti-padrão clássico ("Qual é...", "O que...")
FORBIDDEN_POOL = [
    "Qual é", "Qual a", "O que", "Quais são", "Como deve", "Segundo a norma",
    "De acordo com", "Conforme a", "É obrigatório", "Quando é", "Quem é",
    "Me diga", "Explique", "Pode me dizer", "Em que situação",
]

SYSTEM = (
    "Você escreve pares de pergunta-resposta de alta qualidade para treinar um "
    "assistente de segurança do trabalho brasileiro. Fidelidade total ao fato "
    "normativo fornecido. Responda sempre com um bloco JSON ao final."
)

PROMPT = """FATO NORMATIVO ({ref}):
{fato}

Gere perguntas e respostas sobre ESTE fato, uma por combinação persona×cenário abaixo. Depois, para cada pergunta, crie DUAS variantes: "com_ref" (a pergunta menciona explicitamente a norma/item) e "sem_ref" (a pergunta NÃO menciona norma nenhuma — como alguém que nem sabe qual norma trata disso).

COMBINAÇÕES (gere exatamente {n_combos}):
{combos}

REGRAS DE DIVERSIDADE (obrigatórias):
- As {n_combos} perguntas devem começar com palavras DIFERENTES entre si.
- NENHUMA pergunta pode começar com: {forbidden}.
- Varie o formato: pergunta direta, situação+pergunta, frase incompleta de campo, pedido de confirmação ("tô certo que..."), números concretos quando o fato tiver valores.
- A variante sem_ref deve soar natural, não uma cópia da com_ref com a norma apagada.

REGRAS DA RESPOSTA:
- Autossuficiente: responde completamente usando APENAS o fato acima.
- Preserva números/valores/prazos literalmente.
- Cita a norma e o item ({ref}) na resposta (mesmo quando a pergunta não cita).
- Tom adequado à persona (para leigo, explica o jargão).
- 2 a 5 frases.

Pense primeiro e emita APENAS o JSON final:
```json
[{{"persona": "...", "cenario": "...", "q_com_ref": "...", "q_sem_ref": "...", "resposta": "..."}}, ...]
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="limita unidades (smoke)")
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else treated_norms()
    for nr in nrs:
        units = read_jsonl(SYNTH / f"units_nr{nr}.jsonl")
        if args.limit:
            units = units[: args.limit]
        print(f"[NR-{nr}] {len(units)} unidades")
        ckpt = SYNTH / f"qa_raw_nr{nr}.ckpt.json"

        def worker(u: dict) -> list:
            rng = random.Random(int(u["unit_id"], 16))
            personas = rng.sample(PERSONAS, 3)
            cenarios = rng.sample(CENARIOS, 2)
            combos = [(p, c) for p in personas for c in cenarios]
            combo_txt = "\n".join(
                f"{i+1}. persona={p[0]} ({p[1]}); cenario={c[0]} ({c[1]})"
                for i, (p, c) in enumerate(combos))
            forbidden = ", ".join(f'"{f}"' for f in rng.sample(FORBIDDEN_POOL, 6))
            out = gen_json(PROMPT.format(ref=u["ref"], fato=u["fato"],
                                         n_combos=len(combos), combos=combo_txt,
                                         forbidden=forbidden),
                           system=SYSTEM, temperature=0.8, max_tokens=6144)
            if isinstance(out, dict):
                out = next((v for v in out.values() if isinstance(v, list)), [out])
            rows = []
            for item in out:
                if not isinstance(item, dict) or not item.get("resposta"):
                    continue
                for style, qkey in (("com_ref", "q_com_ref"), ("sem_ref", "q_sem_ref")):
                    q = (item.get(qkey) or "").strip()
                    if not q:
                        continue
                    qa_id = hashlib.sha256(
                        f"{u['unit_id']}:{q}".encode()).hexdigest()[:16]
                    rows.append({
                        "qa_id": qa_id, "nr": nr, "unit_id": u["unit_id"],
                        "ref": u["ref"], "persona": item.get("persona", ""),
                        "cenario": item.get("cenario", ""), "style": style,
                        "question": q, "answer": item["resposta"].strip(),
                        "origin": "expand",
                    })
            return rows

        results = parallel_map(worker, units, max_workers=args.workers,
                               checkpoint_path=ckpt,
                               key_fn=lambda u: u["unit_id"])
        rows, n_err = [], 0
        for u in units:
            r = results.get(u["unit_id"])
            if isinstance(r, dict) and "__error__" in r:
                n_err += 1
                continue
            rows.extend(r or [])
        out = SYNTH / f"qa_raw_nr{nr}.jsonl"
        write_jsonl(out, rows)
        print(f"[ok] {out.name}: {len(rows)} QAs de {len(units)} unidades, {n_err} erros")


if __name__ == "__main__":
    main()
