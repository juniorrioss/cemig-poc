"""
s3_socratic.py — estágio (c) do funil: método socrático multi-chamada para
unidades DENSAS (múltiplas condições/exceções/valores).

Cadeia de 4 chamadas INDEPENDENTES ao 27B por unidade densa (o capitão mandou
usar múltiplas chamadas à vontade):
  1. resposta   : pergunta de aplicação + resposta
  2. por_que    : follow-up "por quê?" sobre a resposta 1 (racional da regra)
  3. e_se       : variação de condição ("e se X mudar?") explorando o limite
  4. limite     : exceção/fronteira da regra (quando NÃO se aplica)

Cada passo vê o histórico anterior e deve PERMANECER fiel ao fato fonte.
O resultado vira QAs independentes (single-turn, estilo pré-treino) com origem
"socratic:nivel".

Saída: data/synth/qa_socratic_nr{N}.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, gen_json, parallel_map, read_jsonl,
                          treated_norms, write_jsonl)

SYSTEM = (
    "Você escreve material socrático de estudo sobre Normas Regulamentadoras "
    "brasileiras, 100% fiel ao fato normativo fornecido. Responda sempre com "
    "um bloco JSON ao final."
)

STEP_PROMPTS = {
    "aplicacao": """FATO NORMATIVO ({ref}):
{fato}

Crie UMA pergunta de APLICAÇÃO prática deste fato (situação concreta de trabalho onde a regra decide algo) e a resposta correta. A pergunta não deve começar com "Qual", "O que" ou "Como". A resposta cita a norma e o item.

```json
{{"pergunta": "...", "resposta": "..."}}
```""",
    "por_que": """FATO NORMATIVO ({ref}):
{fato}

DIÁLOGO ANTERIOR:
P: {prev_q}
R: {prev_a}

Agora o interlocutor pergunta POR QUÊ — ele quer o racional de segurança por trás da regra. Formule a pergunta de "por quê" do jeito que um trabalhador perguntaria (não começar com "Por que é importante") e responda explicando o racional SEM inventar fatos além do fato normativo (pode usar conhecimento geral de segurança para o racional físico, mas a regra em si vem só do fato).

```json
{{"pergunta": "...", "resposta": "..."}}
```""",
    "e_se": """FATO NORMATIVO ({ref}):
{fato}

DIÁLOGO ANTERIOR:
P: {prev_q}
R: {prev_a}

Agora o interlocutor muda UMA condição da situação ("e se ...?") de um jeito que testa o entendimento fino da regra (outro valor, outro contexto, outra pessoa). Responda aplicando o fato normativo à nova condição; se o fato não cobrir a nova condição, a resposta deve dizer isso honestamente.

```json
{{"pergunta": "...", "resposta": "..."}}
```""",
    "limite": """FATO NORMATIVO ({ref}):
{fato}

Crie UMA pergunta sobre o LIMITE ou EXCEÇÃO desta regra — quando ela NÃO se aplica, o que fica de fora, ou qual o valor-fronteira exato — e a resposta fiel ao fato. Se o fato tiver valores numéricos, a pergunta deve girar em torno do valor exato.

```json
{{"pergunta": "...", "resposta": "..."}}
```""",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else treated_norms()
    for nr in nrs:
        units = [u for u in read_jsonl(SYNTH / f"units_nr{nr}.jsonl") if u["densa"]]
        if args.limit:
            units = units[: args.limit]
        print(f"[NR-{nr}] {len(units)} unidades densas")
        ckpt = SYNTH / f"qa_socratic_nr{nr}.ckpt.json"

        def worker(u: dict) -> list:
            rows = []
            # passo 1: aplicação
            step1 = gen_json(STEP_PROMPTS["aplicacao"].format(ref=u["ref"], fato=u["fato"]),
                             system=SYSTEM, temperature=0.7, max_tokens=2048)
            chain = [("aplicacao", step1)]
            # passos 2 e 3 encadeiam no anterior
            prev = step1
            for step in ("por_que", "e_se"):
                res = gen_json(STEP_PROMPTS[step].format(
                    ref=u["ref"], fato=u["fato"],
                    prev_q=prev["pergunta"], prev_a=prev["resposta"]),
                    system=SYSTEM, temperature=0.7, max_tokens=2048)
                chain.append((step, res))
                prev = res
            # passo 4: limite (independente)
            step4 = gen_json(STEP_PROMPTS["limite"].format(ref=u["ref"], fato=u["fato"]),
                             system=SYSTEM, temperature=0.7, max_tokens=2048)
            chain.append(("limite", step4))

            for level, qa in chain:
                q = (qa.get("pergunta") or "").strip()
                a = (qa.get("resposta") or "").strip()
                if not q or not a:
                    continue
                qa_id = hashlib.sha256(f"{u['unit_id']}:soc:{q}".encode()).hexdigest()[:16]
                rows.append({"qa_id": qa_id, "nr": nr, "unit_id": u["unit_id"],
                             "ref": u["ref"], "persona": "socratic",
                             "cenario": level, "style": "sem_ref",
                             "question": q, "answer": a,
                             "origin": f"socratic:{level}"})
            return rows

        results = parallel_map(worker, units, max_workers=args.workers,
                               checkpoint_path=ckpt, key_fn=lambda u: u["unit_id"])
        rows, n_err = [], 0
        for u in units:
            r = results.get(u["unit_id"])
            if isinstance(r, dict) and "__error__" in r:
                n_err += 1
                continue
            rows.extend(r or [])
        out = SYNTH / f"qa_socratic_nr{nr}.jsonl"
        write_jsonl(out, rows)
        print(f"[ok] {out.name}: {len(rows)} QAs, {n_err} erros")


if __name__ == "__main__":
    main()
