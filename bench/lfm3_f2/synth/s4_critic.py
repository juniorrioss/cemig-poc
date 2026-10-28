"""
s4_critic.py — estágio (d) do funil: segunda passada do 27B como CRÍTICO.

Avalia cada QA candidato (expand + socratic) em 3 eixos binários:
  - fiel      : a resposta é 100% fiel ao fato fonte (números incluídos)?
  - autossuf  : a resposta responde a pergunta por completo, sozinha?
  - respondivel: a pergunta é respondível SEM a fonte na frente (não depende
                 de "o trecho acima", "o contexto", etc.)?

QA passa somente com 3/3. Reprovados são mantidos em arquivo próprio para
auditoria (nunca descartados em silêncio).

Saída: data/synth/qa_passed_nr{N}.jsonl + qa_failed_nr{N}.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, gen_json, parallel_map, read_jsonl,
                          treated_norms, write_jsonl)

SYSTEM = (
    "Você é um revisor técnico rigoroso de material de treino sobre Normas "
    "Regulamentadoras. Avalie com rigor e responda com um bloco JSON ao final."
)

PROMPT = """FATO FONTE ({ref}):
{fato}

CANDIDATO A PAR DE TREINO:
Pergunta: {question}
Resposta: {answer}

Avalie o candidato em três critérios binários:
1. "fiel": a resposta é completamente fiel ao fato fonte? Nenhum número, prazo ou condição alterado/inventado (racional físico geral é aceitável; regra normativa inventada NÃO).
2. "autossuf": a resposta responde a pergunta por completo, sem depender de outro texto.
3. "respondivel": a pergunta faz sentido sozinha, sem a fonte na frente — não usa "o trecho", "o texto acima", "o contexto fornecido" nem pressupõe algo que só a fonte diz.

Pense brevemente e emita APENAS o JSON final:
```json
{{"fiel": true, "autossuf": true, "respondivel": true, "motivo": "..."}}
```"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else treated_norms()
    for nr in nrs:
        units = {u["unit_id"]: u for u in read_jsonl(SYNTH / f"units_nr{nr}.jsonl")}
        qas = (read_jsonl(SYNTH / f"qa_raw_nr{nr}.jsonl")
               + read_jsonl(SYNTH / f"qa_socratic_nr{nr}.jsonl"))
        if args.limit:
            qas = qas[: args.limit]
        print(f"[NR-{nr}] {len(qas)} QAs para criticar")
        ckpt = SYNTH / f"critic_nr{nr}.ckpt.json"

        def worker(qa: dict) -> dict:
            u = units[qa["unit_id"]]
            v = gen_json(PROMPT.format(ref=qa["ref"], fato=u["fato"],
                                       question=qa["question"], answer=qa["answer"]),
                         system=SYSTEM, temperature=0.0, max_tokens=1536)
            return {"fiel": bool(v.get("fiel")), "autossuf": bool(v.get("autossuf")),
                    "respondivel": bool(v.get("respondivel")),
                    "motivo": str(v.get("motivo", ""))[:300]}

        results = parallel_map(worker, qas, max_workers=args.workers,
                               checkpoint_path=ckpt, key_fn=lambda q: q["qa_id"])
        passed, failed, n_err = [], [], 0
        for qa in qas:
            v = results.get(qa["qa_id"])
            if not v or "__error__" in v:
                n_err += 1
                continue
            rec = dict(qa, critic=v)
            if v["fiel"] and v["autossuf"] and v["respondivel"]:
                passed.append(rec)
            else:
                failed.append(rec)
        write_jsonl(SYNTH / f"qa_passed_nr{nr}.jsonl", passed)
        write_jsonl(SYNTH / f"qa_failed_nr{nr}.jsonl", failed)
        rate = len(passed) / max(1, len(passed) + len(failed))
        print(f"[ok] NR-{nr}: {len(passed)} aprovados ({rate:.1%}), "
              f"{len(failed)} reprovados, {n_err} erros")


if __name__ == "__main__":
    main()
