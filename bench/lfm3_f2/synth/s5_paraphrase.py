"""
s5_paraphrase.py — parafraseios do texto cru da norma (mistura estilo pré-treino).

Para cada janela do texto da norma tratada, o 27B produz parafraseios em
registros/níveis de formalidade distintos (sorteados por janela):
  - didatico   : professor explicando a matéria para turma de técnicos
  - informal   : veterano de campo explicando "na linguagem da obra"
  - resumo_tec : ficha técnica/resumo estruturado para consulta rápida
  - juridico   : parecer formal, estilo circular interna de conformidade

Fidelidade total (números literais). O texto CRU da janela também entra na
mistura final (s6) — aqui só geramos as variações.

Saída: data/synth/para_nr{N}.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, gen_call, load_norm_corpus, parallel_map,
                          treated_norms, write_jsonl)
from s1_decompose import split_windows

REGISTROS = {
    "didatico": "um professor de curso técnico explicando esta parte da norma para a turma, com exemplos curtos",
    "informal": "um veterano de campo explicando essas regras para um colega novato, na linguagem falada da obra (sem gíria excessiva, mas coloquial)",
    "resumo_tec": "uma ficha técnica de consulta rápida: tópicos curtos, valores e prazos em destaque, zero floreio",
    "juridico": "uma circular interna de conformidade, tom formal, citando itens da norma",
}

SYSTEM = (
    "Você reescreve trechos de Normas Regulamentadoras com fidelidade total: "
    "todo número, prazo, valor e condição preservado literalmente. Não invente "
    "nada; não omita requisitos."
)

PROMPT = """Reescreva o trecho da {norma} abaixo como {registro}.

Regras: fidelidade total (números/prazos/valores literais), cobrir TODOS os requisitos do trecho, citar a norma ({norma}) e os itens numerados quando visíveis. Escreva apenas a reescrita, sem preâmbulo nem comentários.

TRECHO:
{chunk}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-window", type=int, default=2,
                    help="registros sorteados por janela")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else treated_norms()
    for nr in nrs:
        corpus = load_norm_corpus(nr)
        windows = split_windows(corpus["conteudo"])
        if args.limit:
            windows = windows[: args.limit]
        jobs = []
        for i, w in enumerate(windows):
            rng = random.Random(f"{nr}:{i}:para")
            for reg in rng.sample(sorted(REGISTROS), args.per_window):
                jobs.append({"key": f"{i}:{reg}", "window": i, "registro": reg,
                             "chunk": w})
        print(f"[NR-{nr}] {len(windows)} janelas x {args.per_window} registros "
              f"= {len(jobs)} parafraseios")
        ckpt = SYNTH / f"para_nr{nr}.ckpt.json"

        def worker(j: dict) -> dict:
            text = gen_call(PROMPT.format(norma=corpus["norma"],
                                          registro=REGISTROS[j["registro"]],
                                          chunk=j["chunk"]),
                            system=SYSTEM, temperature=0.6, max_tokens=4096,
                            thinking=False)
            return {"text": text.strip()}

        results = parallel_map(worker, jobs, max_workers=args.workers,
                               checkpoint_path=ckpt, key_fn=lambda j: j["key"])
        rows, n_err = [], 0
        for j in jobs:
            r = results.get(j["key"])
            if not r or "__error__" in r:
                n_err += 1
                continue
            pid = hashlib.sha256(f"{nr}:{j['key']}".encode()).hexdigest()[:16]
            rows.append({"para_id": pid, "nr": nr, "window": j["window"],
                         "registro": j["registro"], "text": r["text"]})
        out = SYNTH / f"para_nr{nr}.jsonl"
        write_jsonl(out, rows)
        print(f"[ok] {out.name}: {len(rows)} parafraseios, {n_err} erros")


if __name__ == "__main__":
    main()
