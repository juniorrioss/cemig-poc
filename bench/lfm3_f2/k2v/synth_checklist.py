"""
synth_checklist.py — gera o checklist por QA do K2V (juiz de raciocínio) via
API do 27B, adaptado para PT-BR / segurança do trabalho.

Porte do utils/data_postprocess/synthesize_checklist.py do K2V: o original usa
vLLM offline com domínios EN_AGRI/EN_MEDICINE/EN_LAW; aqui usamos o MESMO 27B
(já servindo em 10.100.0.111:8005) com um guia de pontuação do nosso domínio
(normas regulamentadoras), mantendo o formato de saída (lista JSON de
critérios por item) que o verl/K2V espera.

Entrada: qa-*.json do graphgen-mask (fill_blank).
Saída: data/k2v_qa_checklist.jsonl (question, answer, adjacent_nodes, checklist)

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/k2v/synth_checklist.py \
        --qa out_full/data/graphgen/*/qa-*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synth"))
from common_synth import gen_json, parallel_map, write_jsonl  # noqa: E402

SYSTEM = """Você é um examinador sênior de segurança do trabalho e Normas Regulamentadoras (NRs) brasileiras. Sua tarefa é criar um checklist de correção detalhado para uma [Questão Específica] com base no [Guia Geral de Pontuação].

[Guia Geral de Pontuação]:
Conceitos e Normas
1. Identifica corretamente a norma regulamentadora e o item/anexo pertinente à questão.
2. Define com precisão os conceitos técnicos envolvidos (ex.: zona de risco, trabalho em altura, espaço confinado).
3. Preserva literalmente valores numéricos, prazos, distâncias e limites da norma.
Raciocínio Normativo
4. Aplica a regra ao cenário concreto com a condição correta (quem, quando, onde se aplica).
5. Distingue requisitos obrigatórios ("deve") de recomendações.
6. Reconhece exceções e condições de contorno da regra quando pertinentes.
Argumentação
7. Encadeia as premissas normativas até a conclusão sem saltos lógicos.
8. Não extrapola além do que a norma e o enunciado sustentam.
9. A resposta final decorre do raciocínio apresentado, sem contradições.

Com base no guia acima, crie um checklist detalhado, específico e objetivamente verificável para a [Questão Específica] fornecida. O checklist avaliará o PROCESSO DE RACIOCÍNIO de um modelo ao responder. Cada item deve ser uma frase clara e verificável, ligada ao conteúdo da questão. Produza entre 4 e 8 itens.
Saída APENAS em JSON: ["critério 1", "critério 2", ...]"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="glob dos qa-*.json do graphgen")
    ap.add_argument("--out", default=str(HERE / "data" / "k2v_qa_checklist.jsonl"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qa_all: dict = {}
    for path in sorted(glob.glob(args.qa)):
        qa_all.update(json.load(open(path, encoding="utf-8")))
    items = [{"qa_hash": k, **v} for k, v in qa_all.items()]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} QAs fill-blank")

    ckpt = HERE / "data" / "k2v_checklist.ckpt.json"

    def worker(it: dict) -> dict:
        prompt = (f"[Questão Específica]\nPergunta (preencha a lacuna {{ }}): "
                  f"{it['question']}\nResposta correta: {it['answer']}")
        checklist = gen_json(prompt, system=SYSTEM, temperature=0.7,
                             max_tokens=2048)
        if isinstance(checklist, dict):
            checklist = next((v for v in checklist.values()
                              if isinstance(v, list)), [])
        checklist = [str(c) for c in checklist if isinstance(c, str) and c.strip()]
        return {"checklist": checklist}

    results = parallel_map(worker, items, max_workers=args.workers,
                           checkpoint_path=ckpt, key_fn=lambda it: it["qa_hash"])
    rows, n_err = [], 0
    for it in items:
        r = results.get(it["qa_hash"])
        if not r or "__error__" in r or not r.get("checklist"):
            n_err += 1
            continue
        rows.append({"qa_hash": it["qa_hash"], "question": it["question"],
                     "answer": it["answer"], "type": it.get("type", ""),
                     "adjacent_nodes": it.get("adjacent_nodes", {}),
                     "checklist": r["checklist"]})
    write_jsonl(Path(args.out), rows)
    print(f"[ok] {args.out}: {len(rows)} com checklist, {n_err} erros/vazios")


if __name__ == "__main__":
    main()
