"""
gen_targets.py — item 1d do brief: respostas-alvo do dataset F3 geradas pelo
27B (10.100.0.111:8005) vendo APENAS a passagem ouro (caso i) ou APENAS a
passagem parcial (caso iii). Thinking antes do JSON; parser do último bloco
válido (common_synth da F2, reuso obrigatório).

Entrada: data/train/assignments.jsonl (de build_dataset.py --assign)
Saída:   data/train/targets.jsonl  {qa_id, case, target}

Casos cobertos aqui: "gold" e "partial" (os únicos que precisam do 27B).
"absent" usa recusa-padrão determinística; "closed_book" reusa a resposta
validada do funil — ambos montados no build_dataset.py --render.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3/synth/gen_targets.py [--workers 24]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
sys.path.insert(0, str(HERE))
from common_f3 import f2_synth_modules  # noqa: E402

cs = f2_synth_modules()

TRAIN = F3 / "data" / "train"

SYSTEM = (
    "Você é um assistente técnico de segurança do trabalho, especialista em "
    "Normas Regulamentadoras (NRs) brasileiras. Escreve respostas-alvo de "
    "treino: curtas, fiéis e ancoradas na passagem fornecida. Responda sempre "
    "com um bloco JSON ao final."
)

PROMPT_GOLD = """PASSAGEM NORMATIVA ({src}):
{text}

PERGUNTA DO TRABALHADOR:
{question}

Escreva a resposta ideal para a pergunta, usando APENAS a passagem acima.

Regras:
- 2 a 5 frases, português do Brasil, tom direto de assistente de voz.
- Fiel à passagem: preserve números, prazos e condições literalmente.
- Cite a norma e o item/seção (ex.: "NR-10 item 10.2.8") na resposta.
- NÃO mencione "a passagem", "o trecho" ou "o contexto" — fale da norma.
- Se a pergunta pede algo que a passagem só cobre em parte, responda o que a
  passagem sustenta e diga explicitamente o que ela não cobre.

Pense primeiro e emita APENAS o JSON final no formato
{{"resposta": "<texto completo da resposta>"}}"""

PROMPT_PARTIAL = """PASSAGEM NORMATIVA ({src}) — cobre o tema SÓ EM PARTE:
{text}

PERGUNTA DO TRABALHADOR:
{question}

A passagem acima toca no tema da pergunta mas NÃO contém a resposta completa.
Escreva a resposta ideal: responda SOMENTE o que a passagem sustenta e sinalize
com clareza o limite (o que ela não cobre e onde o trabalhador consultaria).

Regras:
- 2 a 4 frases, português do Brasil, tom direto de assistente de voz.
- NADA além do que a passagem diz; números/condições literais.
- Cite a norma e o item/seção da passagem.
- Sinalize o limite SEM inventar: "os trechos recuperados não detalham X".
- NÃO mencione "a passagem"/"o trecho acima" — refira-se aos trechos das normas recuperados.

Pense primeiro e emita APENAS o JSON final no formato
{{"resposta": "<texto completo da resposta>"}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = cs.read_jsonl(TRAIN / "assignments.jsonl")
    todo = [r for r in rows if r["case"] in ("gold", "partial")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[targets] {len(todo)} alvos (gold+partial) de {len(rows)} assignments")
    ckpt = TRAIN / "targets.ckpt.json"

    def worker(r: dict) -> dict:
        anchor = r["anchor_passage"]
        src = f"{anchor['doc'].upper()} {anchor['section']} — {anchor['title']}"
        tpl = PROMPT_GOLD if r["case"] == "gold" else PROMPT_PARTIAL
        # até 3 tentativas: o 27B às vezes ecoa o placeholder do formato no
        # thinking e o parser pega o eco — rejeita e re-chama
        last = ""
        for _ in range(3):
            v = cs.gen_json(tpl.format(src=src, text=anchor["text"],
                                       question=r["question"]),
                            system=SYSTEM, temperature=0.4, max_tokens=2048)
            if isinstance(v, list):
                v = v[0] if v else {}
            resp = str(v.get("resposta", "")).strip() if isinstance(v, dict) else ""
            if len(resp) > 30 and "<texto" not in resp:
                return {"target": resp}
            last = resp
        raise ValueError(f"resposta inválida 3x: {last!r}")

    # ckpt chaveado por qa_id:case — o caso muda o prompt; rebuilds do assign
    # (quando o pool F3 entrar) reusam só o que continua válido.
    results = cs.parallel_map(worker, todo, max_workers=args.workers,
                              checkpoint_path=ckpt,
                              key_fn=lambda r: f"{r['qa_id']}:{r['case']}")
    out, n_err = [], 0
    for r in todo:
        v = results.get(f"{r['qa_id']}:{r['case']}")
        if not v or "__error__" in v:
            n_err += 1
            continue
        out.append({"qa_id": r["qa_id"], "case": r["case"], "target": v["target"]})
    cs.write_jsonl(TRAIN / "targets.jsonl", out)
    print(f"[ok] targets.jsonl: {len(out)} alvos, {n_err} erros")


if __name__ == "__main__":
    main()
