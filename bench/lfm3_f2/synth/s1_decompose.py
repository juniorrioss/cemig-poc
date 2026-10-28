"""
s1_decompose.py — estágio (a) do funil: decompõe cada norma TRATADA em
unidades de conhecimento (requisitos/artigos/tabelas/limiares) com referência
exata, via 27B (thinking antes do JSON).

Janela deslizante sobre o texto limpo da norma (~6000 chars, corte em quebra
de seção) para caber no contexto do gerador com folga. Cada janela produz uma
lista de unidades; a referência exata (item numerado) ancora a auditoria.

Saída: data/synth/units_nr{N}.jsonl  (1 unidade por linha)

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/synth/s1_decompose.py [--nr 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, gen_json, load_norm_corpus, parallel_map,
                          treated_norms, write_jsonl)

WINDOW = 6000
MIN_TAIL = 1500

SYSTEM = (
    "Você é um engenheiro de segurança do trabalho especialista em Normas "
    "Regulamentadoras brasileiras, preparando material de estudo fiel ao texto "
    "normativo. Responda sempre com um bloco JSON ao final."
)

PROMPT = """Abaixo está um trecho da {norma} ({titulo}).

Extraia TODAS as unidades de conhecimento autocontidas deste trecho. Uma unidade de conhecimento é um fato normativo que pode ser estudado isoladamente: um requisito, uma obrigação, uma proibição, um limiar numérico, uma definição, uma linha de tabela, um prazo, uma competência/responsabilidade.

Regras:
- "ref" = referência EXATA ao item da norma (ex.: "{norma} item 10.2.4 alínea b", "{norma} Anexo II tabela de distâncias"). Se o trecho não mostra o número do item, use a melhor âncora textual visível.
- "fato" = enunciado completo e autossuficiente do fato, com TODOS os números/valores/prazos preservados literalmente.
- "tipo" = um de: requisito | proibicao | limiar | definicao | tabela | prazo | responsabilidade | procedimento
- "densa" = true se a unidade tem múltiplas condições, exceções ou valores inter-relacionados (candidata ao método socrático).
- NÃO invente nada que não esteja no trecho. NÃO resuma perdendo números.
- Ignore cabeçalhos, sumários e texto sem conteúdo normativo.

Pense primeiro sobre a estrutura do trecho e depois emita APENAS o JSON final:
```json
[{{"ref": "...", "fato": "...", "tipo": "...", "densa": false}}, ...]
```

TRECHO:
{chunk}
"""


def split_windows(text: str, window: int = WINDOW) -> list[str]:
    """Janelas com corte preferencial em quebra de linha dupla/seção."""
    out = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + window, n)
        if end < n:
            cut = text.rfind("\n\n", pos + window // 2, end)
            if cut == -1:
                cut = text.rfind("\n", pos + window // 2, end)
            if cut > pos:
                end = cut
        out.append(text[pos:end])
        pos = end
    # cola rabo curto na janela anterior
    if len(out) >= 2 and len(out[-1]) < MIN_TAIL:
        out[-2] += out[-1]
        out.pop()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0, help="só uma NR (default: todas tratadas)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="limita janelas (smoke)")
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else treated_norms()
    for nr in nrs:
        corpus = load_norm_corpus(nr)
        windows = split_windows(corpus["conteudo"])
        if args.limit:
            windows = windows[: args.limit]
        print(f"[{corpus['norma']}] {len(corpus['conteudo'])} chars -> {len(windows)} janelas")

        items = [{"idx": i, "chunk": w} for i, w in enumerate(windows)]
        ckpt = SYNTH / f"units_nr{nr}.ckpt.json"

        def worker(it: dict) -> list:
            units = gen_json(PROMPT.format(norma=corpus["norma"],
                                           titulo=corpus["titulo"],
                                           chunk=it["chunk"]),
                             system=SYSTEM, temperature=0.3, max_tokens=8192)
            if isinstance(units, dict):
                # às vezes o modelo embrulha a lista num objeto
                for v in units.values():
                    if isinstance(v, list):
                        units = v
                        break
                else:
                    units = [units] if units.get("fato") else []
            assert isinstance(units, list), f"esperava lista, veio {type(units)}"
            return units

        results = parallel_map(worker, items, max_workers=args.workers,
                               checkpoint_path=ckpt,
                               key_fn=lambda it: str(it["idx"]))

        rows = []
        n_err = 0
        for i in range(len(items)):
            r = results.get(str(i))
            if isinstance(r, dict) and "__error__" in r:
                n_err += 1
                continue
            for u in r:
                if not isinstance(u, dict) or not u.get("fato"):
                    continue
                uid = hashlib.sha256(
                    f"{nr}:{u.get('ref','')}:{u['fato']}".encode()).hexdigest()[:16]
                rows.append({"unit_id": uid, "nr": nr, "window": i,
                             "ref": u.get("ref", ""), "fato": u["fato"],
                             "tipo": u.get("tipo", ""), "densa": bool(u.get("densa"))})
        # dedup exato por unit_id
        seen, dedup = set(), []
        for r in rows:
            if r["unit_id"] not in seen:
                seen.add(r["unit_id"])
                dedup.append(r)
        out = SYNTH / f"units_nr{nr}.jsonl"
        write_jsonl(out, dedup)
        print(f"[ok] {out.name}: {len(dedup)} unidades "
              f"({sum(1 for r in dedup if r['densa'])} densas), {n_err} janelas com erro")


if __name__ == "__main__":
    main()
