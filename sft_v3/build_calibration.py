#!/usr/bin/env python3
"""
build_calibration.py — PARTE 4: monta o dataset de CALIBRAÇÃO para a quantização (imatrix).

O capitão apontou a pendência: quantizar COM dataset de calibração para recuperar a perda
bf16->Q4 (no v2 foram ~9 p.p.). O llama-imatrix computa uma matriz de importância por ativação
sobre um corpus representativo; a quantização Q4_K_M/Q4_0 usa essa imatrix para preservar os
pesos que mais importam no DOMÍNIO (perguntas de campo + trechos de NR), em vez de assumir
distribuição genérica.

Corpus de calibração = texto no MESMO formato que o modelo vê em produção (system prompt v3 +
contexto normativo + pergunta), FORA do holdout (mesmas muralhas do treino). Cada linha é um
prompt completo renderizado pelo chat template — é o que a imatrix precisa ver.

Fontes (todas fora do holdout/NR reservadas):
  - as próprias perguntas+contextos do train_v3 (oráculo + recusa + parcial + distrator) —
    representam exatamente a distribuição de uso; amostra estratificada.
Saída: um .txt (um prompt renderizado por bloco, separados por linha em branco) que o
llama-imatrix consome com -f.

Higiene: determinístico (seed fixa), não-interativo. Comentários PT-BR; código em inglês.

Uso: classifier/.venv/bin/python build_calibration.py --n 400 --out data/calib_v3.txt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "sft_v2"))
import common_v2 as C  # noqa: E402


def render_prompt(messages: List[Dict[str, str]]) -> str:
    """Renderiza um bloco de calibração no formato de uso (system+user+assistant em texto).

    A imatrix só precisa de texto representativo do domínio; usamos o system prompt v3, o
    contexto normativo e a pergunta E TAMBÉM a resposta-alvo (o modelo ativa os mesmos
    circuitos de síntese/recusa que quantizamos). Formato simples e estável (não depende do
    chat template exato — o llama-imatrix tokeniza texto puro)."""
    parts = []
    for m in messages:
        role = m["role"].upper()
        parts.append(f"<|{role}|>\n{m['content']}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(_HERE / "data" / "train_v3.jsonl"))
    ap.add_argument("--n", type=int, default=400, help="blocos de calibração")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default=str(_HERE / "data" / "calib_v3.txt"))
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for line in Path(args.train).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))

    # amostra estratificada por família (mesma proporção do treino -> representa o uso real).
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_fam.setdefault(r["meta"]["family"], []).append(r)
    rng = random.Random(args.seed)
    total = len(rows)
    picked: List[Dict[str, Any]] = []
    for fam, items in by_fam.items():
        share = max(1, round(args.n * len(items) / total))
        rng.shuffle(items)
        picked.extend(items[:share])
    rng.shuffle(picked)
    picked = picked[: args.n]

    blocks = [render_prompt(r["messages"]) for r in picked]
    text = "\n\n".join(blocks) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")

    fam_counts: Dict[str, int] = {}
    for r in picked:
        f = r["meta"]["family"]
        fam_counts[f] = fam_counts.get(f, 0) + 1
    print(f"calibração: {len(picked)} blocos | famílias {fam_counts} | "
          f"{len(text)} chars -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
