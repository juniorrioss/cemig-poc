"""
prep_train_data.py — converte data/synth/train_mix.jsonl na mistura final de
treino estilo PRÉ-TREINO (texto corrido, CLM) para os braços A e B.

Formato: cada exemplo é um documento de texto puro:
  - raw        : o texto normativo literal da janela
  - paraphrase : a reescrita multi-registro
  - qa         : renderizado "Pergunta: ...\nResposta: ..." (texto, não chat
                 template — simulamos pré-treino, ordem do capitão; o chat é
                 preservado pelo LoRA ser baixo-rank e pelos guard-evals)

Split: 98% treino / 2% validação (seed fixa), estratificado por kind.

Saída: data/train/train.jsonl, data/train/val.jsonl  ({"text": ...})
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F2 = HERE.parent
sys.path.insert(0, str(F2 / "synth"))
from common_synth import read_jsonl, write_jsonl  # noqa: E402

SEED = 42


def render(doc: dict) -> str:
    if doc["kind"] in ("raw", "paraphrase"):
        return doc["text"].strip()
    # QA como texto corrido (estilo pré-treino, com a referência na resposta)
    return f"Pergunta: {doc['question'].strip()}\nResposta: {doc['answer'].strip()}"


def main() -> None:
    mix = read_jsonl(F2 / "data" / "synth" / "train_mix.jsonl")
    rng = random.Random(SEED)
    train, val = [], []
    by_kind: dict[str, list] = {}
    for d in mix:
        by_kind.setdefault(d["kind"], []).append(d)
    for kind, docs in sorted(by_kind.items()):
        rng.shuffle(docs)
        n_val = max(1, int(len(docs) * 0.02))
        for i, d in enumerate(docs):
            rec = {"text": render(d), "kind": kind, "nr": d["nr"], "key": d["key"]}
            (val if i < n_val else train).append(rec)
    rng.shuffle(train)
    out = F2 / "data" / "train"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    import collections
    print("train:", len(train), dict(collections.Counter(r['kind'] for r in train)))
    print("val:", len(val), dict(collections.Counter(r['kind'] for r in val)))


if __name__ == "__main__":
    main()
