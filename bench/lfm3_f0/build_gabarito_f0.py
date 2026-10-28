"""
build_gabarito_f0.py — gabarito de fatos obrigatórios do benchmark_mobile (F0).

Porte do bench/regua/build_gabarito.py para o dataset novo: a fonte de fatos é
answer (gabarito oficial) + evidence (trechos com offset da norma), que o dataset
traz prontos — não há resolução de chunk-ouro via índice. UM gabarito por
base_qa_id (384): answer/evidence são idênticos nos 4 vocabulários; a pergunta
usada no prompt é a do fraseado "clean".

Juiz: Qwen3.8-27B-FP8 (REGUA_JUDGE_URL), paralelo com checkpoint.

Uso (local, juiz acessível direto):
    python3 bench/lfm3_f0/build_gabarito_f0.py [--workers 16] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "regua"))

from common import judge_call, parallel_map  # bench/regua (cliente do juiz)
from common_f0 import load_rows

DATA = HERE / "data"
OUT_PATH = DATA / "gabarito_384.jsonl"
CKPT_PATH = DATA / "gabarito_384.ckpt.json"

EXTRACT_PROMPT = """Você é um auditor técnico de Normas Regulamentadoras brasileiras (NRs).
Sua tarefa: a partir da RESPOSTA-OURO OFICIAL e das EVIDÊNCIAS DA NORMA abaixo, extrair a
lista dos FATOS OBRIGATÓRIOS que QUALQUER resposta correta à pergunta precisa conter.

Um FATO OBRIGATÓRIO é um item concreto e verificável: um equipamento, uma etapa de
procedimento, uma condição, um valor/limite, uma exigência, um prazo.

REGRAS ESTRITAS:
- Extraia SOMENTE fatos presentes na RESPOSTA-OURO ou nas EVIDÊNCIAS. NÃO invente,
  NÃO use conhecimento externo.
- Cada fato deve ser CURTO (1 a 6 palavras), ATÔMICO (um item por entrada) e no
  VOCABULÁRIO da norma.
- Foque nos fatos que respondem à PERGUNTA. Produza de 2 a 6 fatos.
- Não inclua o número da norma/item como fato (a citação não conta).

[PERGUNTA DO TRABALHADOR]
{question}

[RESPOSTA-OURO OFICIAL]
{answer}

[EVIDÊNCIAS DA NORMA]
{evidence}

Retorne EXCLUSIVAMENTE um objeto JSON estrito:
{{"fatos": ["fato 1", "fato 2", "..."]}}"""


def _extract_list(d: dict):
    for k in ("fatos", "facts", "factos"):
        if isinstance(d.get(k), list):
            return d[k]
    for v in d.values():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
        if isinstance(v, dict):  # o juiz às vezes aninha ({"resposta": {"fatos": [...]}})
            lst = _extract_list(v)
            if lst is not None:
                return lst
    return None


def _clean(facts: list) -> list:
    out, seen = [], set()
    for f in facts:
        if not isinstance(f, str):
            continue
        s = re.sub(r"\s+", " ", f).strip().strip('."')
        if not s or len(s) > 80:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:6]


def _parse_facts(raw: str) -> list:
    """Pega o ÚLTIMO bloco JSON válido (o juiz emite thinking antes)."""
    text = raw.strip()
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blocks += re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", text, re.DOTALL)
    for b in reversed(blocks):
        try:
            lst = _extract_list(json.loads(b))
            if lst is not None:
                return _clean(lst)
        except Exception:
            pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parquet", default=str(
        Path.home() / ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/"
        "snapshots/5a5aa32ae4087481be828e071aff169827ec2993/"
        "normas_regulamentadoras/train-00000-of-00001.parquet"))
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    rows = load_rows(args.parquet)
    # 1 item por base_qa_id; pergunta de referência = fraseado clean.
    by_base = {}
    for r in rows:
        if r["vocabulary_level"] == "clean":
            by_base[r["base_qa_id"]] = r
    items = sorted(by_base.values(), key=lambda r: r["base_qa_id"])
    if args.limit:
        items = items[: args.limit]
    print(f"[gabarito] {len(items)} base_qa_ids")

    existing = {}
    if CKPT_PATH.exists():
        existing = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
        print(f"[ckpt] retomando com {len(existing)}")

    def worker(r: dict) -> dict:
        ev = "\n\n".join(e["text"] for e in r["evidence"])[:4000]
        prompt = EXTRACT_PROMPT.format(
            question=r["question"], answer=r["answer"], evidence=ev)
        raw = judge_call(prompt, max_tokens=768, temperature=0.0)
        facts = _parse_facts(raw)
        return {
            "base_qa_id": r["base_qa_id"], "question_clean": r["question"],
            "answer": r["answer"], "question_type": r["question_type"],
            "source_file": r["source_file"], "facts": facts,
            "n_facts": len(facts),
            "provenance": {"source": "answer+evidence(benchmark_mobile)",
                           "judge_model": "Qwen/Qwen3.8-27B-FP8"},
        }

    results = parallel_map(worker, items, max_workers=args.workers,
                           checkpoint_path=CKPT_PATH,
                           key_fn=lambda r: r["base_qa_id"], existing=existing)

    # retry serial de vazios
    empties = [r for r in items
               if not results.get(r["base_qa_id"], {}).get("facts")]
    if empties:
        print(f"[revisão] {len(empties)} sem fatos; retry serial")
        for r in empties:
            try:
                results[r["base_qa_id"]] = worker(r)
            except Exception as e:  # noqa: BLE001
                print(f"  falha {r['base_qa_id']}: {e}")

    ordered = [results[r["base_qa_id"]] for r in items if r["base_qa_id"] in results]
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in ordered:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    nf = [rec["n_facts"] for rec in ordered]
    print(f"[ok] {OUT_PATH}: {len(ordered)} itens | média "
          f"{sum(nf)/max(1,len(nf)):.2f} fatos | vazios={sum(1 for n in nf if n==0)}")


if __name__ == "__main__":
    main()
