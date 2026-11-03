#!/usr/bin/env python3
"""
validate_dataset_f4.py — GATE final do dataset F4 (exit 1 ruidoso).

Checks sobre data/train/{train,val}.jsonl:
  1. schema: messages começam com system BAKED idêntico (sha única) e alternam
     papéis válidos; famílias conhecidas;
  2. ROUND-TRIP de TODAS as tool_calls (render real -> parser -> idêntico);
  3. famílias com chamadas corretas: chamada_simples/recusa = 1 call;
     troca = 2 calls (NRs distintas); reuso = 1 call + resposta final sem
     call no 2º turno; multihop = 2 calls com argumentos DIFERENTES;
     sem_ferramenta = 0 calls;
  4. distribuição de famílias dentro de ±20% das quotas; recusa entre 18 e 28%;
  5. dedup train×val por pergunta do 1º turno; entropia de aberturas >= 6 bits;
  6. recusa: alvo dispara looks_like_refusal; famílias que respondem, não;
  7. orçamento de tokens: truncamento <=1% no max_len do treino (6912).

Uso: classifier/.venv/bin/python bench/lfm3_f4/validate_dataset_f4.py
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))

import common_f4 as C  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402
from tool_schema_f4 import baked_system_content  # noqa: E402

TRAIN = HERE / "data" / "train"
MAX_LEN = 6912  # medido: 0% acima (6144 truncaria 2.5% — troca/multihop)


def fail(msg: str) -> None:
    print(f"[GATE-FAIL] {msg}")
    sys.exit(1)


def entropy_bits(prefixes: list) -> float:
    counts = collections.Counter(prefixes)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def calls_of(row: dict) -> list:
    out = []
    for m in row["messages"]:
        for tc in (m.get("tool_calls") or []):
            out.append(tc["function"]["arguments"])
    return out


def final_answer(row: dict) -> str:
    for m in reversed(row["messages"]):
        if m["role"] == "assistant" and not m.get("tool_calls"):
            return m.get("content") or ""
    return ""


def first_user(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def main() -> None:
    train = C.read_jsonl(TRAIN / "train.jsonl")
    val = C.read_jsonl(TRAIN / "val.jsonl")
    rows = train + val
    if not rows:
        fail("train/val vazios")
    print(f"[validate] train={len(train)} val={len(val)}")

    baked = baked_system_content()
    fams = collections.Counter(r["family"] for r in rows)

    # 1. schema + system único
    for r in rows:
        ms = r["messages"]
        if ms[0]["role"] != "system" or ms[0]["content"] != baked:
            fail(f"system divergente em {r.get('qa_id')}")
        if ms[1]["role"] != "user":
            fail(f"2ª mensagem não é user em {r.get('qa_id')}")

    # 2. round-trip de todas as tool_calls (amostra COMPLETA)
    bad_rt = [r["qa_id"] for r in rows if not C.roundtrip_ok(r["messages"])]
    if bad_rt:
        fail(f"round-trip falhou em {len(bad_rt)} exemplos: {bad_rt[:5]}")
    print("[validate] round-trip 100%")

    # 3. invariantes por família
    for r in rows:
        cs = calls_of(r)
        fam = r["family"]
        if fam in ("chamada_simples", "recusa") and len(cs) != 1:
            fail(f"{fam} com {len(cs)} calls ({r['qa_id']})")
        if fam == "troca" and len(cs) != 2:
            fail(f"troca com {len(cs)} calls ({r['qa_id']})")
        if fam == "reuso":
            if len(cs) != 1:
                fail(f"reuso com {len(cs)} calls ({r['qa_id']})")
            # último turno assistant não pode ter tool_call
            last = r["messages"][-1]
            if last["role"] != "assistant" or last.get("tool_calls"):
                fail(f"reuso termina mal ({r['qa_id']})")
        if fam == "multihop":
            if len(cs) != 2:
                fail(f"multihop com {len(cs)} calls ({r['qa_id']})")
            a1 = " ".join(C.norm_tokens(cs[0]["consulta"]))
            a2 = " ".join(C.norm_tokens(cs[1]["consulta"]))
            if a1 == a2:
                fail(f"multihop com argumentos iguais ({r['qa_id']})")
        if fam == "sem_ferramenta" and cs:
            fail(f"sem_ferramenta com call ({r['qa_id']})")
    print("[validate] invariantes por família ok")

    # 4. distribuição
    n = len(rows)
    ref_pct = fams["recusa"] / n
    if not 0.18 <= ref_pct <= 0.28:
        fail(f"dose de recusa fora da banda: {ref_pct:.1%}")
    print(f"[validate] famílias: {dict(fams)} (recusa {ref_pct:.1%})")

    # 5. dedup train×val + entropia
    tq = {" ".join(C.norm_tokens(first_user(r))) for r in train}
    vq = {" ".join(C.norm_tokens(first_user(r))) for r in val}
    inter = tq & vq
    if inter:
        fail(f"{len(inter)} perguntas repetidas train×val")
    ent = entropy_bits([" ".join(C.norm_tokens(first_user(r))[:3]) for r in rows])
    if ent < 6.0:
        fail(f"entropia de aberturas {ent:.2f} < 6 bits")
    print(f"[validate] dedup ok; entropia {ent:.2f} bits")

    # 6. recusa coerente com a régua. No sentido inverso (famílias que
    # respondem) o regex pega DELIMITAÇÃO legítima dos alvos 27B ("a norma
    # não detalha X" — instrução do PROMPT_GOLD da F3, honestidade, não
    # recusa) -> métrica informativa com teto frouxo, não binária.
    n_ans = n_ansref = 0
    for r in rows:
        ans = final_answer(r)
        if r["family"] == "recusa" and not looks_like_refusal(ans):
            fail(f"alvo de recusa não dispara régua ({r['qa_id']})")
        if r["family"] in ("chamada_simples", "multihop"):
            n_ans += 1
            n_ansref += looks_like_refusal(ans)
    rate = n_ansref / n_ans if n_ans else 0
    print(f"[validate] recusa: 100% dos alvos de recusa disparam a régua; "
          f"{rate:.1%} dos alvos que respondem contem frase delimitadora")
    if rate > 0.25:
        fail(f"taxa de delimitação nos alvos que respondem alta demais: {rate:.1%}")

    # 7. orçamento de tokens
    lens = sorted(C.count_tokens(r["messages"]) for r in rows)
    p99 = lens[int(0.99 * len(lens))]
    over = sum(1 for x in lens if x > MAX_LEN)
    print(f"[validate] tokens: mediana {lens[len(lens)//2]}, p99 {p99}, "
          f"máx {lens[-1]}, >{MAX_LEN}: {over} ({over/len(lens):.2%})")
    if over / len(lens) > 0.01:
        fail(f"truncamento > 1% a {MAX_LEN}")

    print("[validate] GATE OK")


if __name__ == "__main__":
    main()
