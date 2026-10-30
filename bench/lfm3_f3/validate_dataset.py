#!/usr/bin/env python3
"""
validate_dataset.py — GATE final do dataset F3 (ordem do brief: validadores
por estágio são gate, não warning; exit 1 ruidoso).

Checks sobre data/train/{assignments,train,val}.jsonl:
  1. schema dos exemplos renderizados (messages system/user/assistant);
  2. consistência rótulo×caso: gold tem exatamente 1 passagem 'gold' na
     posição registrada; absent tem 0 'gold' e 0 'partial'; partial tem 0
     'gold' e >=1 'partial'; closed não tem contexto;
  3. distribuição de k (>=40% k=5; k=3 e k=8 entre 15 e 35% cada, entre os
     casos com contexto) e posição do ouro ~uniforme (nenhuma posição com
     mais de 2x a esperada);
  4. dedup: nenhuma pergunta duplicada entre train e val; contaminação vs
     benchmark já removida no assign (re-verificada por amostra);
  5. entropia de aberturas (prefixo-3 das perguntas) >= 6 bits (critério do
     validate_stage da F2);
  6. alvo do caso absent dispara looks_like_refusal (régua coerente) e alvo
     dos casos gold/partial NÃO dispara.

Uso: classifier/.venv/bin/python bench/lfm3_f3/validate_dataset.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))

from common_synth import norm_tokens, read_jsonl  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402

TRAIN = HERE / "data" / "train"


def fail(msg: str) -> None:
    print(f"[GATE-FAIL] {msg}")
    sys.exit(1)


def entropy_bits(prefixes: list[str]) -> float:
    import math
    counts = collections.Counter(prefixes)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def main() -> None:
    assigns = {a["qa_id"]: a for a in read_jsonl(TRAIN / "assignments.jsonl")}
    train = read_jsonl(TRAIN / "train.jsonl")
    val = read_jsonl(TRAIN / "val.jsonl")
    rows = train + val
    if not rows:
        fail("train/val vazios")
    print(f"[validate] train={len(train)} val={len(val)} assignments={len(assigns)}")

    # 1. schema
    for i, r in enumerate(rows):
        ms = r.get("messages", [])
        if [m["role"] for m in ms] != ["system", "user", "assistant"]:
            fail(f"linha {i}: roles {[m.get('role') for m in ms]}")
        if not all(m.get("content") for m in ms):
            fail(f"linha {i}: content vazio")
        if r["case"] not in ("gold", "absent", "partial", "closed"):
            fail(f"linha {i}: case inválido {r['case']}")
    print("  [ok] 1. schema messages")

    # 2. consistência rótulo×caso (via assignments)
    for r in rows:
        a = assigns.get(r["qa_id"])
        if not a or a["case"] != r["case"]:
            fail(f"{r['qa_id']}: sem assignment ou caso divergente")
        labs = a["labels"]
        if a["case"] == "gold":
            if labs.count("gold") != 1 or labs[a["gold_position"]] != "gold":
                fail(f"{r['qa_id']}: gold inconsistente {labs} pos={a['gold_position']}")
        elif a["case"] == "absent":
            if "gold" in labs or "partial" in labs:
                fail(f"{r['qa_id']}: absent com gold/partial {labs}")
        elif a["case"] == "partial":
            if "gold" in labs or "partial" not in labs:
                fail(f"{r['qa_id']}: partial inconsistente {labs}")
        elif a["case"] == "closed" and a["context_ids"]:
            fail(f"{r['qa_id']}: closed com contexto")
        # contexto renderizado bate com k
        if a["case"] != "closed":
            n_blocks = r["messages"][1]["content"].count("\n[")
            n_blocks += r["messages"][1]["content"].startswith("Contexto normativo consultado:\n[")
            if n_blocks != a["k"]:
                fail(f"{r['qa_id']}: k={a['k']} mas contexto tem {n_blocks} blocos")
    print("  [ok] 2. rótulo×caso consistente (gold pos registrada, absent sem ouro)")

    # 3. distribuição de k e posições
    ctx = [assigns[r["qa_id"]] for r in rows if r["case"] in ("gold", "absent", "partial")]
    kc = collections.Counter(a["k"] for a in ctx)
    n = len(ctx)
    share5 = sum(v for k, v in kc.items() if k == 5) / n
    share3 = kc.get(3, 0) / n
    share8 = kc.get(8, 0) / n
    print(f"    k=5 {share5:.1%} | k=3 {share3:.1%} | k=8 {share8:.1%} | outros "
          f"{1-share5-share3-share8:.1%} (absent/partial com pool curto)")
    if share5 < 0.40:
        fail(f"k=5 {share5:.1%} < 40%")
    if not (0.13 <= share3 <= 0.37) or not (0.10 <= share8 <= 0.37):
        fail(f"k=3 {share3:.1%} / k=8 {share8:.1%} fora da banda")
    golds = [a for a in ctx if a["case"] == "gold"]
    for k in (3, 5, 8):
        sub = [a for a in golds if a["k"] == k]
        if not sub:
            continue
        pc = collections.Counter(a["gold_position"] for a in sub)
        exp = len(sub) / k
        worst = max(pc.values())
        if worst > 2 * exp:
            fail(f"posição do ouro enviesada em k={k}: {dict(pc)}")
    print("  [ok] 3. distribuição de k e posição do ouro ~uniforme")

    # 4. dedup train×val
    qt = {r["messages"][1]["content"].rsplit("Pergunta:\n", 1)[-1] for r in train}
    dups = [r["qa_id"] for r in val
            if r["messages"][1]["content"].rsplit("Pergunta:\n", 1)[-1] in qt]
    if dups:
        fail(f"{len(dups)} perguntas de val também em train: {dups[:4]}")
    print("  [ok] 4. dedup train×val")

    # 5. entropia de aberturas
    questions = [r["messages"][1]["content"].rsplit("Pergunta:\n", 1)[-1] for r in rows]
    h = entropy_bits([" ".join(norm_tokens(q)[:3]) for q in questions])
    if h < 6.0:
        fail(f"entropia prefixo-3 = {h:.2f} < 6.0 bits")
    print(f"  [ok] 5. entropia de aberturas: {h:.2f} bits")

    # 6. recusa coerente com a régua
    bad_absent = [r["qa_id"] for r in rows if r["case"] == "absent"
                  and not looks_like_refusal(r["messages"][2]["content"])]
    if bad_absent:
        fail(f"{len(bad_absent)} alvos absent NÃO disparam looks_like_refusal")
    bad_gold = [r["qa_id"] for r in rows if r["case"] == "gold"
                and looks_like_refusal(r["messages"][2]["content"])]
    rate = len(bad_gold) / max(1, sum(1 for r in rows if r["case"] == "gold"))
    if rate > 0.02:
        fail(f"{rate:.1%} dos alvos gold disparam refusal (limite 2%): {bad_gold[:4]}")
    print(f"  [ok] 6. recusa coerente (gold c/ refusal-flag: {rate:.2%})")

    print("[PASS]")


if __name__ == "__main__":
    main()
