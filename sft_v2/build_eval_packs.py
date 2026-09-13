#!/usr/bin/env python3
"""
build_eval_packs.py — Constrói os packs de VALIDAÇÃO e de TESTE DE RECUSA (Partes 2 e 3).

Usa APENAS os chunks reservados por split_chunks (valtest, FORA do treino e do holdout),
particionados em dois blocos disjuntos:
  - val_pack.json  (~150 itens, 4 famílias na mesma proporção do treino):
      métrica de VALIDAÇÃO da Parte 2 — a lição do v1 foi que a loss caiu enquanto a
      qualidade regrediu; a régua na validação (fora do treino) revela isso.
  - refusal_test_pack.json (a NOVA MÉTRICA obrigatória da Parte 3):
      itens ANSWERABLE (contexto responde — o modelo NÃO deve recusar) e UNANSWERABLE
      (contexto NÃO responde — o modelo DEVE recusar). Permite medir:
        recusa-correta%  = recusou nos unanswerable
        recusa-indevida% = recusou nos answerable (dano igualmente grave)

Cada item traz o contexto FORMATADO (portátil), a família, os fatos (gabarito p/ a régua),
o trecho-ouro real (checagem de alucinação) e as flags answerable/expects_refusal.

O 27B (professor) só gera (pergunta, fatos) por chunk — não o alvo (o alvo aqui é o
comportamento do ALUNO, medido pela régua). Paralelo, checkpoint, não-interativo.
Comentários PT-BR; código em inglês.

Uso: ../classifier/.venv/bin/python build_eval_packs.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import common_v2 as C
from gen_data import gen_qa

_HERE = Path(__file__).resolve().parent
_lock = Lock()


def _make_item(item_id: str, question: str, context: str, family: str, facts: List[str],
               gold_text: str, answerable: bool, doc: str, section: str,
               extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_id": item_id, "question": question, "context": context,
        "family": family, "facts": facts, "gold_text": gold_text[:3000],
        "answerable": answerable, "expects_refusal": (not answerable),
        "doc": doc, "section": section, "retrieval_hit": answerable,
        "captain_case": False, **extra,
    }


def build_item(chunk: Dict[str, Any], family: str, pool: List[Dict[str, Any]],
               url: str, model: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    """Gera um item de avaliação de dada família (contexto + pergunta + fatos)."""
    qa = gen_qa(chunk, url, model)
    if not qa:
        return None
    question, facts = qa
    gold_text = chunk["text"]
    iid = f"{family}-{chunk['id']}"
    if family == "oracle":
        ctx = C.format_context_single(chunk)
        return _make_item(iid, question, ctx, "oracle", facts, gold_text, True,
                          chunk["doc"], chunk["section"], {})
    if family == "recusa":
        wrong = C.pick_wrong_norm_chunk(chunk, pool, rng)
        if not wrong:
            return None
        ctx = C.format_context_single(wrong)
        return _make_item(iid, question, ctx, "recusa", facts, gold_text, False,
                          chunk["doc"], chunk["section"],
                          {"wrong_doc": wrong["doc"], "wrong_chunk_id": wrong["id"]})
    if family == "distrator":
        dist = C.pick_same_norm_distractor(chunk, pool, facts, rng)
        if not dist:
            return None
        ctx = C.format_context_single(dist)
        return _make_item(iid, question, ctx, "distrator", facts, gold_text, False,
                          chunk["doc"], chunk["section"],
                          {"distractor_section": dist["section"], "distractor_chunk_id": dist["id"]})
    if family == "parcial":
        partial_text, removed = C.make_partial_context(chunk, facts)
        if removed is None:
            return None
        pc = {**chunk, "text": partial_text}
        ctx = C.format_context_single(pc)
        # parcial é "answerable parcial": responde parte; NÃO deve recusar por completo,
        # mas DEVE declarar o que falta. Para a métrica de recusa binária, tratamos
        # parcial como caso à parte (não conta em recusa-correta nem indevida).
        return _make_item(iid, question, ctx, "parcial", facts, gold_text, True,
                          chunk["doc"], chunk["section"], {"removed_sentence": removed[:200]})
    return None


def run_pack(chunks: List[Dict[str, Any]], plan: Dict[str, int], pool: List[Dict[str, Any]],
             url: str, model: str, workers: int, seed: int, label: str) -> List[Dict[str, Any]]:
    """Gera itens conforme o plano {family: n}, um chunk por item (sem reuso entre famílias).

    Os chunks-fonte são PARTICIONADOS entre as famílias em fatias proporcionais ao plano
    (evita que a 1ª família consuma todo o pool). O `pool` completo segue disponível para a
    seleção de contexto errado/distrator.
    """
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    avail = list(chunks)
    rng.shuffle(avail)
    # overshoot por família (parcial falha ~metade: a remoção do dado-chave nem sempre dá).
    ov = {"oracle": 1.2, "recusa": 1.25, "distrator": 1.25, "parcial": 2.4}
    # necessidade bruta e normalização p/ caber no pool disponível (fatias disjuntas).
    need = {fam: int(n * ov.get(fam, 1.5)) + 6 for fam, n in plan.items()}
    total_need = sum(need.values())
    scale = min(1.0, len(avail) / max(1, total_need))
    slices: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for family, n in plan.items():
        take = min(len(avail) - cursor, max(n + 4, int(need[family] * scale)))
        slices[family] = avail[cursor:cursor + take]
        cursor += take
    for family, n in plan.items():
        got = 0
        batch = slices[family]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(build_item, c, family, pool, url, model,
                              random.Random(seed + c["id"])): c for c in batch}
            for fut in as_completed(futs):
                if got >= n:
                    for f in futs:
                        f.cancel()
                    break
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                if rec is not None:
                    with _lock:
                        items.append(rec)
                        got += 1
        print(f"  [{label}/{family}] {got}/{n}", flush=True)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--val-out", default=str(_HERE / "data" / "val_pack.json"))
    ap.add_argument("--refusal-out", default=str(_HERE / "data" / "refusal_test_pack.json"))
    args = ap.parse_args()

    _, valtest = C.split_chunks(seed=args.seed, min_len=args.min_len)
    print(f"valtest chunks disponíveis (FORA do treino/holdout): {len(valtest)}", flush=True)

    # Particiona valtest em dois blocos DISJUNTOS por chunk-fonte. A família PARCIAL rende
    # ~metade (nem todo chunk permite remover o dado-chave), então val recebe a fatia maior.
    rng = random.Random(args.seed)
    rng.shuffle(valtest)
    cut = int(len(valtest) * 0.62)
    val_chunks = valtest[:cut]
    ref_chunks = valtest[cut:]
    print(f"partição: val={len(val_chunks)} refusal-test={len(ref_chunks)} (disjuntos)", flush=True)

    t0 = time.time()
    # VAL: ~140 itens na proporção do treino (50/21/21/10), ajustado ao pool de val.
    val_plan = {"oracle": 70, "recusa": 30, "parcial": 25, "distrator": 15}
    val_items = run_pack(val_chunks, val_plan, valtest, args.url, args.model,
                         args.workers, args.seed, "val")
    val_pack = {
        "metadata": {"task": "poc-sft-v2", "pack": "validation", "n": len(val_items),
                     "plan": val_plan, "note": "fora do treino e do holdout (valtest bloco A)"},
        "items": val_items,
    }
    Path(args.val_out).write_text(json.dumps(val_pack, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    # REFUSAL-TEST: answerable (oracle) + unanswerable (recusa/distrator) + parcial.
    ref_plan = {"oracle": 30, "recusa": 25, "distrator": 20, "parcial": 10}
    ref_items = run_pack(ref_chunks, ref_plan, valtest, args.url, args.model,
                         args.workers, args.seed + 1, "refusal")
    n_ans = sum(1 for it in ref_items if it["answerable"] and it["family"] == "oracle")
    n_unans = sum(1 for it in ref_items if not it["answerable"])
    ref_pack = {
        "metadata": {"task": "poc-sft-v2", "pack": "refusal_test", "n": len(ref_items),
                     "plan": ref_plan, "n_answerable": n_ans, "n_unanswerable": n_unans,
                     "note": "fora do treino e do holdout (valtest bloco B)"},
        "items": ref_items,
    }
    Path(args.refusal_out).write_text(json.dumps(ref_pack, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    print(f"\nval_pack: {len(val_items)} | refusal_test: {len(ref_items)} "
          f"(answerable={n_ans}, unanswerable={n_unans}) | {(time.time()-t0)/60:.1f}min",
          flush=True)


if __name__ == "__main__":
    main()
