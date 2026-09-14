#!/usr/bin/env python3
"""
build_eval_sets.py — PASSO 4: monta as suites de avaliação (métricas NOVAS do tool-calling).

Fontes 100% FORA do treino (muralhas): chunks de VAL/refusal (sft_v2.split_chunks, os mesmos
reservados no treino), NRs reservadas (33/16/26) e o holdout 151 (só para MEDIR argumento —
nunca treina nem calibra). Todas as perguntas passam pelo filtro lexical vs holdout quando
geradas sinteticamente.

Suites geradas (data/eval_*.json):
  a) eval_decision.json  — matriz "chamar vs deveria chamar":
       - positivos (deveria CHAMAR): perguntas técnicas de VAL-chunks (novas) + casos do capitão.
       - negativos (NÃO deveria): saudação/agradecimento/repetição/bom senso/fora de escopo.
  b) eval_args.json      — QUALIDADE DOS ARGUMENTOS (a métrica central):
       as 151 do holdout com {question, gold_chunk_id, relevant_ids}. Mede recall@k de
       (i) pergunta crua, (ii) consulta do MODELO, (iii) consulta do 27B — todas no índice v4.
  c) eval_reuse.json     — REUSO no multiturno: diálogos de 2 turnos onde o 2º turno é sobre o
       MESMO assunto (deveria REUSAR, não chamar) OU muda de assunto (deveria chamar de novo).
  d) reusa os packs e2e do sft_v2 (oracle/retrieved/refusal) para aprovação/alucinação/recusa.

Higiene: paralelo, checkpoint, procedência. Comentários PT-BR; código em inglês.
Uso: ../classifier/.venv/bin/python build_eval_sets.py --n-decision 120 --n-reuse 80
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import common_tools as C
from gen_dialogs import (FOLLOWUP_NEW_SYSTEM, FOLLOWUP_NEW_USER, NOTOOL_SYSTEM, NOTOOL_USER,
                         SCHEMA_NEWQ, SCHEMA_NOTOOL, _gen_qa)

_HERE = Path(__file__).resolve().parent
_lock = Lock()

# Casos do capitão (positivos: DEVEM chamar) — falas cruas literais.
CAPTAIN_CASES = [
    {"id": "cap_13_8kv", "question": "qual a distância segura pra trabalhar perto de rede de 13,8 kV?"},
    {"id": "cap_poste", "question": "o poste que eu preciso subir não parece firme, e agora?"},
    {"id": "cap_camiseta", "question": "posso trabalhar de camiseta rasgada nesse calor?"},
]


def load_holdout_args() -> List[Dict[str, Any]]:
    """151 do holdout: {id, question, gold_chunk_id, relevant_ids, doc, section}. SÓ p/ medir."""
    qa = C._ft.QA_V2_PATH  # corpus/qa_pairs_v2.jsonl
    out = []
    for line in Path(qa).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gid = r.get("chunk_id", -1)
        out.append({"id": r["id"], "question": r["question"], "gold_chunk_id": gid,
                    "relevant_ids": r.get("relevant_chunk_ids", []) or [gid],
                    "doc": str(r.get("doc", "")).lower(), "section": str(r.get("section", ""))})
    return out


def gen_positive_question(chunk: Dict[str, Any], url: str, model: str, lf) -> Optional[Dict[str, Any]]:
    """Pergunta técnica coloquial (deveria CHAMAR) de um chunk de VAL (fora do treino)."""
    qa = _gen_qa(chunk, url, model)
    if not qa:
        return None
    q, facts = qa
    with _lock:
        if not lf.accept(q, meta={"src": chunk["id"], "suite": "decision_pos"}):
            return None
    return {"id": f"pos_{chunk['id']}", "question": q, "should_call": True,
            "gold_chunk_id": chunk["id"], "doc": chunk["doc"], "section": chunk["section"],
            "facts": facts, "gold_text": chunk["text"]}


def gen_negative_question(cat: str, url: str, model: str) -> Optional[Dict[str, Any]]:
    """Pergunta que NÃO deveria chamar (saudação/agradecimento/repetição/bom senso/fora)."""
    obj = C.call_vllm_json(
        [{"role": "system", "content": NOTOOL_SYSTEM},
         {"role": "user", "content": NOTOOL_USER.format(categoria=cat)}],
        SCHEMA_NOTOOL, url=url, model=model, temperature=0.9, max_tokens=200)
    if not obj:
        return None
    q = " ".join(str(obj.get("pergunta", "")).strip().split())
    if len(q) < 3:
        return None
    return {"id": f"neg_{cat}_{abs(hash(q)) % 100000}", "question": q, "should_call": False,
            "categoria": cat}


def gen_reuse_dialog(chunk: Dict[str, Any], url: str, model: str, lf) -> Optional[Dict[str, Any]]:
    """Diálogo de 2 turnos: t1 técnico (chama); t2 MESMO assunto (deveria REUSAR)."""
    from gen_dialogs import FOLLOWUP_REUSE_SYSTEM, FOLLOWUP_REUSE_USER, SCHEMA_FOLLOWUP
    qa = _gen_qa(chunk, url, model)
    if not qa:
        return None
    q1, facts = qa
    with _lock:
        if not lf.accept(q1, meta={"src": chunk["id"], "suite": "reuse_t1"}):
            return None
    obj = C.call_vllm_json(
        [{"role": "system", "content": FOLLOWUP_REUSE_SYSTEM},
         {"role": "user", "content": FOLLOWUP_REUSE_USER.format(
             doc=chunk["doc"], sec=chunk["section"], title=chunk["title"],
             text=chunk["text"][:1500], q1=q1)}],
        SCHEMA_FOLLOWUP, url=url, model=model, temperature=0.8, max_tokens=256)
    if not obj:
        return None
    q2 = " ".join(str(obj.get("pergunta", "")).strip().split())
    if len(q2) < 4:
        return None
    return {"id": f"reuse_{chunk['id']}", "turns": [q1, q2], "reuse_turn": 2,
            "should_call_turn2": False, "gold_chunk_id": chunk["id"],
            "doc": chunk["doc"], "section": chunk["section"], "facts": facts,
            "gold_text": chunk["text"]}


def gen_newtopic_dialog(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any], url: str,
                        model: str, lf) -> Optional[Dict[str, Any]]:
    """Diálogo de 2 turnos: t1 assunto A (chama); t2 assunto B DIFERENTE (deveria chamar de novo)."""
    qa = _gen_qa(chunk_a, url, model)
    if not qa:
        return None
    q1, _ = qa
    with _lock:
        if not lf.accept(q1, meta={"src": chunk_a["id"], "suite": "newtopic_t1"}):
            return None
    obj = C.call_vllm_json(
        [{"role": "system", "content": FOLLOWUP_NEW_SYSTEM},
         {"role": "user", "content": FOLLOWUP_NEW_USER.format(
             q1=q1, doc=chunk_b["doc"], sec=chunk_b["section"], title=chunk_b["title"],
             text=chunk_b["text"][:1300])}],
        SCHEMA_NEWQ, url=url, model=model, temperature=0.85, max_tokens=140)
    if not obj:
        return None
    q2 = " ".join(str(obj.get("pergunta", "")).strip().split())
    if len(q2) < 4:
        return None
    return {"id": f"newtopic_{chunk_a['id']}_{chunk_b['id']}", "turns": [q1, q2],
            "reuse_turn": None, "should_call_turn2": True, "gold_chunk_id": chunk_b["id"],
            "doc": chunk_b["doc"], "section": chunk_b["section"]}


def _parallel(fn, tasks, workers, target, label):
    out = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                out.append(r)
            if len(out) >= target:
                for f in futs:
                    f.cancel()
                break
    print(f"[{label}] {len(out)}/{target} ({time.time()-t0:.0f}s)", flush=True)
    return out[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--n-decision-pos", type=int, default=90)
    ap.add_argument("--n-decision-neg", type=int, default=60)
    ap.add_argument("--n-reuse", type=int, default=70)
    ap.add_argument("--n-newtopic", type=int, default=50)
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # VAL-chunks (fora do treino) + chunks das NRs reservadas p/ OOD estrutural na decisão.
    _, valtest = C.V2.split_chunks(seed=17, min_len=200)  # MESMA seed do treino (reserva idêntica)
    reserved = [c for c in C.load_chunks() if c["doc"] in C.RESERVED_NRS and len(c["text"]) >= 200]
    val_pool = valtest + reserved
    rng.shuffle(val_pool)
    lf = C.LexicalFilter()

    # a) decisão — positivos
    pos_tasks = val_pool[: int(args.n_decision_pos * 2.2) + 10]
    pos = _parallel(lambda c: gen_positive_question(c, args.url, args.model, lf),
                    pos_tasks, args.workers, args.n_decision_pos, "decision_pos")
    for cc in CAPTAIN_CASES:
        pos.append({**cc, "should_call": True, "captain_case": True})
    # negativos
    cats = ["saudacao", "agradecimento", "repeticao", "bom_senso", "fora_de_escopo"]
    neg_tasks = [cats[i % len(cats)] for i in range(int(args.n_decision_neg * 1.6) + 10)]
    neg = _parallel(lambda cat: gen_negative_question(cat, args.url, args.model),
                    neg_tasks, args.workers, args.n_decision_neg, "decision_neg")
    decision = {"metadata": {"suite": "decision", "created": _now(),
                             "n_pos": len(pos), "n_neg": len(neg)},
                "items": pos + neg}
    _dump("data/eval_decision.json", decision)

    # b) argumentos — holdout 151 (só medição)
    args_set = {"metadata": {"suite": "args", "note": "holdout 151 SÓ para medir recall",
                             "created": _now()}, "items": load_holdout_args()}
    _dump("data/eval_args.json", args_set)

    # c) reuso + novo-tópico
    reuse_tasks = val_pool[: int(args.n_reuse * 3.0) + 10]
    reuse = _parallel(lambda c: gen_reuse_dialog(c, args.url, args.model, lf),
                      reuse_tasks, args.workers, args.n_reuse, "reuse")
    nt_tasks = [(val_pool[i % len(val_pool)],
                 rng.choice([c for c in val_pool if c["doc"] != val_pool[i % len(val_pool)]["doc"]]))
                for i in range(int(args.n_newtopic * 3.0) + 10)]
    newtopic = _parallel(lambda ab: gen_newtopic_dialog(ab[0], ab[1], args.url, args.model, lf),
                         nt_tasks, args.workers, args.n_newtopic, "newtopic")
    reuse_set = {"metadata": {"suite": "reuse", "created": _now(),
                              "n_reuse": len(reuse), "n_newtopic": len(newtopic)},
                 "items": reuse + newtopic}
    _dump("data/eval_reuse.json", reuse_set)

    lf.dump_discards(_HERE / "data" / "eval_lexical_discards.json")
    print("suites de avaliação prontas.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(rel: str, obj: Dict[str, Any]) -> None:
    p = _HERE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"salvo {rel}: {len(obj['items'])} itens", flush=True)


if __name__ == "__main__":
    main()
