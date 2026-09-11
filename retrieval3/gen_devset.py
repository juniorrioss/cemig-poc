#!/usr/bin/env python3
"""
gen_devset.py — dev-set sintético próprio p/ CALIBRAÇÃO (k do RRF, pesos, thresholds).

O holdout (151 + 20) é INTOCÁVEL. Toda calibração usa este dev-set sintético, gerado
de chunks amostrados do índice, com gold = o chunk-fonte (chunk_id + doc + section).

ANTI-CONTAMINAÇÃO: o prompt gera a fala a partir SÓ do texto normativo do chunk; nunca
vê o holdout. Para evitar viés otimista (a fala "copiar" o vocabulário do chunk — erro
histórico do dataset invertido, ver AGENTS.md finetune), o prompt exige linguagem 100%
coloquial de campo, com gírias, SEM repetir os termos técnicos do trecho.

Uso:
    python3 retrieval3/gen_devset.py --n 200 --seed 13 --out retrieval3/data/dev_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "Você é um eletricista/operário brasileiro de campo, leigo em termos jurídicos. "
    "Dado um TRECHO de uma Norma Regulamentadora, formule UMA pergunta curta, falada, "
    "do jeito que você perguntaria em voz alta no serviço — com gíria e linguagem do dia a dia.\n"
    "Regras rígidas:\n"
    "1. NÃO use os termos técnicos do trecho (ex.: se o trecho fala 'talabarte', você diz 'cinto/corda'; "
    "se fala 'desenergização', você diz 'desligar a energia').\n"
    "2. NÃO cite números de item nem o nome/número da norma.\n"
    "3. Uma única pergunta, curta e natural. Responda SOMENTE a pergunta, sem aspas nem preâmbulo."
)

USER_TEMPLATE = (
    "TRECHO (seção {section} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\nSua pergunta de campo:"
)


def call_vllm(base_url: str, model: str, user: str, timeout: int = 90) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 96,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def clean_question(raw: str) -> str:
    q = raw.strip().strip('"').strip()
    q = q.split("\n")[0].strip()
    return q


def load_holdout_chunk_ids() -> set:
    """IDs de chunk-ouro do holdout — excluídos da amostragem do dev-set (higiene)."""
    ids = set()
    for name in ["corpus/qa_pairs_v2.jsonl", "bench/data/smoke_qa_20.jsonl"]:
        p = _HERE.parent / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("chunk_id", -1) >= 0:
                ids.add(r["chunk_id"])
            for cid in r.get("relevant_chunk_ids", []):
                ids.add(cid)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE.parent / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--out", default=str(_HERE / "data" / "dev_set.jsonl"))
    ap.add_argument("--base-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-len", type=int, default=250, help="len mínimo do texto do chunk")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks").fetchall()]
    con.close()

    holdout_ids = load_holdout_chunk_ids()
    pool = [r for r in rows if r["id"] not in holdout_ids and len(r["text"]) >= args.min_len]
    random.seed(args.seed)
    random.shuffle(pool)
    sample = pool[:args.n]
    print(f"chunks totais={len(rows)} elegíveis={len(pool)} amostrados={len(sample)} "
          f"(excluídos {len(holdout_ids)} chunk-ouro do holdout)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# " + json.dumps({
                "artifact": "dev_set_synthetic", "task": "poc-retrieval-v3",
                "purpose": "calibration_only_never_holdout",
                "model": args.model, "endpoint": args.base_url,
                "seed": args.seed, "created_utc": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    ok = 0
    for i, c in enumerate(sample, 1):
        if c["id"] in done:
            continue
        try:
            raw = call_vllm(args.base_url, args.model, USER_TEMPLATE.format(
                section=c["section"], title=c["title"], text=c["text"][:1800]))
            q = clean_question(raw)
            if len(q) < 8:
                continue
            rec = {"id": f"dev-{c['id']}", "question": q, "doc": c["doc"],
                   "section": c["section"], "chunk_id": c["id"], "relevant_chunk_ids": [c["id"]]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
        except Exception as e:
            print(f"  falha chunk {c['id']}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} ok={ok} {i/(time.time()-t0):.1f}/s", flush=True)
    fh.close()
    print(f"dev-set pronto: {ok} novas perguntas em {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
