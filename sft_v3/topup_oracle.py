#!/usr/bin/env python3
"""
topup_oracle.py — completa o déficit de ORÁCULO do v3 gerando MÚLTIPLAS perguntas por chunk.

O rebalance esgotou os 260 chunks de treino ainda-não-usados como fonte de oráculo (só 139
aprovaram na régua). O brief autoriza "se faltar oráculo, gere mais". Um mesmo trecho de NR
cobre vários fatos → gera-se, com temperatura 0.8, PERGUNTAS COLOQUIAIS DISTINTAS sobre o
mesmo chunk (cada uma filtrada pela régua honesta). Deduplicação por pergunta (Jaccard) evita
quase-duplicatas. Fonte: TODO o pool de treino (chunks já usados no v2 podem render outra
pergunta sobre outro fato — a régua e a dedup garantem qualidade e diversidade).

Muralhas idênticas ao v2 (holdout-ouro fora, NR reservadas fora, val/refusal fora do treino).
Anexa ao mesmo train_v3.jsonl (checkpoint incremental). Higiene: paralelo, não-interativo.
Comentários PT-BR; código em inglês.

Uso: ../classifier/.venv/bin/python topup_oracle.py --out data/train_v3.jsonl --need 700
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent / "sft_v2"
sys.path.insert(0, str(_V2))

import common_v2 as C  # noqa: E402
from gen_data import gen_oracle  # noqa: E402

_lock = Lock()


def existing_questions(path: Path) -> Set[str]:
    """Conjunto de perguntas de oráculo já no arquivo (dedup por texto normalizado)."""
    qs: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = json.loads(s)["meta"]
        if m.get("family") == "oracle" and m.get("question"):
            qs.add(m["question"].strip().lower())
    return qs


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "data" / "train_v3.jsonl"))
    ap.add_argument("--need", type=int, default=700)
    ap.add_argument("--rounds", type=int, default=4, help="passes sobre o pool de treino")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=200)
    args = ap.parse_args()

    out_path = Path(args.out)
    seen_q = existing_questions(out_path)
    print(f"oráculo já presente: {len(seen_q)} perguntas | déficit alvo: {args.need}", flush=True)

    train_chunks, _ = C.split_chunks(seed=args.seed, min_len=args.min_len)
    # prioriza seções numéricas (Q&A de campo melhores) e embaralha por round.
    rng = random.Random(args.seed)
    lf = C.LexicalFilter()
    fh = out_path.open("a", encoding="utf-8")
    got = 0
    t0 = time.time()

    for rnd in range(args.rounds):
        if got >= args.need:
            break
        pool = list(train_chunks)
        rng.shuffle(pool)
        remaining = args.need - got
        # gera mais tarefas que o necessário (régua reprova parte; dedup remove parte).
        n_tasks = min(len(pool), int(remaining * 2.2) + 30)
        tasks = pool[:n_tasks]
        print(f"== round {rnd+1}/{args.rounds}: {len(tasks)} chunks, faltam {remaining} ==",
              flush=True)

        def work(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # gen_oracle usa temperatura 0.8 na pergunta (diversidade) e 0.0 na resposta.
            return gen_oracle(c, args.url, args.model, lf, args.threshold)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, c): c["id"] for c in tasks}
            tried = 0
            for fut in as_completed(futs):
                tried += 1
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                if rec is not None:
                    q = rec["meta"]["question"].strip().lower()
                    # dedup: rejeita se muito parecida com alguma já aceita.
                    dup = q in seen_q or any(jaccard(q, e) >= 0.8 for e in seen_q)
                    if not dup:
                        with _lock:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                            seen_q.add(q)
                            got += 1
                if tried % 60 == 0:
                    rate = tried / max(1e-9, time.time() - t0)
                    print(f"  tried={tried}/{len(tasks)} kept={got}/{args.need} "
                          f"{rate:.1f}/s", flush=True)
                if got >= args.need:
                    for f in futs:
                        f.cancel()
                    break

    fh.close()
    print(f"\ntop-up oráculo: +{got} pares em {(time.time()-t0)/60:.1f}min", flush=True)

    # relatório final por família.
    final: Dict[str, int] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        fam = json.loads(s)["meta"].get("family", "?")
        final[fam] = final.get(fam, 0) + 1
    tot = sum(final.values())
    ref_pct = 100 * (tot - final.get("oracle", 0)) / max(1, tot)
    print(f"train_v3 agora: {final} | total {tot} | recusa-famílias {ref_pct:.1f}% "
          f"-> {'OK' if ref_pct <= 25.0 else 'ACIMA'}", flush=True)


if __name__ == "__main__":
    main()
