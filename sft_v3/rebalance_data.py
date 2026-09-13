#!/usr/bin/env python3
"""
rebalance_data.py — PARTE 1 do SFT v3: rebalanceia os dados para recusa <= 25% no total.

Decisão do capitão após ver o v2 ("podemos manter os exemplos de recusa em 25% ao máximo"):
o v2 tinha 52% de famílias de recusa (recusa 20,8 + parcial 20,8 + distrator 10,4) e o efeito
colateral foi recusa-indevida de 23-40% (recusa até o óbvio). O v3 aperta para <= 25%.

ALVO v3 (total ~3050, famílias de recusa 24,6% < 25%):
  - oráculo   2300 (75,4%)  — o dobro do peso: é onde a APROVAÇÃO mora.
  - recusa     300 ( 9,8%)  — chunk de OUTRA norma (falha real mais comum: retrieval erra a NR).
  - parcial    300 ( 9,8%)  — dado-chave removido (2ª falha real: chunk certo, dado ausente).
  - distrator  150 ( 4,9%)  — mesma norma, seção vizinha (o caso camiseta; o mais raro/difícil).

JUSTIFICATIVA DA DIVISÃO (pedida no brief): recusa e parcial são os dois modos de falha de
retrieval mais frequentes no campo (norma errada / dado ausente) → 10% cada; distrator é o
mais raro e o mais difícil de discriminar, e o v2 já provou que 10% bastava para NÃO cometer
o erro da camiseta → 5%. Somam 24,6%, folga segura sob o teto de 25%.

REUSO (ordem do brief: NÃO refazer o que já existe): as famílias de recusa e o oráculo do v2
já foram filtrados por régua (3076 pares aprovados em sft_v2/data/train_v2.jsonl). Este script:
  1. mantém TODOS os 1476 oráculos do v2 (já régua-aprovados);
  2. subamostra as famílias de recusa do pool v2 com SEED FIXA (300/300/150);
  3. gera oráculo FRESCO só do déficit (2300-1476 = 824) de chunks de treino AINDA NÃO usados
     como fonte de oráculo no v2 (reusa gen_data.gen_oracle — mesma régua honesta, 27B).

Muralhas herdadas do v2 (via common_v2/finetune2): holdout-ouro fora, NR-33/16/26 reservadas,
Jaccard<0.4 vs holdout, chunks de val/refusal FORA do treino. Higiene: paralelo, checkpoint,
não-interativo, procedência. Comentários PT-BR; código em inglês.

Uso (classifier/.venv; juiz 27B default 10.100.0.111:8005):
  ../classifier/.venv/bin/python rebalance_data.py \
     --v2 ../sft_v2/data/train_v2.jsonl --out data/train_v3.jsonl \
     --target-oracle 2300 --target-recusa 300 --target-parcial 300 --target-distrator 150
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
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent / "sft_v2"
sys.path.insert(0, str(_V2))

import common_v2 as C  # noqa: E402
from gen_data import gen_oracle  # noqa: E402  (mesma régua honesta do v2/v1)

_lock = Lock()


def load_v2_pool(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Carrega o pool v2 agrupado por família (pula o cabeçalho comentado)."""
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        d = json.loads(s)
        fam = d["meta"].get("family", "?")
        by_fam.setdefault(fam, []).append(d)
    return by_fam


def subsample(pool: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Subamostra n exemplos com seed fixa (reprodutível)."""
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)
    return items[: min(n, len(items))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(_V2 / "data" / "train_v2.jsonl"))
    ap.add_argument("--out", default=str(_HERE / "data" / "train_v3.jsonl"))
    ap.add_argument("--target-oracle", type=int, default=2300)
    ap.add_argument("--target-recusa", type=int, default=300)
    ap.add_argument("--target-parcial", type=int, default=300)
    ap.add_argument("--target-distrator", type=int, default=150)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=200)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v2 = load_v2_pool(Path(args.v2))
    print("pool v2:", {k: len(v) for k, v in v2.items()}, flush=True)

    targets = {"oracle": args.target_oracle, "recusa": args.target_recusa,
               "parcial": args.target_parcial, "distrator": args.target_distrator}
    total = sum(targets.values())
    print(f"alvos v3: {targets} | total {total} | "
          f"recusa-famílias {100*(total-targets['oracle'])/total:.1f}%", flush=True)

    # ------------------------------------------------------------------ #
    # 1. Famílias de recusa: subamostra do pool v2 (já régua-aprovado).
    # ------------------------------------------------------------------ #
    kept: List[Dict[str, Any]] = []
    for fam in ("recusa", "parcial", "distrator"):
        sub = subsample(v2.get(fam, []), targets[fam], args.seed)
        kept.extend(sub)
        print(f"[{fam}] reusado do v2: {len(sub)}/{targets[fam]}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Oráculo: mantém TODOS os do v2 + gera o déficit fresco.
    # ------------------------------------------------------------------ #
    v2_oracle = v2.get("oracle", [])
    kept.extend(v2_oracle)
    used_oracle_chunks = {d["meta"].get("src_chunk_id") for d in v2_oracle}
    deficit = max(0, targets["oracle"] - len(v2_oracle))
    print(f"[oracle] reusado do v2: {len(v2_oracle)} | déficit a gerar: {deficit}", flush=True)

    # Escreve o que já temos (checkpoint) antes de gerar o resto.
    out_path.write_text("# " + json.dumps({
        "artifact": "sft_v3_rebalanced", "task": "poc-sft-v3", "model": args.model,
        "threshold": args.threshold, "seed": args.seed, "targets": targets,
        "refusal_pct_target": round(100 * (total - targets["oracle"]) / total, 1),
        "reused_from_v2": {"oracle": len(v2_oracle), "recusa": targets["recusa"],
                           "parcial": targets["parcial"], "distrator": targets["distrator"]},
        "reserved_nrs": C.RESERVED_NRS,
        "created_utc": datetime.now(timezone.utc).isoformat()},
        ensure_ascii=False) + "\n", encoding="utf-8")
    fh = out_path.open("a", encoding="utf-8")
    for rec in kept:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()

    if deficit > 0:
        # chunks de treino AINDA NÃO usados como fonte de oráculo no v2.
        train_chunks, _ = C.split_chunks(seed=args.seed, min_len=args.min_len)
        fresh = [c for c in train_chunks if c["id"] not in used_oracle_chunks]
        print(f"[oracle] chunks frescos disponíveis: {len(fresh)}", flush=True)
        lf = C.LexicalFilter()
        overshoot = 1.7  # oráculo aprova bem, mas alguns reprovam na régua
        n_tasks = int(deficit * overshoot) + 20
        tasks = fresh[:n_tasks]
        got = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(gen_oracle, c, args.url, args.model, lf, args.threshold): c["id"]
                    for c in tasks}
            tried = 0
            for fut in as_completed(futs):
                tried += 1
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                if rec is not None:
                    with _lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        got += 1
                if tried % 40 == 0:
                    rate = tried / max(1e-9, time.time() - t0)
                    print(f"  [oracle-fresh] tried={tried}/{len(tasks)} kept={got}/{deficit} "
                          f"{rate:.1f}/s", flush=True)
                if got >= deficit:
                    for f in futs:
                        f.cancel()
                    break
        print(f"[oracle-fresh] final: {got}/{deficit} (tried {tried}, "
              f"{(time.time()-t0)/60:.1f}min)", flush=True)

    fh.close()

    # ------------------------------------------------------------------ #
    # 3. Relatório final por família.
    # ------------------------------------------------------------------ #
    final: Dict[str, int] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        fam = json.loads(s)["meta"].get("family", "?")
        final[fam] = final.get(fam, 0) + 1
    tot = sum(final.values())
    print(f"\nSFT v3 rebalanceado: {final} | total {tot}", flush=True)
    ref_pct = 100 * (tot - final.get("oracle", 0)) / max(1, tot)
    print(f"famílias de recusa: {ref_pct:.1f}% (teto do capitão: 25%) -> "
          f"{'OK' if ref_pct <= 25.0 else 'ACIMA DO TETO!'}", flush=True)


if __name__ == "__main__":
    main()
