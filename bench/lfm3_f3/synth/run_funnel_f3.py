"""
run_funnel_f3.py — roda o funil da F2 (s1 decompose -> s2 expand -> s4 critic)
para as NRs EXTRAS da F3, com saída em bench/lfm3_f3/data/synth e
subamostragem seedada para controlar custo do 27B (a F3 precisa de VARIEDADE
de perguntas p/ ancorar o retriever, não cobertura exaustiva da norma).

Reuso obrigatório do brief: os módulos s1/s2/s4 da F2 são IMPORTADOS (não
copiados); o redirecionamento de saída é feito patchando common_synth.SYNTH
ANTES do import dos estágios (o `from common_synth import SYNTH` deles liga
no valor já patchado).

Etapas por NR (6, 12, 17, 22, 35):
  1. s1_decompose --nr N          -> units_nrN.jsonl
  2. subsample_units              -> trunca a MAX_UNITS_PER_NR (seed fixa)
  3. s2_expand --nr N             -> qa_raw_nrN.jsonl
  4. subsample_qa                 -> MAX_QA_PER_UNIT por unidade (seed fixa)
  5. s4_critic --nr N             -> qa_passed_nrN.jsonl / qa_failed_nrN.jsonl
     (s3 socrático PULADO: o alvo da F3 é pergunta-âncora, não cadeia densa;
      decisão registrada no README)

Gates (validate_stage.py da F2) são rodados pelo Makefile/driver após cada
estágio. Dedup anticontaminação vs benchmark fica no build_dataset.py.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3/synth/run_funnel_f3.py \
        [--nr 6] [--stage all|s1|s2|s4] [--limit 2]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_f3 import (EXTRA_NRS, F3_SYNTH, MAX_QA_PER_UNIT,  # noqa: E402
                       MAX_UNITS_PER_NR, SEED, f2_synth_modules)

common_synth = f2_synth_modules()
read_jsonl = common_synth.read_jsonl
write_jsonl = common_synth.write_jsonl


def subsample_units(nr: int) -> None:
    """Trunca units_nrN a MAX_UNITS_PER_NR (sorteio seedado, estável)."""
    path = F3_SYNTH / f"units_nr{nr}.jsonl"
    units = read_jsonl(path)
    if len(units) <= MAX_UNITS_PER_NR:
        print(f"[subsample-units] NR-{nr}: {len(units)} <= {MAX_UNITS_PER_NR}, mantém")
        return
    rng = random.Random(f"{SEED}:units:{nr}")
    keep = rng.sample(units, MAX_UNITS_PER_NR)
    # preserva ordem original p/ reprodutibilidade de leitura
    keep_ids = {u["unit_id"] for u in keep}
    kept = [u for u in units if u["unit_id"] in keep_ids]
    write_jsonl(path, kept)
    print(f"[subsample-units] NR-{nr}: {len(units)} -> {len(kept)}")


def subsample_qa(nr: int) -> None:
    """Mantém MAX_QA_PER_UNIT QAs por unidade (sorteio seedado) antes do crítico."""
    path = F3_SYNTH / f"qa_raw_nr{nr}.jsonl"
    rows = read_jsonl(path)
    by_unit: dict[str, list] = {}
    for r in rows:
        by_unit.setdefault(r["unit_id"], []).append(r)
    kept = []
    for uid, qs in sorted(by_unit.items()):
        rng = random.Random(f"{SEED}:qa:{uid}")
        kept.extend(qs if len(qs) <= MAX_QA_PER_UNIT
                    else rng.sample(qs, MAX_QA_PER_UNIT))
    write_jsonl(path, kept)
    print(f"[subsample-qa] NR-{nr}: {len(rows)} -> {len(kept)} "
          f"({len(by_unit)} unidades)")


def run_stage(mod_name: str, nr: int, workers: int, limit: int) -> None:
    """Importa o estágio da F2 (com SYNTH já patchado) e roda via sys.argv."""
    import importlib
    mod = importlib.import_module(mod_name)
    argv = ["x", "--nr", str(nr), "--workers", str(workers)]
    if limit:
        argv += ["--limit", str(limit)]
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old


def ensure_empty_socratic(nr: int) -> None:
    """s4 lê qa_socratic_nrN.jsonl; a F3 pula o s3 -> arquivo vazio."""
    p = F3_SYNTH / f"qa_socratic_nr{nr}.jsonl"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=0, help="só uma NR (default: todas extras)")
    ap.add_argument("--stage", default="all", choices=["all", "s1", "s2", "s4"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="limita janelas/unidades (smoke)")
    args = ap.parse_args()

    nrs = [args.nr] if args.nr else EXTRA_NRS
    for nr in nrs:
        if args.stage in ("all", "s1"):
            run_stage("s1_decompose", nr, min(args.workers, 12), args.limit)
            subsample_units(nr)
        if args.stage in ("all", "s2"):
            run_stage("s2_expand", nr, args.workers, args.limit)
            subsample_qa(nr)
        if args.stage in ("all", "s4"):
            ensure_empty_socratic(nr)
            run_stage("s4_critic", nr, max(args.workers, 24), args.limit)


if __name__ == "__main__":
    main()
