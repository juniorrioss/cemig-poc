#!/usr/bin/env python3
"""
score_p2.py — julgamento (juiz 27B, régua honesta) da avaliação do P2.

Reusa SEM alterar: `bench/lfm3_f2/eval/score_f2.py` (4 células padrão),
`bench/lfm3_f6/score_f6.py` (e2e op_d50, cita item-ouro) e
`bench/lfm3_f2/probe/run_probe.py judge` (probe geral). Só redireciona
entradas/saídas para `--work/eval` (gerações puxadas do container).

Também calcula, por tag, o REWARD do contrato (`verifier.py`) nas células
estáticas como métrica auxiliar SEM juiz.

Uso: classifier/.venv/bin/python bench/lfm3_p1p2/eval/score_p2.py --tag p2-base
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable
CELLS = ("ranking", "distrator", "oraculo", "closed_book")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", default="/mnt/d/cemig-scratch/p1p2/eval")
    ap.add_argument("--skip", default="", help="ex.: probe,e2e")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    work = Path(args.work)
    skip = set(args.skip.split(",")) if args.skip else set()

    # --- 4 células: score_f2 lê F2/data/gen e escreve F2/data/scored -> shim
    sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))
    root = work / "f2root"
    (root / "data" / "gen").mkdir(parents=True, exist_ok=True)
    for c in CELLS:
        src = work / f"gen_{args.tag}_{c}.jsonl"
        if src.exists():
            shutil.copy(src, root / "data" / "gen" / src.name)
    code = ("import sys; sys.path.insert(0, %r); import score_f2 as S; from pathlib import Path; "
            "S.F2 = Path(%r); sys.argv = ['x'] + sys.argv[1:]; S.main()"
            % (str(REPO / "bench" / "lfm3_f2" / "eval"), str(root)))
    for c in CELLS:
        if (root / "data" / "gen" / f"gen_{args.tag}_{c}.jsonl").exists():
            subprocess.run([PY, "-c", code, "--tag", args.tag, "--cell", c,
                            "--workers", str(args.workers)], check=True)

    # --- e2e op_d50 (score_f6, suite recorte)
    if "e2e" not in skip and (work / f"e2e_gen_{args.tag}.jsonl").exists():
        e2e_dir = work / "f6root" / "data" / "e2e"
        e2e_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(work / f"e2e_gen_{args.tag}.jsonl", e2e_dir / f"gen_{args.tag}.jsonl")
        code6 = ("import sys; sys.path.insert(0, %r); import common_f6 as F6; from pathlib import Path; "
                 "F6.DATA = Path(%r); F6.CACHE = Path(%r); import score_f6 as S; S.F6 = F6; "
                 "sys.argv = ['x'] + sys.argv[1:]; S.main()"
                 % (str(REPO / "bench" / "lfm3_f6"), str(work / "f6root" / "data"),
                    str(work / "f6root" / "cache")))
        subprocess.run([PY, "-c", code6, "--suite", "recorte", "--tag", args.tag,
                        "--workers", str(args.workers)], check=True)

    # --- probe (run_probe judge lê/escreve ao lado do script -> shim de HERE)
    if "probe" not in skip and (work / f"probe_gen_{args.tag}.jsonl").exists():
        pdir = work / "probe"
        pdir.mkdir(exist_ok=True)
        shutil.copy(work / f"probe_gen_{args.tag}.jsonl", pdir / f"probe_gen_{args.tag}.jsonl")
        codep = ("import sys; sys.path.insert(0, %r); import run_probe as P; from pathlib import Path; "
                 "P.HERE = Path(%r); "
                 "sys.argv = ['x', 'judge', '--tag', %r]; P.main()"
                 % (str(REPO / "bench" / "lfm3_f2" / "probe"), str(pdir), args.tag))
        # load_items lê os JSONs congelados de HERE; judge lê gens de HERE -> copia JSONs
        for f in ("ood_general.json", "ood_behavioral.json"):
            shutil.copy(REPO / "bench" / "lfm3_f2" / "probe" / f, pdir / f)
        subprocess.run([PY, "-c", codep], check=True)
    print(f"[ok] {args.tag}")


if __name__ == "__main__":
    main()
