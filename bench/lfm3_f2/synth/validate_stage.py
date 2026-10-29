"""
validate_stage.py — validador leve por estágio do funil (sem LLM, segundos,
GATE: exit 1 ruidoso em violação). Ordem do firstmate após auditoria amostral
(bug do s3: 33% dos "sem_ref" tinham referência explícita).

Checks:
  1. consistência style vs regex de referência (NR-\\d|item \\d|norma|anexo):
     <2% de violação;
  2. entropia dos 3 primeiros tokens das perguntas >= piso (default 6.0 bits
     para estágios de QA; registrado via --min-entropy);
  3. share da abertura (1º token) mais comum < 5% — frouxo p/ PT ("qual/o/em"
     são inevitáveis em parte): aplicado ao PREFIXO de 3 tokens;
  4. near-dup intra-estágio (assinatura prefixo-8 + jaccard>=0.85) < 10%;
  5. rastreabilidade: unit_id e ref presentes em 100% das linhas;
  6. JSON schema: campos obrigatórios por tipo de linha.

Uso:
    python3 validate_stage.py data/synth/qa_raw_nr10.jsonl [--min-entropy 6.0]
    python3 validate_stage.py data/synth/units_nr10.jsonl --kind units
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import jaccard, norm_tokens  # noqa: E402

REF_RE = re.compile(r"\bNR[- ]?\d+|\bitem \d|\bnorma regulamentadora|\banexo \b",
                    re.IGNORECASE)

QA_FIELDS = {"qa_id", "nr", "unit_id", "ref", "question", "answer", "style", "origin"}
UNIT_FIELDS = {"unit_id", "nr", "ref", "fato", "tipo", "densa"}
PARA_FIELDS = {"para_id", "nr", "registro", "text"}


def entropy_bits(prefixes: list[str]) -> float:
    counts = collections.Counter(prefixes)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def fail(msg: str) -> None:
    print(f"[GATE-FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["qa", "units", "para"], default="qa")
    ap.add_argument("--min-entropy", type=float, default=6.0)
    ap.add_argument("--max-style-viol", type=float, default=0.02)
    ap.add_argument("--max-top-prefix", type=float, default=0.05)
    ap.add_argument("--max-neardup", type=float, default=0.10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        fail(f"{args.path}: vazio")
    print(f"[validate] {args.path}: {len(rows)} linhas, kind={args.kind}")

    # 6. schema
    required = {"qa": QA_FIELDS, "units": UNIT_FIELDS, "para": PARA_FIELDS}[args.kind]
    for i, r in enumerate(rows):
        missing = required - set(r)
        if missing:
            fail(f"linha {i}: campos ausentes {missing}")
    print("  [ok] 6. schema")

    # 5. rastreabilidade (qa/units)
    if args.kind in ("qa", "units"):
        key = "unit_id"
        n_bad = sum(1 for r in rows if not r.get(key) or not r.get("ref"))
        if n_bad:
            fail(f"rastreabilidade: {n_bad} linhas sem {key}/ref")
        print("  [ok] 5. rastreabilidade unit_id+ref 100%")

    if args.kind != "qa":
        print("[PASS] (checks de pergunta não se aplicam)")
        return

    questions = [r["question"] for r in rows]

    # 1. style vs regex
    viol = sum(1 for r in rows
               if (r["style"] == "sem_ref" and REF_RE.search(r["question"]))
               or (r["style"] == "com_ref" and not REF_RE.search(r["question"])))
    rate = viol / len(rows)
    if rate >= args.max_style_viol:
        fail(f"style x referência: {viol}/{len(rows)} ({rate:.1%}) >= {args.max_style_viol:.0%}")
    print(f"  [ok] 1. style x referência: {rate:.2%}")

    # 2. entropia prefixo-3
    prefixes3 = [" ".join(norm_tokens(q)[:3]) for q in questions]
    h = entropy_bits(prefixes3)
    if h < args.min_entropy:
        fail(f"entropia prefixo-3 = {h:.2f} bits < piso {args.min_entropy}")
    print(f"  [ok] 2. entropia prefixo-3: {h:.2f} bits")

    # 3. share do prefixo mais comum
    top, top_n = collections.Counter(prefixes3).most_common(1)[0]
    share = top_n / len(prefixes3)
    if share >= args.max_top_prefix:
        fail(f"prefixo mais comum '{top}' = {share:.1%} >= {args.max_top_prefix:.0%}")
    print(f"  [ok] 3. top-prefixo '{top}': {share:.2%}")

    # 4. near-dup (bucketed por assinatura p/ ser O(n*k))
    sig_buckets: dict[str, list[int]] = collections.defaultdict(list)
    for i, q in enumerate(questions):
        sig_buckets[" ".join(norm_tokens(q)[:8])].append(i)
    n_dup = 0
    for idxs in sig_buckets.values():
        for a in range(1, len(idxs)):
            if any(jaccard(questions[idxs[a]], questions[idxs[b]]) >= 0.85
                   for b in range(a)):
                n_dup += 1
    dup_rate = n_dup / len(questions)
    if dup_rate >= args.max_neardup:
        fail(f"near-dup: {n_dup} ({dup_rate:.1%}) >= {args.max_neardup:.0%}")
    print(f"  [ok] 4. near-dup: {dup_rate:.2%}")

    print("[PASS]")


if __name__ == "__main__":
    main()
