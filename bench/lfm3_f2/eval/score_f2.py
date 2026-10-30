"""
score_f2.py — régua honesta sobre as gerações F2. Porte direto do
bench/lfm3_f0/score_f0.py com: --tag (arquivos gen_{tag}_{cell}.jsonl),
gabarito/parquet do F0 reusados, saída data/scored/scored_{tag}_{cell}.jsonl.

Uso (local, juiz direto):
    classifier/.venv/bin/python bench/lfm3_f2/eval/score_f2.py \
        --tag f2-A-r8-ep1 --cell closed_book [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # bench/lfm3_f2/eval
F2 = HERE.parent
REPO = F2.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
sys.path.insert(0, str(REPO / "bench" / "regua"))

from common import parallel_map  # bench/regua
from judge_regua import judge_disambiguate  # bench/regua
from ruler import deterministic_pass  # bench/regua
from common_f0 import CELLS, load_rows, looks_like_refusal, read_jsonl  # lfm3_f0

GABARITO = REPO / "bench" / "lfm3_f0" / "data" / "gabarito_384.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parquet", default=str(
        Path.home() / ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/"
        "snapshots/5a5aa32ae4087481be828e071aff169827ec2993/"
        "normas_regulamentadoras/train-00000-of-00001.parquet"))
    args = ap.parse_args()

    gens = read_jsonl(F2 / "data" / "gen" / f"gen_{args.tag}_{args.cell}.jsonl")
    if args.limit:
        gens = gens[: args.limit]
    gab = {g["base_qa_id"]: g for g in read_jsonl(GABARITO)}
    rows = {r["id"]: r for r in load_rows(args.parquet)}
    out_dir = F2 / "data" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scored_{args.tag}_{args.cell}.jsonl"
    ckpt = out_dir / f"scored_{args.tag}_{args.cell}.ckpt.json"
    existing = {}
    if ckpt.exists():
        existing = json.loads(ckpt.read_text(encoding="utf-8"))
        print(f"[ckpt] retomando com {len(existing)}")
    print(f"[score] tag={args.tag} cell={args.cell} n={len(gens)}")

    def worker(g: dict) -> dict:
        row = rows[g["id"]]
        gb = gab[g["base_qa_id"]]
        facts = gb["facts"]
        evidence_text = "\n\n".join(e["text"] for e in row["evidence"])
        answer = g["answer"].replace("\u2011", "-").replace("\u2013", "-")
        question = row["question"]

        r = deterministic_pass(question, answer, facts)
        truncated = (not g["stopped_eos"]) and not answer.strip()
        refused = False if truncated else looks_like_refusal(answer)

        judge = None
        if answer.strip() and not r.empty:
            judge = judge_disambiguate(
                question, answer, r.facts_ambiguous, evidence_text)
            confirmed = set(f.lower() for f in judge["fatos_presentes"])
            extra = [f for f in r.facts_ambiguous if f.lower() in confirmed]
            n = max(1, len(facts))
            r.coverage = (len(r.facts_present) + len(extra)) / n
            r.hallucination = bool(judge["alucinacao"])

        answerable = args.cell != "distrator"
        if answerable:
            approved = (r.coverage >= 0.5 and not r.tautology
                        and not r.hallucination and not refused
                        and not truncated)
            refusal_correct = None
            refusal_undue = refused
        else:
            approved = refused and not truncated
            refusal_correct = refused and not truncated
            refusal_undue = None

        return {
            "id": g["id"], "base_qa_id": g["base_qa_id"],
            "vocabulary_level": g["vocabulary_level"],
            "question_type": g["question_type"], "cell": args.cell,
            "tag": args.tag,
            "answer": answer, "n_gen_tokens": g["n_gen_tokens"],
            "stopped_eos": g["stopped_eos"],
            "coverage": round(r.coverage, 4), "n_facts": len(facts),
            "tautology": r.tautology, "hallucination": r.hallucination,
            "refused": refused, "truncated": truncated,
            "refusal_correct": refusal_correct,
            "refusal_undue": refusal_undue,
            "citation_present": r.citation_present,
            "approved": approved, "empty": r.empty,
            "answered_correct_anyway": (not answerable and not refused
                                        and r.coverage >= 0.5),
            "judge_note": (judge or {}).get("justificativa", ""),
            "detail": r.detail,
        }

    results = parallel_map(worker, gens, max_workers=args.workers,
                           checkpoint_path=ckpt, key_fn=lambda g: g["id"],
                           existing=existing)
    ordered = [results[g["id"]] for g in gens if g["id"] in results]
    errs = [r for r in ordered if "__error__" in r]
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in ordered:
            if "__error__" not in rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ok] {out_path.name}: {len(ordered)-len(errs)} pontuados, {len(errs)} erros")


if __name__ == "__main__":
    main()
