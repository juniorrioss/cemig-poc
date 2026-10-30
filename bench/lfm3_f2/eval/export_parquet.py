"""
export_parquet.py — publica os parquets canônicos de avaliação (data/eval/),
um por checkpoint×célula, com o SCHEMA ESTENDIDO de transparência (ordem do
capitão: "ninguém pode comparar células sem saber o que o modelo viu"):

  colunas do benchmark (id, base_qa_id, question, answer_gold, question_type,
  source_id, vocabulary_level, query_features, source_file, evidence_text)
+ answer_model      : resposta final avaliada
+ raw_output        : saída integral do modelo (thinking + resposta)
+ thinking          : rascunho <think> separado
+ colunas de veredito (coverage..approved, judge_note)
+ context_given     : texto EXATO do contexto passado (vazio em closed-book)
+ context_type      : none | oracle | ranking_1de5 | distractor_only
+ system_prompt_sha : sha256 do system prompt (texto único em
                      data/eval/system_prompts.json)
+ system_prompt_ref : chave no system_prompts.json
+ checkpoint        : ex. baseline-F0 | A-r8-ep5
+ gen_params        : JSON dos parâmetros de geração

Uso:
  # baseline retroativo (reusa gen/scored do lfm3_f0)
  classifier/.venv/bin/python eval/export_parquet.py --baseline
  # células F2 (gen em data/gen/, scored em data/scored/)
  classifier/.venv/bin/python eval/export_parquet.py --tag f2-A-r8-ep5 --cell closed_book
  # tudo que existir em data/scored/
  classifier/.venv/bin/python eval/export_parquet.py --all
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
F2 = HERE.parent
REPO = F2.parents[1]
F0_DATA = REPO / "bench" / "lfm3_f0" / "data"
OUT = F2 / "data" / "eval"
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
from common_f0 import SYSTEM_PROMPT, build_cell_context, load_rows, render_user  # noqa: E402

GEN_PARAMS = json.dumps({"temperature": 0.1, "top_k": 50,
                         "repetition_penalty": 1.1, "max_new_tokens": 3072,
                         "dtype": "bfloat16", "chat_template": "official",
                         "thinking": "on"})

CELL_CONTEXT_TYPE = {"closed_book": "none", "oraculo": "oracle",
                     "ranking": "ranking_1de5", "distrator": "distractor_only"}


def bench_parquet() -> str:
    pats = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "datasets--UFGCEMIGONA--benchmark_mobile/snapshots/*/"
                         "normas_regulamentadoras/train-00000-of-00001.parquet"))
    assert pats
    return pats[0]


def read_jsonl_any(path: Path) -> list[dict]:
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    return []


def ensure_system_prompt_registry() -> tuple[str, str]:
    """Registra o system prompt único; retorna (ref, sha)."""
    reg_path = OUT / "system_prompts.json"
    sha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]
    ref = "f0_f2_fixed_v1"
    reg = {}
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg[ref] = {"sha256_16": sha, "text": SYSTEM_PROMPT}
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    return ref, sha


def build_context_given(row: dict, cell: str) -> str:
    """Reconstrói deterministicamente o contexto exato (mesma seed por id)."""
    ctx = build_cell_context(row, cell)
    if not ctx["passages"]:
        return ""
    return render_user(row, ctx["passages"]).rsplit("\n\nPergunta:", 1)[0]


def export_cell(checkpoint: str, cell: str, gens: list[dict],
                scored: list[dict], bench_rows: dict) -> Path:
    ref, sha = ensure_system_prompt_registry()
    smap = {s["id"]: s for s in scored}
    gmap = {g["id"]: g for g in gens}
    recs = []
    for g in gens:
        s = smap.get(g["id"])
        if not s:
            continue
        b = bench_rows[g["id"]]
        recs.append({
            "id": b["id"], "base_qa_id": b["base_qa_id"],
            "question": b["question"], "answer_gold": b["answer"],
            "question_type": b["question_type"], "source_id": b["source_id"],
            "vocabulary_level": b["vocabulary_level"],
            "query_features": json.dumps(b["query_features"], ensure_ascii=False)
                if not isinstance(b["query_features"], str) else b["query_features"],
            "source_file": b["source_file"],
            "evidence_text": "\n\n".join(e["text"] for e in b["evidence"]),
            "answer_model": s["answer"],
            "raw_output": g.get("raw", ""),
            "thinking": g.get("thinking", ""),
            "n_gen_tokens": s["n_gen_tokens"], "stopped_eos": s["stopped_eos"],
            "coverage": s["coverage"], "n_facts": s["n_facts"],
            "tautology": s["tautology"], "hallucination": s["hallucination"],
            "refused": s["refused"], "truncated": s["truncated"],
            "refusal_correct": s["refusal_correct"],
            "refusal_undue": s["refusal_undue"],
            "citation_present": s["citation_present"],
            "approved": s["approved"],
            "answered_correct_anyway": s.get("answered_correct_anyway"),
            "judge_note": s.get("judge_note", ""),
            "context_given": build_context_given(b, cell),
            "context_type": CELL_CONTEXT_TYPE[cell],
            "system_prompt_sha": sha, "system_prompt_ref": ref,
            "checkpoint": checkpoint, "cell": cell,
            "gen_params": GEN_PARAMS,
        })
    out = OUT / f"lfm3-{checkpoint}-{cell.replace('_','-')}.parquet"
    pq.write_table(pa.Table.from_pylist(recs), out, compression="zstd")
    print(f"[ok] {out.name}: {len(recs)} linhas")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cell", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    bench_rows = {r["id"]: r for r in load_rows(bench_parquet())}
    OUT.mkdir(parents=True, exist_ok=True)

    if args.baseline:
        for cell in ("closed_book", "oraculo", "ranking", "distrator"):
            gens = read_jsonl_any(F0_DATA / f"gen_{cell}.jsonl")
            scored = read_jsonl_any(F0_DATA / f"scored_{cell}.jsonl")
            export_cell("baseline", cell, gens, scored, bench_rows)
        return

    pairs = []
    if args.all:
        for sp in sorted((F2 / "data" / "scored").glob("scored_*.jsonl")):
            m = re.match(r"scored_(.+)_(closed_book|oraculo|ranking|distrator)$", sp.stem)
            if m:
                pairs.append((m.group(1), m.group(2)))
    elif args.tag and args.cell:
        pairs = [(args.tag, args.cell)]
    else:
        ap.error("use --baseline, --all ou --tag+--cell")

    for tag, cell in pairs:
        gens = read_jsonl_any(F2 / "data" / "gen" / f"gen_{tag}_{cell}.jsonl")
        scored = read_jsonl_any(F2 / "data" / "scored" / f"scored_{tag}_{cell}.jsonl")
        ckpt = tag.replace("f2-", "")
        export_cell(ckpt, cell, gens, scored, bench_rows)


if __name__ == "__main__":
    main()
