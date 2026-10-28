"""
retrofit_f0_parquet.py — fase 0 do F2: converte os scored_*.jsonl do baseline F0
para o formato tabular pedido pelo capitão (um parquet por célula de avaliação).

Schema: colunas do dataset benchmark_mobile (menos os chunks pesados) + colunas
de resposta/veredito do modelo. Padrão de nome: lfm3-<variante>-<celula>.parquet.

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/retrofit_f0_parquet.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
F0_DATA = HERE.parent / "lfm3_f0" / "data"
OUT = HERE / "data" / "eval"

# Colunas do benchmark carregadas (chunks_256/512 ficam de fora: pesados e
# reconstrutíveis do dataset; evidence vira texto concatenado).
BENCH_COLS = ("id", "base_qa_id", "question", "answer", "question_type",
              "source_id", "vocabulary_level", "query_features", "source_file")

# Colunas de veredito emitidas pelo score (régua honesta).
VERDICT_COLS = ("answer_model", "n_gen_tokens", "stopped_eos", "coverage",
                "n_facts", "tautology", "hallucination", "refused", "truncated",
                "refusal_correct", "refusal_undue", "citation_present",
                "approved", "answered_correct_anyway", "judge_note")


def find_benchmark_parquet() -> str:
    pats = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "datasets--UFGCEMIGONA--benchmark_mobile/snapshots/*/"
                         "normas_regulamentadoras/train-00000-of-00001.parquet"))
    assert pats, "benchmark_mobile parquet não encontrado no cache HF"
    return pats[0]


def build_cell_table(scored_path: Path, bench_rows: dict) -> pa.Table:
    recs = []
    for line in scored_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        b = bench_rows[s["id"]]
        rec = {c: b[c] for c in BENCH_COLS}
        rec["query_features"] = json.dumps(rec["query_features"], ensure_ascii=False) \
            if not isinstance(rec["query_features"], str) else rec["query_features"]
        rec["answer_gold"] = rec.pop("answer")  # answer do dataset = gabarito
        rec["evidence_text"] = "\n\n".join(e["text"] for e in b["evidence"])
        rec["answer_model"] = s["answer"]
        for c in VERDICT_COLS[1:]:
            rec[c] = s.get(c)
        rec["cell"] = s["cell"]
        recs.append(rec)
    return pa.Table.from_pylist(recs)


def main() -> None:
    bench = pq.read_table(find_benchmark_parquet()).to_pylist()
    bench_rows = {r["id"]: r for r in bench}
    OUT.mkdir(parents=True, exist_ok=True)
    for scored in sorted(F0_DATA.glob("scored_*.jsonl")):
        cell = scored.stem.replace("scored_", "")
        out = OUT / f"lfm3-baseline-{cell.replace('_', '-')}.parquet"
        table = build_cell_table(scored, bench_rows)
        pq.write_table(table, out, compression="zstd")
        print(f"[ok] {out.name}: {table.num_rows} linhas, "
              f"{len(table.schema.names)} colunas")


if __name__ == "__main__":
    main()
