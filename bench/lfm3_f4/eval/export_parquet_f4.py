"""
export_parquet_f4.py — parquets canônicos do F4.

(a) células ESTÁTICAS (shim do export_parquet da F2, schema estendido);
(b) suites de TOOL (schema estendido com context_type NOVO:
    tool_loop  = diálogo com simulador de tool real (decision/reuse/switch/refusal)
    multihop   = cadeia de 2 hops (hop1 parcial congelado + hop2 ao vivo)).

Uso:
  # estáticas
  classifier/.venv/bin/python bench/lfm3_f4/eval/export_parquet_f4.py \
      --tag f4-r32-ep2 --cell ranking
  # tool suites
  classifier/.venv/bin/python bench/lfm3_f4/eval/export_parquet_f4.py \
      --tools f4-r32-ep2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
F4 = HERE.parent
REPO = F4.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))
sys.path.insert(0, str(F4))

import export_parquet  # noqa: E402

export_parquet.F2 = F4
export_parquet.OUT = F4 / "data" / "eval"

SUITE_CONTEXT_TYPE = {"decision": "tool_loop", "reuse": "tool_loop",
                      "switch": "tool_loop", "refusal": "tool_loop",
                      "multihop": "multihop"}


def export_tools(tag: str) -> None:
    import common_f4 as C
    out_dir = F4 / "data" / "eval"
    for suite, ctype in SUITE_CONTEXT_TYPE.items():
        suite_rows = {s["id"]: s for s in C.read_jsonl(out_dir / f"suite_{suite}.jsonl")}
        traces = C.read_jsonl(out_dir / f"trace_{tag}_{suite}.jsonl")
        if not traces:
            continue
        recs = []
        for t in traces:
            s = suite_rows.get(t["id"], {})
            recs.append({
                "id": t["id"], "suite": suite,
                "question": s.get("question") or s.get("followup")
                or s.get("new_question") or "",
                "answer_model": t["final"],
                "n_calls": t["n_calls"],
                "calls": json.dumps(t["calls"], ensure_ascii=False),
                "syntactic_fail": t["syntactic_fail"],
                "raw_last": t.get("raw_last", ""),
                "suite_item": json.dumps(s, ensure_ascii=False),
                "context_type": ctype,
                "checkpoint": tag, "cell": f"tool_{suite}",
            })
        out = out_dir / f"lfm3-{tag}-tool-{suite}.parquet"
        pq.write_table(pa.Table.from_pylist(recs), out, compression="zstd")
        print(f"[ok] {out.name}: {len(recs)} linhas")


if __name__ == "__main__":
    if "--tools" in sys.argv:
        export_tools(sys.argv[sys.argv.index("--tools") + 1])
    else:
        export_parquet.main()
