"""
export_parquet_f3.py — shim do export canônico da F3: reusa
bench/lfm3_f2/eval/export_parquet.py (schema estendido: context_given/
context_type/system_prompt/checkpoint/gen_params/raw_output) com saída em
bench/lfm3_f3/data/eval. Zero lógica nova.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3/eval/export_parquet_f3.py --all
    classifier/.venv/bin/python bench/lfm3_f3/eval/export_parquet_f3.py \
        --tag f3-r32-ep2 --cell ranking
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
REPO = F3.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))

import export_parquet  # noqa: E402

export_parquet.F2 = F3
export_parquet.OUT = F3 / "data" / "eval"

if __name__ == "__main__":
    export_parquet.main()
