"""
export_parquet_f3b.py — shim do export canônico para o campeão do sweep F3b:
reusa bench/lfm3_f2/eval/export_parquet.py (schema estendido de transparência)
com dados/saída em bench/lfm3_f3b/data. Zero lógica nova.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3b/eval/export_parquet_f3b.py \
        --tag f3b-data2x-ep4 --cell ranking
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3B = HERE.parent
REPO = F3B.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "eval"))

import export_parquet  # noqa: E402

export_parquet.F2 = F3B
export_parquet.OUT = F3B / "data" / "eval"

if __name__ == "__main__":
    export_parquet.main()
