#!/usr/bin/env python3
"""
retrieve_f3b.py — retriever real v4 sobre o pool do funil f3b (eixo 2,
data2x). Reusa bench/lfm3_f3/retrieve_f3.py com o diretório de synth e a
saída apontados para o f3b. Zero lógica nova.

Uso (llama-server :8399 no ar):
    classifier/.venv/bin/python bench/lfm3_f3b/retrieve_f3b.py [--limit N]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3b
REPO = HERE.parents[1]
F3 = REPO / "bench" / "lfm3_f3"

sys.path.insert(0, str(REPO / "bench" / "ctx_topk"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))

spec = importlib.util.spec_from_file_location("f3_retrieve", F3 / "retrieve_f3.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

rt.F3_SYNTH = HERE / "data" / "synth"
rt.OUT_DIR = HERE / "data" / "retrieval"

if __name__ == "__main__":
    # --pool f3 com F3_SYNTH patchado = pool do funil f3b
    sys.argv = [sys.argv[0], "--pool", "f3",
                "--out", str(HERE / "data" / "retrieval" / "topk_f3b.jsonl")] + sys.argv[1:]
    rt.main()
