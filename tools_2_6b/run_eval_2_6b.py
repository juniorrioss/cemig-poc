#!/usr/bin/env python3
"""run_eval_2_6b.py — decisão/argumentos/reuso do 2.6B, reusando tools_v1/run_eval.py.

Aplica o patch do renderer 2.6B (thinking-OFF) e delega ao main() do tools_v1/run_eval — assim
a bateria é byte-a-byte a mesma do 1.2B (comparabilidade). Mesmos flags do run_eval original.
Uso: ../classifier/.venv/bin/python run_eval_2_6b.py --url ... --label ... --topk 2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools_v1"))
import patch_2_6b  # noqa: F401,E402  (patch ANTES de importar run_eval)
assert patch_2_6b.PATCHED
import run_eval  # noqa: E402
if __name__ == "__main__":
    run_eval.main()
