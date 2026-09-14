#!/usr/bin/env python3
"""harness_oraculo_2_6b.py — oráculo/híbrido/puro/recusa do 2.6B, reusando tools_oraculo/harness_oraculo.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools_v1"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools_oraculo"))
import patch_2_6b  # noqa: F401,E402
assert patch_2_6b.PATCHED
import harness_oraculo  # noqa: E402
if __name__ == "__main__":
    harness_oraculo.main()
