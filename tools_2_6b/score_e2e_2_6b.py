#!/usr/bin/env python3
"""score_e2e_2_6b.py — e2e (aprovação/alucinação/recusa) do 2.6B, reusando tools_v1/score_e2e.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools_v1"))
import patch_2_6b  # noqa: F401,E402
assert patch_2_6b.PATCHED
import score_e2e  # noqa: E402
if __name__ == "__main__":
    score_e2e.main()
