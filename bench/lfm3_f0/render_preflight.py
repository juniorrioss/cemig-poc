"""Renderiza 1 prompt byte a byte por célula (preflight item a)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "cemig-poc/lfm3_f0"))
from common_f0 import CELLS, build_messages, load_rows
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
rows = load_rows(Path.home() / "cemig-poc/lfm3_f0/normas_train.parquet")
row = rows[0]
out = {}
for cell in CELLS:
    built = build_messages(row, cell)
    text = tok.apply_chat_template(built["messages"], add_generation_prompt=True, tokenize=False)
    out[cell] = {"id": row["id"], "shuffle_order": built["shuffle_order"],
                 "correct_position": built["correct_position"],
                 "n_chars": len(text), "prompt": text}
import transformers, torch
out["_versions"] = {"python": sys.version.split()[0],
                    "transformers": transformers.__version__,
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available()}
Path(Path.home() / "cemig-poc/lfm3_f0/preflight_prompts.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1))
print("RENDER_OK")
