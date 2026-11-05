#!/usr/bin/env python3
"""merge_f4.py — funde o LoRA f4-r32-ep2 (checkpoint-284) no merged data2x-ep4 -> HF full bf16.

Roda na Spark (venv ~/cemig-poc/lfm3_f2/.venv). Publica md5 dos shards em <out>/md5.json.
"""
import argparse, hashlib, json
from pathlib import Path


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
m.save_pretrained(str(out))
tok.save_pretrained(str(out))
# arquivos de código remoto + template do checkpoint (o template do TREINO)
for f in ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"):
    src = Path(a.model) / f
    if src.exists():
        (out / f).write_bytes(src.read_bytes())
tmpl = Path(a.adapter) / "chat_template.jinja"
if tmpl.exists():
    (out / "chat_template.jinja").write_bytes(tmpl.read_bytes())
sums = {p.name: md5(p) for p in sorted(out.glob("*.safetensors"))}
rec = {"base": a.model, "adapter": a.adapter, "shard_md5": sums,
       "aggregate_md5": hashlib.md5(json.dumps(sums, sort_keys=True).encode()).hexdigest()}
(out / "md5.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
