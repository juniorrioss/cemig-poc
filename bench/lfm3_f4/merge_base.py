#!/usr/bin/env python3
"""
merge_base.py — funde o LoRA campeão do F3b (data2x-ep4 = checkpoint-1396 de
lfm3-f3b-data-2x) no LFM3-1.7B instruct, produzindo o CHECKPOINT-BASE do F4
(o novo LoRA de tool-calling treina SOBRE este merged, ordem do brief).

Publica o md5 de cada shard + md5 agregado em data/merged_base_md5.json.

Uso (local .venv-f4 ou Spark ~/jupyterlab/.venv):
    python bench/lfm3_f4/merge_base.py \
        --adapter /tmp/f4_base/checkpoint-1396 \
        --out /tmp/f4_base/lfm3-1.7b-data2x-ep4-merged
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    sums = {p.name: md5(p) for p in sorted(out.glob("*.safetensors"))}
    agg = hashlib.md5(json.dumps(sums, sort_keys=True).encode()).hexdigest()
    rec = {"base": args.model, "adapter": args.adapter,
           "out": str(out), "shard_md5": sums, "aggregate_md5": agg}
    here = Path(__file__).resolve().parent
    (here / "data").mkdir(exist_ok=True)
    (here / "data" / "merged_base_md5.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
