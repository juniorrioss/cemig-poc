#!/usr/bin/env python3
"""
merge_policy.py — funde um LoRA sobre um modelo HF (bf16) e publica md5.
Usado 2x no P1/P2:
  1) f4-r32-ep2 = merged data2x-ep4 (`/workspace/cemig/f4/merged-rebuilt`) +
     LoRA `lfm3-f4-tools-r32/checkpoint-284` -> POLÍTICA DE PARTIDA do P1 e P2
     (o RLVR treina um LoRA NOVO r32 MLP sobre este merged, igual ao F4 fez
     sobre o data2x).
  2) p2 final (opcional, p/ GGUF/embarque).

Uso (container): python merge_policy.py --base B --adapter A --out O
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    # código custom do LFM3 (trust_remote_code) viaja junto
    for f in Path(args.base).glob("*.py"):
        shutil.copy(f, out / f.name)
    sums = {p.name: md5(p) for p in sorted(out.glob("*.safetensors"))}
    rec = {"base": args.base, "adapter": args.adapter, "out": str(out),
           "shard_md5": sums,
           "aggregate_md5": hashlib.md5(json.dumps(sums, sort_keys=True).encode()).hexdigest()}
    (out / "merge_md5.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
