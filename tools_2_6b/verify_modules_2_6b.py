#!/usr/bin/env python3
"""
verify_modules_2_6b.py — PASSO 0 (parte de MÓDULOS): roda na DGX Spark (o base_hf mora lá).

Confirma os nomes de módulo do 2.6B no model.safetensors.index.json (self_attn.{q,k,v,out}
_proj 8x, feed_forward.{w1,w2,w3} 30x, conv.* 22x — já conhecidos, mas o brief exige CONFIRMAR)
e, montando o MESMO LoRA por REGEX do sft_v2/train_v2, publica targeted_module_names +
print_trainable_parameters ANTES de treinar, com os 3 ASSERTS (mlp presente, attn presente,
conv AUSENTE).

Uso (na Spark): ~/jupyterlab/.venv/bin/python ~/cemig-poc/tools_2_6b/verify_modules_2_6b.py \
    --base ~/cemig-poc/base_hf --lora-r 64 --out ~/cemig-poc/tools_2_6b/data/passo0_modules.json
Comentários PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# MESMO regex do sft_v2/train_v2 (SÓ atenção + MLP; NÃO casa conv.*).
LORA_TARGETS = r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))$"


def confirm_index(base: Path) -> dict:
    """Confirma os nomes de módulo pelo index do safetensors (self_attn/feed_forward/conv)."""
    idx = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    fams = Counter()
    for k in idx["weight_map"]:
        if ".self_attn." in k:
            fams["self_attn." + k.split(".self_attn.")[1].split(".weight")[0]] += 1
        elif ".feed_forward." in k:
            fams["feed_forward." + k.split(".feed_forward.")[1].split(".weight")[0]] += 1
        elif ".conv." in k:
            fams["conv." + k.split(".conv.")[1].split(".weight")[0]] += 1
    return dict(fams)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "cemig-poc" / "base_hf"))
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"
                                         / "passo0_modules.json"))
    args = ap.parse_args()

    base = Path(args.base)
    fams = confirm_index(base)
    print("=== nomes de módulo confirmados no index (2.6B) ===")
    for k, v in sorted(fams.items()):
        print(f"  {k}: {v}")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules=LORA_TARGETS,
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    pm = get_peft_model(model, lora)
    pm.print_trainable_parameters()
    targeted = sorted(set(getattr(pm, "targeted_module_names", [])))
    n_attn = sum(1 for m in targeted if ".self_attn." in m)
    n_mlp = sum(1 for m in targeted if ".feed_forward." in m)
    n_conv = sum(1 for m in targeted if ".conv." in m)
    has_mlp = n_mlp > 0
    has_attn = n_attn > 0
    has_conv = n_conv > 0
    print(f"=== targeted_module_names: {len(targeted)} (attn={n_attn} mlp={n_mlp} conv={n_conv}) ===")
    print("amostra:", targeted[:8])
    assert has_mlp, "LoRA NÃO cobre o MLP!"
    assert has_attn, "LoRA NÃO cobre a atenção!"
    assert not has_conv, "LoRA está tocando o ShortConv (conv.*)!"
    print("VERIFICAÇÃO OK: LoRA cobre attn + MLP e NÃO toca o ShortConv.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "base": args.base, "lora_r": args.lora_r, "index_module_families": fams,
        "n_targeted": len(targeted), "n_attn": n_attn, "n_mlp": n_mlp, "n_conv": n_conv,
        "has_mlp": has_mlp, "has_attn": has_attn, "has_conv": has_conv,
        "targeted_sample": targeted[:24],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] -> {args.out}")


if __name__ == "__main__":
    main()
