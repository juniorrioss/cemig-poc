"""
train_f3b.py — treino do sweep F3b (one-factor-at-a-time sobre a receita F3)
na H100 (container cemig_mobile_walcy, GPU 2 = cuda:0 interno).

Generalização mínima do train_f3.py da F3: parametriza RANK, ALVOS LoRA
(mlp = w1/w2/w3; mlp_attn = + q/k/v/o_proj da atenção — NUNCA
self_attn.gate_proj, que é o gate sigmoide da atenção do LFM3), LR e os
arquivos de dados. Todo o resto é IDÊNTICO à receita F3 (cosine, bs8xga4,
max_len 6144, assistant_only_loss, seed 42, ckpt por época).

ASSERT de módulos obrigatório (brief):
  mlp      -> 72 lora_A, todos em *.feed_forward.*, 0 em *.self_attn.*
  mlp_attn -> 116 lora_A = 72 feed_forward + 44 self_attn.{q,k,v,o}_proj
              (11 camadas de atenção x 4), 0 em gate_proj/conv.

Uso (dentro do container):
    python train_f3b.py --rank 32 --targets mlp --epochs 4 --lr 1e-4 \
        --train /workspace/cemig/f3b/train_recusa35.jsonl \
        --val /workspace/cemig/f3b/val_recusa35.jsonl \
        --run lfm3-f3b-recusa-35 \
        --out /workspace/cemig/checkpoints/lfm3-f3b-recusa-35
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

MLP_TARGETS = ["w1", "w2", "w3"]
# q/k/v/o da atenção por SUFIXO completo para nunca capturar conv.in_proj/
# conv.out_proj nem self_attn.gate_proj.
ATTN_REGEX = (r".*\.(feed_forward\.(w1|w2|w3)"
              r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--targets", choices=["mlp", "mlp_attn"], default="mlp")
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_PROJECT", "lfm3-1.7b-cemig")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16)
    model.config.use_cache = False

    if args.targets == "mlp":
        target_modules = MLP_TARGETS
    else:
        target_modules = ATTN_REGEX

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    keep = {"messages"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(ds["train"])
    cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name=args.run,
        report_to=["wandb"],
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x nao tem warmup_ratio -> ~3% dos steps reais
        warmup_steps=max(10, int(0.03 * args.epochs * n_train
                                 / (args.batch * args.grad_accum))),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        assistant_only_loss=True,
        packing=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=lora)

    # ---- ASSERT obrigatório de módulos (aborta, não avisa) -----------------
    lora_a = [n for n, _ in trainer.model.named_parameters() if "lora_A" in n]
    ff = [n for n in lora_a if ".feed_forward." in n]
    attn = [n for n in lora_a if ".self_attn." in n]
    gate = [n for n in lora_a if "gate_proj" in n]
    conv = [n for n in lora_a if ".conv." in n]
    assert len(gate) == 0, f"lora_A em gate_proj (PROIBIDO): {gate[:4]}"
    assert len(conv) == 0, f"lora_A em conv (PROIBIDO): {conv[:4]}"
    assert len(ff) == 72, f"esperava 72 lora_A em feed_forward, veio {len(ff)}"
    if args.targets == "mlp":
        assert len(attn) == 0, f"lora_A em self_attn (PROIBIDO no mlp): {attn[:4]}"
        assert len(lora_a) == 72, f"lora_A total != 72: {len(lora_a)}"
    else:
        ok = [n for n in attn if re.search(
            r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.", n)]
        assert len(attn) == 44 and len(ok) == 44, (
            f"esperava 44 lora_A em self_attn.q/k/v/o, veio {len(attn)} "
            f"(validos {len(ok)}): {attn[:6]}")
        assert len(lora_a) == 116, f"lora_A total != 116: {len(lora_a)}"
    trainer.model.print_trainable_parameters()
    premises = {
        "model": args.model, "rank": args.rank, "targets": args.targets,
        "run": args.run, "n_lora_A": len(lora_a), "n_feed_forward": len(ff),
        "n_self_attn": len(attn),
        "trainable": sum(p.numel() for p in trainer.model.parameters()
                         if p.requires_grad),
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
        "assistant_only_loss": True,
        "n_train": n_train, "n_val": len(ds["val"]),
        "train_file": args.train, "val_file": args.val,
    }
    (out_dir / "premises.json").write_text(json.dumps(premises, indent=2))
    print(json.dumps(premises, indent=2), flush=True)

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(out_dir / "final"))
    print(f"[ok] {out_dir}")


if __name__ == "__main__":
    main()
