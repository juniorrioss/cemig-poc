"""
train_lora_f2.py — braço A (LoRA no INSTRUCT) e metade do braço B (mesmo LoRA
sobre o BASE). Injeção de conhecimento estilo pré-treino: CLM sobre a mistura
raw+paraphrase+QA, MUITAS épocas, LR baixo, checkpoint POR época.

ASSERT obrigatório (ordem do brief): target_modules=["w1","w2","w3"] deve
atingir exatamente 72 matrizes lora_A, todas em *.feed_forward.*, zero em
*.self_attn.* (no LFM3, gate_proj é gate da ATENÇÃO, não MLP).

Logging: Weights & Biases (ordem do capitão). WANDB_MODE=offline por default
(a Spark tem ~/.netrc com api.wandb.ai -> pode usar online; decidimos por env).
Runs: lfm3-f2-A-r{8,32} | lfm3-f2-B-base-r{8,32}. Dir persistente: --wandb-dir.

Roda na Spark (venv lfm3_f2/.venv, transformers>=5.12, peft, trl).

Uso (exemplos):
    python train_lora_f2.py --model LiquidAI/LFM3-1.7B --rank 8  --run lfm3-f2-A-r8
    python train_lora_f2.py --model LiquidAI/LFM3-1.7B --rank 32 --run lfm3-f2-A-r32
    python train_lora_f2.py --model LiquidAI/LFM3-1.7B-Base --rank 8 --run lfm3-f2-B-base-r8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--run", required=True, help="nome do run wandb/output")
    ap.add_argument("--train", default="data/train/train.jsonl")
    ap.add_argument("--val", default="data/train/val.jsonl")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--wandb-dir", default=os.path.expanduser("~/cemig-poc/lfm3_f2/wandb"))
    args = ap.parse_args()

    # wandb: offline por default; exporte WANDB_MODE=online se a CLI estiver logada
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "cemig-lfm3-f2")
    os.environ["WANDB_DIR"] = args.wandb_dir
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16)

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["w1", "w2", "w3"],
    )
    model = get_peft_model(model, lora)

    # ---- ASSERT obrigatório do brief ---------------------------------------
    lora_a = [n for n, _ in model.named_parameters() if "lora_A" in n]
    ff = [n for n in lora_a if ".feed_forward." in n]
    attn = [n for n in lora_a if ".self_attn." in n]
    assert len(ff) == 72, f"esperava 72 lora_A em feed_forward, veio {len(ff)}"
    assert len(attn) == 0, f"lora_A em self_attn (PROIBIDO): {attn[:4]}"
    assert len(lora_a) == 72, f"lora_A total != 72: {len(lora_a)}"
    model.print_trainable_parameters()
    premises = {
        "model": args.model, "rank": args.rank, "run": args.run,
        "n_lora_A": len(lora_a), "n_feed_forward": len(ff),
        "n_self_attn": len(attn),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
    }
    out_dir = Path(args.out) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "premises.json").write_text(json.dumps(premises, indent=2))
    print(json.dumps(premises, indent=2))

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, batched=True,
                remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=args.run,
        report_to=["wandb"],
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,  # TODOS os checkpoints (curva época)
        seed=42,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=collator)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[ok] {out_dir}")


if __name__ == "__main__":
    main()
