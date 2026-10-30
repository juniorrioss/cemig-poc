"""
train_f3.py — treino da F3 (robustez de uso de contexto) na H100 (container
cemig_mobile_walcy, GPU 2 do nó = cuda:0 interno).

Base = LiquidAI/LFM3-1.7B INSTRUCT limpo (decisão conservadora do capitão —
NÃO partir do A-r32-ep2 da F2). LoRA target_modules=["w1","w2","w3"] com o
ASSERT obrigatório da F2 (72 lora_A em *.feed_forward.*, 0 em *.self_attn.* —
no LFM3, gate_proj é gate da ATENÇÃO). r32, SFT chat com
assistant_only_loss=True (TRL; o chat_template do LFM3 tem {% generation %}),
2-4 épocas com checkpoint POR época (a F2 provou pico cedo).

wandb: projeto lfm3-1.7b-cemig, run lfm3-f3-robustez-r32 (online no container;
.netrc já copiado na F2).

Dados: train.jsonl/val.jsonl do build_dataset.py --render (messages chat).
O template do LFM3 injeta <think></think> vazio antes do conteúdo do
assistant (sem campo thinking) — o modelo aprende a responder direto, e a
avaliação (thinking-ON) separa o rascunho da resposta como no F0.

Uso (dentro do container):
    python train_f3.py --rank 32 --epochs 4 \
        --train /workspace/cemig/f3/train.jsonl --val /workspace/cemig/f3/val.jsonl \
        --out /workspace/cemig/checkpoints/lfm3-f3-robustez-r32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--run", default="lfm3-f3-robustez-r32")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--out", required=True)
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

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["w1", "w2", "w3"],
    )

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    keep = {"messages"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SFTConfig(
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        assistant_only_loss=True,
        packing=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,   # TODOS os checkpoints (curva por época)
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=lora)

    # ---- ASSERT obrigatório (padrão F2; abortam, não avisam) ---------------
    lora_a = [n for n, _ in trainer.model.named_parameters() if "lora_A" in n]
    ff = [n for n in lora_a if ".feed_forward." in n]
    attn = [n for n in lora_a if ".self_attn." in n]
    assert len(ff) == 72, f"esperava 72 lora_A em feed_forward, veio {len(ff)}"
    assert len(attn) == 0, f"lora_A em self_attn (PROIBIDO): {attn[:4]}"
    assert len(lora_a) == 72, f"lora_A total != 72: {len(lora_a)}"
    trainer.model.print_trainable_parameters()
    premises = {
        "model": args.model, "rank": args.rank, "run": args.run,
        "n_lora_A": len(lora_a), "n_feed_forward": len(ff), "n_self_attn": len(attn),
        "trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
        "assistant_only_loss": True,
        "n_train": len(ds["train"]), "n_val": len(ds["val"]),
    }
    (out_dir / "premises.json").write_text(json.dumps(premises, indent=2))
    print(json.dumps(premises, indent=2), flush=True)

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[ok] {out_dir}")


if __name__ == "__main__":
    main()
