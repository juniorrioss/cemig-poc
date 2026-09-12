#!/usr/bin/env python3
"""
train_sft.py — SFT de DESTILAÇÃO do LFM2.5-2.6B (roda na DGX Spark, GB10 121 GB).

Alvo: ensinar o ESTILO ACIONÁVEL do 27B (respostas APROVADAS pela régua honesta), não
imitar o comportamento atual do 2.6B. Dataset: data/train_sft.jsonl (pares régua-aprovados
de gen_train.py). LoRA (deltas pequenos, base forte preservado — a lição do finetune2 é que
full/over-fit regride; LoRA conservador é o caminho).

Decisão de LoRA vs full (justificativa):
  - A GB10 tem 121 GB unificada e caberia full-FT, MAS o finetune2 mostrou que treino
    AGRESSIVO sobre distribuição estreita REGRIDE um base forte (encolhe resposta, cita item
    errado). LoRA com r moderado limita o dano e preserva a capacidade geral do base — é a
    escolha conservadora pedida pelo brief. Salvamos 1 checkpoint POR ÉPOCA para avaliar a
    curva e ABORTAR se regredir.

Targets LoRA: atenção + MLP (q/k/v/out_proj, gate/up/down_proj). O shortconv (in_proj/conv)
fica de fora (recorrente; instável a LoRA, como visto no engine-upgrade).

Checkpoints por época em ~/cemig-poc/train/sft_ep{N}. Merge+GGUF Q4_0 por época (p/ avaliar
no MESMO engine do teto). Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("slm_oraculo.train_sft")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_sft(path: Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append({"messages": json.loads(s)["messages"]})
    logger.info("SFT: %d exemplos de %s", len(rows), path)
    return Dataset.from_list(rows)


def convert_and_quant(merged_dir: Path, tag: str, llama_dir: Path, out_dir: Path) -> None:
    """Converte merged HF -> GGUF bf16 -> quantiza Q4_0 (engine do teto)."""
    convert = llama_dir / "convert_hf_to_gguf.py"
    quant = llama_dir / "build-cuda" / "bin" / "llama-quantize"
    bf16 = out_dir / f"lfm2.5-2.6b-{tag}-bf16.gguf"
    q4 = out_dir / f"lfm2.5-2.6b-{tag}-Q4_0.gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(convert), str(merged_dir),
                    "--outfile", str(bf16), "--outtype", "bf16"], check=True)
    subprocess.run([str(quant), str(bf16), str(q4), "Q4_0"], check=True)
    logger.info("GGUF exportado: %s / %s", bf16.name, q4.name)


def export_epoch(base_id: str, adapter_dir: Path, tag: str, models_dir: Path,
                 llama_dir: Path) -> None:
    """Funde adapter no base e exporta GGUF bf16+Q4_0 do checkpoint da época."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Exportando merged de %s (%s)...", adapter_dir, tag)
    base = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True,
                                                dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = adapter_dir.parent / f"merged_{tag}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tok.save_pretrained(str(merged_dir))
    del base, merged
    torch.cuda.empty_cache()
    convert_and_quant(merged_dir, tag, llama_dir, models_dir)


class EpochExportCallback:
    """Callback simples: após cada época, salva adapter e exporta GGUF."""
    # NB: usamos on_epoch_end via TrainerCallback.


def main() -> None:
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "cemig-poc" / "base_hf"))
    ap.add_argument("--data", default=str(Path.home() / "cemig-poc" / "train" / "train_sft.jsonl"))
    ap.add_argument("--out-root", default=str(Path.home() / "cemig-poc" / "train"))
    ap.add_argument("--models-dir", default=str(Path.home() / "cemig-poc" / "models"))
    ap.add_argument("--llama-dir", default=str(Path.home() / "cemig-poc" / "llama.cpp"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA indisponível."); sys.exit(2)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    llama_dir = Path(args.llama_dir)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_sft(Path(args.data))

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=LORA_TARGETS,
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)

    cfg = SFTConfig(
        output_dir=str(out_root / "sft_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )

    base_id = args.base

    class ExportCb(TrainerCallback):
        def on_epoch_end(self, cfg_, state, control, **kw):
            ep = int(round(state.epoch))
            adapter_dir = out_root / f"sft_ep{ep}"
            kw["model"].save_pretrained(str(adapter_dir))
            tok.save_pretrained(str(adapter_dir))
            logger.info("Época %d: adapter salvo em %s", ep, adapter_dir)
            # Export GGUF é caro; fazemos por subprocess separado p/ não interromper o treino.
            # Aqui só marcamos; o export roda no final por --export-epochs.
            return control

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora, callbacks=[ExportCb()])
    logger.info("SFT: iniciando %d épocas, %d exemplos, LoRA r=%d...",
                args.epochs, len(ds), args.lora_r)
    res = trainer.train()
    logger.info("SFT concluído. loss=%.4f", res.training_loss)

    # Exporta GGUF de cada checkpoint de época salvo.
    for ep in range(1, args.epochs + 1):
        adapter_dir = out_root / f"sft_ep{ep}"
        if adapter_dir.exists():
            try:
                export_epoch(base_id, adapter_dir, f"sft_ep{ep}", models_dir, llama_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("Falha export época %d: %s", ep, e)


if __name__ == "__main__":
    main()
