#!/usr/bin/env python3
"""
train.py — Treino de síntese do LFM2.5-2.6B em duas fases (5070, bf16 + LoRA r=16 + grad ckpt).

FASE A (SFT): ensina o FORMATO (2 chunks + pergunta -> resposta 2-4 frases com citação exata).
  LoRA r=16, 1 época, lr 1e-4, completion-only loss (mascaramento do prompt via SFTTrainer).

FASE B (DPO): sobre o adapter da FASE A. rejected = saída real do 2.6B nota-baixa; chosen = edição
  mínima do 27B. DPOTrainer, beta 0.1, lr 5e-6, 1 época.

QLoRA 4-bit é fallback automático se estourar VRAM (OOM) na 5070 (12 GB). Se mesmo o QLoRA não
couber, o script aborta com instrução p/ reportar 'paused: preciso da DGX/H100' (brief).

Checkpoints/exports vão para ~/train-poc/ (FORA do repo). Ao final da fase escolhida exporta
adapter + merged + GGUF Q4 (via llama.cpp) do melhor.

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("finetune2.train")

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
DEFAULT_BASE = "LiquidAI/LFM2.5-2.6B"
TRAIN_ROOT = Path.home() / "train-poc"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def load_sft_dataset(path: Path) -> Dataset:
    """Carrega data/sft.jsonl (ChatML) -> dataset com coluna 'messages' p/ o SFTTrainer."""
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        r = json.loads(s)
        rows.append({"messages": r["messages"]})
    logger.info("SFT: %d exemplos de %s", len(rows), path)
    return Dataset.from_list(rows)


def load_dpo_dataset(path: Path) -> Dataset:
    """Carrega data/dpo.jsonl -> dataset {prompt, chosen, rejected} (formato do DPOTrainer)."""
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        r = json.loads(s)
        rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    logger.info("DPO: %d pares de %s", len(rows), path)
    return Dataset.from_list(rows)


def build_lora_config(r: int, alpha: int):
    from peft import LoraConfig, TaskType
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)


def load_base_model(model_id: str, use_qlora: bool):
    from transformers import AutoModelForCausalLM
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if use_qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        logger.info("Carregando base em QLoRA 4-bit (nf4).")
    else:
        logger.info("Carregando base em bf16.")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def run_sft(args: argparse.Namespace, use_qlora: bool) -> Path:
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_sft_dataset(Path(args.sft_data))
    model = load_base_model(args.base, use_qlora)
    model.config.use_cache = False

    out_dir = TRAIN_ROOT / "sft_adapter"
    ckpt_dir = TRAIN_ROOT / "sft_ckpt"
    cfg = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_bs,
        gradient_accumulation_steps=args.sft_grad_accum,
        learning_rate=args.sft_lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",
        bf16=not use_qlora,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,   # mascara o prompt; gradiente só na resposta
        packing=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=build_lora_config(args.lora_r, args.lora_alpha),
    )
    logger.info("FASE A (SFT): iniciando...")
    res = trainer.train()
    logger.info("FASE A concluída. loss=%.4f", res.training_loss)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    logger.info("Adapter SFT salvo em %s", out_dir)
    return out_dir


def run_dpo(args: argparse.Namespace, sft_adapter: Path, use_qlora: bool) -> Path:
    from peft import PeftModel
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dpo_dataset(Path(args.dpo_data))

    # Modelo base + adapter SFT como ponto de partida (policy); DPO treina novos deltas LoRA
    # em cima. ref_model=None -> o DPOTrainer usa o modelo com adapter desabilitado como referência.
    base = load_base_model(args.base, use_qlora)
    model = PeftModel.from_pretrained(base, str(sft_adapter), is_trainable=True)
    model.config.use_cache = False

    out_dir = TRAIN_ROOT / "dpo_adapter"
    ckpt_dir = TRAIN_ROOT / "dpo_ckpt"
    cfg = DPOConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.dpo_epochs,
        per_device_train_batch_size=args.dpo_bs,
        gradient_accumulation_steps=args.dpo_grad_accum,
        learning_rate=args.dpo_lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",
        bf16=not use_qlora,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.dpo_beta,
        max_length=args.max_length,
        report_to="none",
        seed=args.seed,
    )
    # ref_model=None + peft: usa o adapter desabilitado como referência (economia de VRAM).
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         processing_class=tok)
    logger.info("FASE B (DPO): iniciando sobre o adapter SFT...")
    res = trainer.train()
    logger.info("FASE B concluída. loss=%.4f", res.training_loss)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    logger.info("Adapter DPO salvo em %s", out_dir)
    return out_dir


def export_merged_and_gguf(base_id: str, adapter_dir: Path, tag: str) -> Optional[Path]:
    """Funde o adapter no base, salva merged e converte para GGUF Q4 via llama.cpp."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Exportando merged de %s...", adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = TRAIN_ROOT / f"merged_{tag}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tok.save_pretrained(str(merged_dir))
    logger.info("Merged salvo em %s", merged_dir)

    convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert.exists():
        logger.warning("convert_hf_to_gguf.py ausente; pulando GGUF.")
        return merged_dir
    f16 = merged_dir / f"lfm2.5-2.6b-{tag}-f16.gguf"
    q4 = merged_dir / f"lfm2.5-2.6b-{tag}-Q4_0.gguf"
    try:
        subprocess.run([sys.executable, str(convert), str(merged_dir),
                        "--outfile", str(f16), "--outtype", "f16"], check=True,
                       capture_output=True, text=True)
        quant = Path.home() / "llama.cpp" / "build-quant" / "bin" / "llama-quantize"
        if not quant.exists():
            quant = Path.home() / "llama.cpp" / "build-cuda" / "bin" / "llama-quantize"
        if quant.exists():
            # NB: stderr do binário não é UTF-8 (barras de progresso) -> capturar como bytes.
            subprocess.run([str(quant), str(f16), str(q4), "Q4_0"], check=True,
                           capture_output=True)
            logger.info("GGUF Q4_0 exportado: %s (%.0f MB)", q4, q4.stat().st_size / 1e6)
        else:
            logger.warning("llama-quantize ausente; ficou só o f16 GGUF.")
    except subprocess.CalledProcessError as e:
        logger.warning("Falha na conversão GGUF: %s", e.stderr[-500:] if e.stderr else e)
    return merged_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--phase", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--sft-data", default=str(_HERE / "data" / "sft.jsonl"))
    ap.add_argument("--dpo-data", default=str(_HERE / "data" / "dpo.jsonl"))
    ap.add_argument("--sft-adapter", default=str(TRAIN_ROOT / "sft_adapter"),
                    help="adapter SFT p/ iniciar a FASE B quando --phase dpo")
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1536)
    # FASE A
    ap.add_argument("--sft-epochs", type=float, default=1.0)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--sft-bs", type=int, default=2)
    ap.add_argument("--sft-grad-accum", type=int, default=8)
    # FASE B
    ap.add_argument("--dpo-epochs", type=float, default=1.0)
    ap.add_argument("--dpo-lr", type=float, default=5e-6)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-bs", type=int, default=1)
    ap.add_argument("--dpo-grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qlora", action="store_true", help="força QLoRA 4-bit desde o início")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        logger.error("CUDA indisponível. paused: preciso da DGX/H100.")
        sys.exit(2)

    def _with_oom_fallback(fn):
        """Roda fn(use_qlora); em OOM tenta QLoRA; se ainda estourar, sinaliza DGX/H100."""
        try:
            return fn(args.qlora)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM em bf16. Repetindo em QLoRA 4-bit...")
            try:
                return fn(True)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM mesmo em QLoRA. REPORTE: paused: preciso da DGX/H100.")
                sys.exit(3)

    sft_adapter = Path(args.sft_adapter)
    if args.phase in ("sft", "both"):
        sft_adapter = _with_oom_fallback(lambda q: run_sft(args, q))
        if not args.no_export and args.phase == "sft":
            export_merged_and_gguf(args.base, sft_adapter, "sft")

    if args.phase in ("dpo", "both"):
        dpo_adapter = _with_oom_fallback(lambda q: run_dpo(args, sft_adapter, q))
        if not args.no_export:
            export_merged_and_gguf(args.base, dpo_adapter, "dpo")

    logger.info("Treino concluído (fase=%s).", args.phase)


if __name__ == "__main__":
    main()
