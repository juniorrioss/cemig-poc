#!/usr/bin/env python3
"""
train_v2.py — SFT v2 do LFM2.5-2.6B (roda na DGX Spark). Varredura de RANK, 1 época.

Diferenças do v1 (slm_oraculo/train_sft.py), todas por diagnóstico do capitão:
  1. 1 ÉPOCA (o v1 mostrou overfitting: loss caiu, alucinação subiu ep1->ep3). num_train_epochs=1.
  2. VALIDATION SET INTERNO: separa uma fatia do dataset (eval_split) FORA do treino p/ o
     trainer computar eval_loss ao fim da época (curva no wandb junto da loss). A validação
     de QUALIDADE (régua) é externa (eval.py sobre val_pack) — a lição do v1 é que loss não
     revela regressão; por isso registramos AMBAS.
  3. RANK configurável (--lora-r) com alpha=2r (o launcher roda 16/32/64 com MESMOS dados e
     MESMA seed; única variável = rank).
  4. dados v2 (4 famílias: oráculo + recusa + parcial + distrator) — ensina a RECUSAR.

VERIFICAÇÃO OBRIGATÓRIA (ordem do capitão, após o bug dos targets do v1): antes de treinar,
imprime targeted_module_names + print_trainable_parameters e ABORTA por assert se o LoRA não
cobrir MLP+atenção ou tocar o ShortConv (conv.*). Um assert vale mais que um log.

Arquitetura do LFM2.5-2.6B: self_attn.(q|k|v|out)_proj + feed_forward.(w1|w2|w3);
ShortConv conv.(in_proj|conv|out_proj) EXCLUÍDO (recorrente, instável a LoRA).

Comentários PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sft_v2.train")

# REGEX (PEFT aceita str): SÓ atenção + MLP (feed_forward.w1/w2/w3). NÃO casa conv.*.
LORA_TARGETS = r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))$"


def load_sft(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        d = json.loads(s)
        rows.append({"messages": d["messages"], "family": d["meta"].get("family", "?")})
    return rows


def convert_and_quant(merged_dir: Path, tag: str, llama_dir: Path, out_dir: Path) -> None:
    convert = llama_dir / "convert_hf_to_gguf.py"
    quant = llama_dir / "build-cuda" / "bin" / "llama-quantize"
    bf16 = out_dir / f"lfm2.5-2.6b-{tag}-bf16.gguf"
    q4 = out_dir / f"lfm2.5-2.6b-{tag}-Q4_0.gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(convert), str(merged_dir),
                    "--outfile", str(bf16), "--outtype", "bf16"], check=True)
    subprocess.run([str(quant), str(bf16), str(q4), "Q4_0"], check=True)
    logger.info("GGUF exportado: %s / %s", bf16.name, q4.name)


def export_final(base_id: str, adapter_dir: Path, tag: str, models_dir: Path,
                 llama_dir: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Exportando merged de %s (%s)...", adapter_dir, tag)
    base = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True,
                                                dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = adapter_dir.parent / f"merged_{tag}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(base_id, trust_remote_code=True).save_pretrained(str(merged_dir))
    del base, merged
    torch.cuda.empty_cache()
    convert_and_quant(merged_dir, tag, llama_dir, models_dir)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "cemig-poc" / "base_hf"))
    ap.add_argument("--data", default=str(Path.home() / "cemig-poc" / "train_v2" / "train_v2.jsonl"))
    ap.add_argument("--out-root", default=str(Path.home() / "cemig-poc" / "train_v2"))
    ap.add_argument("--models-dir", default=str(Path.home() / "cemig-poc" / "models"))
    ap.add_argument("--llama-dir", default=str(Path.home() / "cemig-poc" / "llama.cpp"))
    ap.add_argument("--epochs", type=int, default=1)  # 1 ÉPOCA (ordem do capitão)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 = 2*r (default)")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--eval-frac", type=float, default=0.05, help="fatia interna p/ eval_loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="rótulo do artefato (default sft_v2_r{r})")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA indisponível."); sys.exit(2)

    lora_alpha = args.lora_alpha or (2 * args.lora_r)
    tag = args.tag or f"sft_v2_r{args.lora_r}"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    llama_dir = Path(args.llama_dir)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_sft(Path(args.data))
    logger.info("SFT v2: %d exemplos de %s", len(rows), args.data)
    # split interno reprodutível p/ eval_loss (a MESMA seed em todos os ranks -> comparável).
    ds_all = Dataset.from_list(rows).shuffle(seed=args.seed)
    n_eval = max(32, int(len(ds_all) * args.eval_frac))
    ds_eval = ds_all.select(range(n_eval)).remove_columns(["family"])
    ds_train = ds_all.select(range(n_eval, len(ds_all))).remove_columns(["family"])
    logger.info("split interno: treino=%d eval_loss=%d (seed=%d)", len(ds_train), len(ds_eval), args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(r=args.lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGETS,
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)

    cfg = SFTConfig(
        output_dir=str(out_root / f"ckpt_r{args.lora_r}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",       # eval_loss ao FIM da época (curva no wandb)
        save_strategy="epoch",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        assistant_only_loss=True,
        completion_only_loss=False,
        packing=False,
        group_by_length=True,
        dataloader_num_workers=2,
        report_to="wandb",
        run_name=f"sft_v2_r{args.lora_r}",
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        seed=args.seed,
    )

    from transformers import TrainerCallback

    class MemLogCb(TrainerCallback):
        """Loga memória e tok/s por passo (telemetria — lição do incidente OOM do v1)."""
        def on_log(self, cfg_, state, control, logs=None, **kw):
            if logs is not None and torch.cuda.is_available():
                logs["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                logs["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            return control

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds_train, eval_dataset=ds_eval,
                         processing_class=tok, peft_config=lora, callbacks=[MemLogCb()])

    # VERIFICAÇÃO OBRIGATÓRIA (asserts que abortam) — publicada no log antes do treino.
    trainer.model.print_trainable_parameters()
    targeted = sorted(set(getattr(trainer.model, "targeted_module_names", [])))
    logger.info("=== PREMISSAS VERIFICADAS (rank=%d, alpha=%d) ===", args.lora_r, lora_alpha)
    logger.info("targeted_module_names (%d): %s", len(targeted), targeted[:16])
    has_mlp = any("feed_forward.w" in m for m in targeted)
    has_attn = any(("self_attn.q_proj" in m or "self_attn.out_proj" in m) for m in targeted)
    has_conv = any(".conv." in m for m in targeted)
    assert has_mlp, f"LoRA NÃO cobre o MLP! targeted={targeted[:20]}"
    assert has_attn, f"LoRA NÃO cobre a atenção! targeted={targeted[:20]}"
    assert not has_conv, f"LoRA está adaptando o ShortConv (conv.*)! targeted={targeted[:20]}"
    logger.info("VERIFICAÇÃO OK: LoRA cobre atenção + MLP, e NÃO toca o ShortConv.")
    # dump das premissas p/ o relatório (procedência).
    (out_root / f"premises_r{args.lora_r}.json").write_text(json.dumps({
        "rank": args.lora_r, "alpha": lora_alpha, "n_targeted": len(targeted),
        "targeted_sample": targeted[:24], "has_mlp": has_mlp, "has_attn": has_attn,
        "has_conv": has_conv, "n_train": len(ds_train), "n_eval": len(ds_eval),
        "epochs": args.epochs, "lr": args.lr, "bs": args.bs, "grad_accum": args.grad_accum,
        "max_length": args.max_length, "seed": args.seed,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    logger.info("SFT v2: iniciando %d época(s), LoRA r=%d/alpha=%d...",
                args.epochs, args.lora_r, lora_alpha)
    res = trainer.train()
    logger.info("SFT v2 concluído. train_loss=%.4f", res.training_loss)
    try:
        ev = trainer.evaluate()
        logger.info("eval_loss final=%.4f", ev.get("eval_loss", float("nan")))
        (out_root / f"metrics_r{args.lora_r}.json").write_text(json.dumps({
            "train_loss": res.training_loss, **ev}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("eval falhou: %s", e)

    adapter_dir = out_root / tag
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    logger.info("adapter salvo em %s", adapter_dir)

    # libera o trainer antes do export (recarrega o base) — folga de memória.
    del trainer, model
    torch.cuda.empty_cache()
    try:
        export_final(args.base, adapter_dir, tag, models_dir, llama_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha export %s: %s", tag, e)


if __name__ == "__main__":
    main()
