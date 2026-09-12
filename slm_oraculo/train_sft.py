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
    escolha conservadora pedida pelo brief. Salvamos 1 adapter POR ÉPOCA e SELECIONAMOS o
    melhor pela régua externa em oráculo (train.py não aborta sozinho; a seleção é feita
    por eval_checkpoint.sh + consolidate.py, ordem do capitão).

ARQUITETURA DO LFM2.5-2.6B (verificada no weight_map, msg 003 do capitão):
  - atenção: self_attn.(q_proj|k_proj|v_proj|out_proj)  [8 camadas de atenção]
  - MLP:     feed_forward.(w1|w2|w3)                    [30 camadas — NÃO é gate/up/down_proj]
  - ShortConv: conv.(in_proj|conv|out_proj)             [22 camadas recorrentes — EXCLUIR]
Erro anterior: LORA_TARGETS=["...out_proj",...gate/up/down_proj] adaptou os 22 conv.out_proj
(ShortConv, recorrente/instável) e NÃO tocou o MLP (gate/up/down_proj não existem). Corrigido
para REGEX explícito que casa SÓ atenção + feed_forward.w1/w2/w3.

Checkpoints por época em ~/cemig-poc/train/sft_ep{N}. Merge+GGUF bf16+Q4_0 por época (p/
avaliar no MESMO engine do teto, separando efeito do SFT do efeito da quantização).
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

# Reduz fragmentação do caching allocator na memória unificada — ANTES de importar torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("slm_oraculo.train_sft")

# REGEX explícito (PEFT aceita str em target_modules): SÓ atenção + MLP (feed_forward.w1/w2/w3).
# NÃO casa conv.* (ShortConv recorrente). O '$' ancora o fim do nome do módulo.
LORA_TARGETS = r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))$"


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
    # Config ORIGINAL (ordem do capitão contra cautela excessiva, msg 003): bs=8, max_len=2048.
    # A causa do travamento foi treino + 2 llama-servers residentes competindo pelos 121 GB
    # unificados, NÃO o treino ser grande. run_train.sh mata os servers antes de treinar.
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
        # Dataset é conversacional ({messages:[...]}); o chat template do LFM2.5 tem
        # {% generation %}/{% endgeneration %}, então mascara-se a loss pelo turno do
        # assistant (assistant_only_loss), NÃO por completion_only_loss (esse é p/
        # datasets prompt/completion). Ordem do capitão, msg 003.
        assistant_only_loss=True,
        completion_only_loss=False,
        packing=False,
        group_by_length=True,   # agrupa por tamanho -> menos padding -> menor pico de VRAM
        dataloader_num_workers=2,
        report_to="wandb",
        run_name=f"slm_oraculo_sft_r{args.lora_r}",
        logging_first_step=True,
        include_num_input_tokens_seen=True,  # throughput (tokens vistos) na telemetria
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
            return control

    class MemLogCb(TrainerCallback):
        """Loga memória alocada/reservada por passo (lição do incidente: telemetria)."""
        def on_log(self, cfg_, state, control, logs=None, **kw):
            if logs is not None and torch.cuda.is_available():
                logs["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                logs["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
                logs["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            return control

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora,
                         callbacks=[ExportCb(), MemLogCb()])

    # VERIFICAÇÃO OBRIGATÓRIA (msg 003): os módulos adaptados DEVEM ser atenção + MLP
    # (feed_forward.w1/w2/w3) e NENHUM conv.* (ShortConv recorrente). Assert, não log —
    # isto teria pego o erro anterior na hora.
    trainer.model.print_trainable_parameters()
    targeted = set(getattr(trainer.model, "targeted_module_names", []))
    logger.info("Módulos adaptados (%d): %s", len(targeted), sorted(targeted)[:12])
    has_mlp = any(("feed_forward.w1" in m or "feed_forward.w2" in m or "feed_forward.w3" in m)
                  for m in targeted)
    has_conv = any(".conv." in m for m in targeted)
    has_attn = any(("self_attn.q_proj" in m or "self_attn.out_proj" in m) for m in targeted)
    assert has_mlp, f"LoRA NÃO cobre o MLP (feed_forward.w1/w2/w3)! targeted={sorted(targeted)[:20]}"
    assert has_attn, f"LoRA NÃO cobre a atenção! targeted={sorted(targeted)[:20]}"
    assert not has_conv, f"LoRA está adaptando o ShortConv (conv.*)! targeted={sorted(targeted)[:20]}"
    logger.info("VERIFICAÇÃO OK: LoRA cobre atenção + MLP, e NÃO toca o ShortConv.")

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
