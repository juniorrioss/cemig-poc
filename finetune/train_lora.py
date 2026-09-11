#!/usr/bin/env python3
"""
train_lora.py - PEFT LoRA Fine-Tuning for Query Rewriter (Turn 1 Mode C).

Treina um adaptador LoRA no modelo LiquidAI/LFM2.5-1.2B-Instruct para converter
falas coloquiais de campo em palavras-chave técnicas para busca BM25 nas NRs.
Incorpora ~10% de dados de manutenção de síntese (qa_pairs_v2) para ancoragem comportamental.
Salva tanto o adaptador isolado quanto o modelo fundido (merged) e converte para GGUF.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_lora")

DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
DEFAULT_DATA_DIR = Path("finetune/data")
DEFAULT_ADAPTER_DIR = Path("finetune/adapter")
DEFAULT_MERGED_DIR = Path("finetune/merged_model")
DEFAULT_QA_PAIRS = Path("corpus/qa_pairs_v2.jsonl")
DEFAULT_DB_PATH = Path("corpus/index_hf_36nr.db")

# System prompt de síntese (Turno 2) para ancoragem de manutenção
SYNTHESIS_SYSTEM_PROMPT = (
    "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras (NR-10, NR-06, NR-35, NR-12 e NR-18).\n"
    "Suas diretrizes mandatórias:\n"
    "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
    "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. Não adicione procedimentos não contidos nas normas.\n"
    "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
    "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
    "'Não sei com base nas normas consultadas.' Não tente adivinhar."
)


def load_chatml_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Carrega dados no formato ChatML."""
    if not filepath.exists():
        return []
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(data)
            except json.JSONDecodeError:
                continue
    return items


def load_maintenance_pairs(
    qa_path: Path,
    db_path: Path,
    target_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Carrega pares de síntese com chunk normativo injetado a partir de qa_pairs_v2.jsonl.
    Esses exemplos ancoram a capacidade de resposta técnica e impedem o esquecimento catastrófico.
    """
    if not qa_path.exists() or not db_path.exists():
        logger.warning("qa_pairs_v2 ou index.db não encontrados. Pulando dados de manutenção.")
        return []

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Mapeia chunk_id para texto
    chunk_cache: Dict[int, str] = {}
    cur.execute("SELECT id, text FROM chunks")
    for r in cur.fetchall():
        chunk_cache[r["id"]] = r["text"]
    con.close()

    maintenance_items: List[Dict[str, Any]] = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = p.get("chunk_id")
            chunk_text = chunk_cache.get(cid, "")
            if not chunk_text and p.get("relevant_chunk_ids"):
                for rcid in p["relevant_chunk_ids"]:
                    if rcid in chunk_cache:
                        chunk_text = chunk_cache[rcid]
                        break

            if not chunk_text:
                continue

            # Formata no estilo de Turno 2 do Modo C
            user_content = f"Contexto normativo consultado:\n[1] {chunk_text}\n\nPergunta do eletricista:\n{p['question']}"
            assistant_content = p["golden_answer"]

            maintenance_items.append({
                "messages": [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": {
                    "id": f"maint-{p['id']}",
                    "type": "maintenance_synthesis",
                    "doc": p.get("doc", ""),
                    "section": p.get("section", "")
                }
            })

    random.shuffle(maintenance_items)
    selected = maintenance_items[:target_count]
    logger.info("Carregados %d pares de manutenção de síntese (ancoragem).", len(selected))
    return selected


def prepare_tokenized_dataset(
    items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024
) -> Dataset:
    """
    Tokeniza os dados com mascaramento de perda completion-only:
    Os tokens de sistema e usuário recebem label -100.
    Apenas os tokens de resposta do assistente geram gradiente.
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in items:
        msgs = item["messages"]
        if len(msgs) < 2:
            continue

        # Formatação completa e formatação do prompt (sem resposta do assistente)
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

        full_toks = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
        prompt_toks = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt")

        inp_ids = full_toks["input_ids"][0]
        att_mask = full_toks["attention_mask"][0]
        p_len = len(prompt_toks["input_ids"][0])

        labels = inp_ids.clone()
        # Mascara o prompt com -100
        labels[:min(p_len, len(labels))] = -100

        # Se após o mascaramento nenhum token do assistente sobrou, descarta
        if (labels != -100).sum() == 0:
            continue

        input_ids_list.append(inp_ids.tolist())
        labels_list.append(labels.tolist())
        attention_mask_list.append(att_mask.tolist())

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    })


def convert_adapter_to_gguf(adapter_dir: Path) -> Optional[Path]:
    """Converte o adaptador LoRA salvo para formato GGUF via script oficial do llama.cpp."""
    convert_script = Path.home() / "llama.cpp" / "convert_lora_to_gguf.py"
    if not convert_script.exists():
        logger.warning("Script %s não encontrado. Pulando conversão de LoRA para GGUF.", convert_script)
        return None

    gguf_path = adapter_dir / "rewriter-lora-f16.gguf"
    cmd = [
        sys.executable,
        str(convert_script),
        str(adapter_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ]
    logger.info("Convertendo adaptador LoRA para GGUF: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Conversão de LoRA para GGUF concluída: %s (%.2f MB)",
                    gguf_path, gguf_path.stat().st_size / (1024 * 1024))
        return gguf_path
    except subprocess.CalledProcessError as e:
        logger.warning("Falha na conversão de LoRA para GGUF: %s\nStderr: %s", e, e.stderr)
        return None


def convert_merged_to_gguf(merged_dir: Path) -> Optional[Path]:
    """Converte o modelo fundido para GGUF via convert_hf_to_gguf.py."""
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.warning("Script %s não encontrado. Pulando conversão do modelo fundido.", convert_script)
        return None

    gguf_path = merged_dir / "lfm2.5-1.2b-rewriter-f16.gguf"
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ]
    logger.info("Convertendo modelo fundido para GGUF: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Conversão do modelo fundido para GGUF concluída: %s (%.2f MB)",
                    gguf_path, gguf_path.stat().st_size / (1024 * 1024))
        return gguf_path
    except subprocess.CalledProcessError as e:
        logger.warning("Falha na conversão do modelo fundido para GGUF: %s\nStderr: %s", e, e.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Treino LoRA do rewriter (LiquidAI/LFM2.5-1.2B-Instruct).")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="ID do modelo no HF")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Diretório com train/val.jsonl")
    parser.add_argument("--adapter-dir", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Diretório para salvar o adapter")
    parser.add_argument("--merged-dir", type=str, default=str(DEFAULT_MERGED_DIR), help="Diretório para salvar o modelo fundido")
    parser.add_argument("--qa-path", type=str, default=str(DEFAULT_QA_PAIRS), help="Arquivo qa_pairs_v2.jsonl")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco index_hf_36nr.db")
    parser.add_argument("--lora-r", type=int, default=16, help="Rank LoRA (default 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Alpha LoRA (default 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default 1e-4)")
    parser.add_argument("--epochs", type=int, default=2, help="Épocas de treino (default 2)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size por GPU (default 4)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default 2)")
    parser.add_argument("--maint-ratio", type=float, default=0.10, help="Proporção de dados de manutenção de síntese (0.10)")
    parser.add_argument("--max-length", type=int, default=1024, help="Tamanho máximo de sequência")
    parser.add_argument("--seed", type=int, default=42, help="Seed randômica")
    parser.add_argument("--suffix", type=str, default="", help="Sufixo dos arquivos de split (ex.: _clean -> train_clean.jsonl).")
    parser.add_argument("--no-extra-maint", action="store_true", help="Não injeta manutenção extra (splits já contêm ancoragem).")

    args = parser.parse_args()

    # Configuração de seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / f"train{args.suffix}.jsonl"
    val_path = data_dir / f"val{args.suffix}.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}. Execute validate_dataset.py primeiro.")

    train_items = load_chatml_jsonl(train_path)
    val_items = load_chatml_jsonl(val_path)
    logger.info("Carregados %d itens de treino e %d de validação de %s", len(train_items), len(val_items), data_dir)

    # Injeta ~10% de pares de manutenção de síntese (ancoragem), salvo se os splits já a contêm
    if args.no_extra_maint:
        maint_items = []
        logger.info("Ancoragem extra desativada (--no-extra-maint); splits já contêm ancoragem.")
    else:
        maint_count = max(5, int(len(train_items) * args.maint_ratio))
        maint_items = load_maintenance_pairs(Path(args.qa_path), Path(args.db_path), target_count=maint_count)
    full_train_items = train_items + maint_items
    random.shuffle(full_train_items)

    logger.info("Conjunto de treino final: %d itens (%d reescrita + %d manutenção síntese)",
                len(full_train_items), len(train_items), len(maint_items))

    # Carrega Tokenizer e Modelo
    logger.info("Carregando tokenizer e modelo base: %s...", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepara datasets tokenizados com loss masking
    train_dataset = prepare_tokenized_dataset(full_train_items, tokenizer, max_length=args.max_length)
    val_dataset = prepare_tokenized_dataset(val_items, tokenizer, max_length=args.max_length) if val_items else None

    logger.info("Datasets preparados: Treino com %d exemplos, Validação com %d exemplos.",
                len(train_dataset), len(val_dataset) if val_dataset else 0)

    # Carrega modelo na GPU RTX 5070 em bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Carregando pesos em device: %s com bfloat16...", device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )

    # Configuração de LoRA direcionada exclusivamente às camadas de atenção
    # Conforme arquitetura LFM2.5 inspecionada: q_proj, k_proj, v_proj, out_proj
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        logger.warning("Falha ao aplicar LoRA com r=%d: %s. Tentando r=8...", args.lora_r, e)
        lora_config.r = 8
        lora_config.lora_alpha = 16
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Habilita gradient checkpointing para manter VRAM < 6GB na RTX 5070
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    output_checkpoints = Path("finetune/checkpoints")
    output_checkpoints.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_checkpoints),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True if device == "cuda" else False,
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if val_dataset and len(val_dataset) > 0 else "no",
        save_total_limit=1,
        dataloader_num_workers=2,
        report_to="none"
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset and len(val_dataset) > 0 else None,
        data_collator=collator
    )

    logger.info("Iniciando treinamento LoRA...")
    train_result = trainer.train()
    logger.info("Treinamento finalizado. Loss final: %.4f", train_result.training_loss)

    # 1. Salva o adaptador LoRA isolado
    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Salvando adaptador LoRA em %s...", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # 2. Converte adaptador para GGUF
    convert_adapter_to_gguf(adapter_dir)

    # 3. Funde o adaptador com o modelo base (merge and unload)
    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Realizando merge do adaptador com o modelo base...")
    merged_model = model.merge_and_unload()
    logger.info("Salvando modelo fundido em %s...", merged_dir)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # 4. Converte modelo fundido para GGUF
    convert_merged_to_gguf(merged_dir)

    logger.info("Pipeline de treinamento e exportação concluído com sucesso!")


if __name__ == "__main__":
    main()
