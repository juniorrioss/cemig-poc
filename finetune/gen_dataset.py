#!/usr/bin/env python3
"""
gen_dataset.py - Inverted synthetic dataset generation (chunk -> worker utterances)
for fine-tuning the technical query rewriter (Turn 1 Mode C).

Gera falas coloquiais de operários/eletricistas a partir de trechos de NRs (chunks)
utilizando vLLM corporativo (Qwen/Qwen3.8-27B-FP8) ou fallback local (llama-server).
Suporta retentativas com backoff exponencial, timeout e salvamento incremental em JSONL.
"""

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("gen_dataset")

# NRs prioritárias conforme especificação Firstmate (5 críticas + 4 adicionais + utilidades industriais)
PRIORITY_NRS_TIER1 = ["nr-10", "nr-06", "nr-35", "nr-12", "nr-18"]
PRIORITY_NRS_TIER2 = ["nr-01", "nr-16", "nr-26", "nr-33"]
PRIORITY_NRS_TIER3 = [
    "nr-17", "nr-11", "nr-07", "nr-09", "nr-20", "nr-05",
    "nr-38", "nr-13", "nr-19", "nr-24", "nr-22", "nr-31"
]

DEFAULT_DB_PATH = Path("corpus/index_hf_36nr.db")
DEFAULT_OUTPUT_PATH = Path("finetune/data/synthetic_raw.jsonl")
DEFAULT_VLLM_URL = "http://10.100.0.111:8005/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LOCAL_MODEL = "LFM2.5-1.2B-Instruct-Q4_K_M"

# Erros fonéticos comuns em transcrição ASR de campo (áudio ruidoso, sotaque mineiro/regional)
COMMON_ASR_PHONETICS = [
    ("disjuntor", "disjunto"),
    ("fusível", "fuzivel"),
    ("fusível", "fuzíve"),
    ("eletricista", "eletrecista"),
    ("aterramento", "ateramento"),
    ("NR-10", "nr déis"),
    ("NR-35", "nr trinta e cinco"),
    ("NR-06", "nr seis"),
    ("NR-12", "nr doze"),
    ("talabarte", "talabart"),
    ("ancoragem", "ancorage"),
    ("equipotencialização", "equipotensializacao"),
    ("subestação", "subestacao"),
    ("chave faca", "xave faca"),
    ("transformador", "trasformador"),
    ("seccionadora", "secionadora"),
    ("luva de vaqueta", "luva de baqueta"),
    ("alta tensão", "alta tensao"),
    ("bloqueio", "bloqueo"),
]


def load_prioritized_chunks(db_path: Path, max_chunks: int = 800) -> List[Dict[str, Any]]:
    """
    Carrega chunks do banco SQLite ordenados por prioridade regulamentar.
    Prioriza NRs críticas para o ambiente elétrico e industrial da CEMIG.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    all_chunks: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()

    for nr_tier in [PRIORITY_NRS_TIER1, PRIORITY_NRS_TIER2, PRIORITY_NRS_TIER3]:
        for nr in nr_tier:
            cur.execute(
                "SELECT id, doc, section, title, page, text FROM chunks WHERE doc = ? ORDER BY id ASC",
                (nr,)
            )
            rows = cur.fetchall()
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_chunks.append({
                        "id": r["id"],
                        "doc": r["doc"],
                        "section": r["section"],
                        "title": r["title"],
                        "page": r["page"],
                        "text": r["text"]
                    })
                    if len(all_chunks) >= max_chunks:
                        break
            if len(all_chunks) >= max_chunks:
                break
        if len(all_chunks) >= max_chunks:
            break

    # Se ainda faltar para o limite de chunks, preenche com as demais NRs
    if len(all_chunks) < max_chunks:
        cur.execute("SELECT id, doc, section, title, page, text FROM chunks ORDER BY id ASC")
        rows = cur.fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_chunks.append({
                    "id": r["id"],
                    "doc": r["doc"],
                    "section": r["section"],
                    "title": r["title"],
                    "page": r["page"],
                    "text": r["text"]
                })
                if len(all_chunks) >= max_chunks:
                    break

    con.close()
    logger.info("Carregados %d chunks priorizados de %s", len(all_chunks), db_path)
    return all_chunks


def load_existing_checkpoint(output_path: Path) -> Tuple[Set[int], int]:
    """
    Carrega IDs de chunks já processados para possibilitar retomada incremental.
    """
    if not output_path.exists():
        return set(), 0

    processed_chunk_ids: Set[int] = set()
    total_pairs = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                c_id = data.get("chunk_id")
                if c_id is not None:
                    processed_chunk_ids.add(c_id)
                total_pairs += 1
            except json.JSONDecodeError:
                continue

    logger.info("Checkpoint carregado: %d chunks já processados (%d pares existentes).",
                len(processed_chunk_ids), total_pairs)
    return processed_chunk_ids, total_pairs


def build_system_prompt() -> str:
    """Retorna o prompt de sistema especializado para o gerador sintético."""
    return (
        "Você é um especialista sênior em Normas Regulamentadoras brasileiras (NRs do MTE/segurança do trabalho) "
        "e engenharia de dados para LLMs de campo na CEMIG (setor elétrico/industrial).\n"
        "Sua tarefa é ler um trecho oficial de NR (chunk normativo) e criar de 2 a 4 dúvidas/falas realistas de campo "
        "que um trabalhador, eletricista ou técnico faria e cuja resposta técnica mandatória está exatamente contida no texto.\n\n"
        "DIRETRIZES DE GERAÇÃO:\n"
        "1. DIVERSIDADE DE PERSONAS:\n"
        "   - 'novato': Dúvidas ingênuas, receio, vocabulário leigo, perguntas conceituais de iniciante.\n"
        "   - 'veterano': Confiança operacional, contestação de regras vistas como lentidão/burocracia, proposta de atalho perigoso ('jeitinho').\n"
        "2. VARIAÇÃO DE REGISTRO:\n"
        "   - 'giria': Linguagem oral coloquial de canteiro/subestação ('gato', 'meter a chave', 'tá vivo', 'chave faca', 'cinta', 'talabarte', 'tá osso', 'dar um grau').\n"
        "   - 'formal': Linguagem técnica direta de encarregado ou supervisor de segurança.\n"
        "3. SIMULAÇÃO DE ERRO FONÉTICO DE ASR (~25% a 30% das falas):\n"
        "   - Simule transcrição imperfeita de áudio ruidoso (ex: 'disjunto', 'eletrecista', 'fuzivel', 'ateramento', 'nr déis', 'ancorage', 'xave faca', 'trasformador').\n"
        "4. CONTINUAÇÕES MULTITURNO (~25% das falas):\n"
        "   - Defina 'is_multiturn: true', com 'history' curto que dá contexto situacional e uma pergunta ('question') que depende dele.\n"
        "5. PALAVRAS-CHAVE TÉCNICAS (golden_keywords):\n"
        "   - Extraia de 3 a 6 termos técnicos e conceituais indispensáveis para o algoritmo BM25 recuperar este trecho no índice SQLite FTS5.\n"
        "   - Devem conter o nome da norma (ex: 'nr-10'), número do item se relevante (ex: '10.5.1') e palavras-chave específicas do procedimento/risco.\n\n"
        "FORMATO DE RESPOSTA:\n"
        "Retorne ESTRITAMENTE um array JSON contendo entre 2 e 4 objetos com os seguintes campos:\n"
        "[\n"
        "  {\n"
        "    \"persona\": \"novato\" | \"veterano\",\n"
        "    \"register\": \"giria\" | \"formal\",\n"
        "    \"asr_error\": true | false,\n"
        "    \"is_multiturn\": true | false,\n"
        "    \"history\": \"contexto prévio ou vazio\",\n"
        "    \"question\": \"fala do trabalhador\",\n"
        "    \"golden_keywords\": \"3 a 6 termos técnicos separados por espaço\"\n"
        "  }\n"
        "]"
    )


def build_user_prompt(chunk: Dict[str, Any]) -> str:
    """Constrói a mensagem de entrada do chunk para geração invertida."""
    # Trunca texto se for excessivamente longo para manter contexto enxuto
    text_sample = chunk["text"]
    if len(text_sample) > 800:
        text_sample = text_sample[:800] + "..."

    return (
        f"Norma Regulamentadora: {chunk['doc'].upper()}\n"
        f"Item/Seção: {chunk['section']}\n"
        f"Título: {chunk['title']}\n"
        f"Conteúdo Normativo:\n{text_sample}\n\n"
        f"Gere o array JSON com 2 a 3 falas realistas de operários que demandam este trecho."
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=25),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError)),
    reraise=True
)
def call_generation_endpoint(
    base_url: str,
    model: str,
    chunk: Dict[str, Any],
    timeout: float = 35.0
) -> str:
    """
    Chama a API OpenAI-compatível com política de retry exponencial.
    Suporta reconexão automática em oscilações de VPN.
    """
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(chunk)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.35,
        "max_tokens": 380,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content", "")
    return content


def extract_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """Extrai e valida o array JSON gerado pelo LLM, tratando marcações markdown."""
    text = raw_text.strip()
    # Remove blocos de código markdown se presentes
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if match:
            text = match.group(1)
        else:
            match_any = re.search(r"\[[\s\S]*\]", text)
            if match_any:
                text = match_any.group(0)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Se devolveu um único objeto ou chave wrapper
            if "items" in parsed and isinstance(parsed["items"], list):
                return parsed["items"]
            return [parsed]
    except Exception:
        # Fallback de busca por colchetes
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                sub = text[start_idx:end_idx + 1]
                parsed = json.loads(sub)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def process_single_chunk(
    chunk: Dict[str, Any],
    active_url: str,
    active_model: str,
    seq_start: int,
    force_fallback: bool = False
) -> List[Dict[str, Any]]:
    """Processa um único chunk: invoca o endpoint ou usa gerador heurístico e formata os pares."""
    chunk_id = chunk["id"]
    if force_fallback:
        return heuristic_seed_generation(chunk, seq_start)

    endpoint_ready = test_endpoint_health(active_url, timeout=2.0)
    if endpoint_ready:
        try:
            raw_resp = call_generation_endpoint(active_url, active_model, chunk)
            extracted = extract_json_array(raw_resp)
            if extracted:
                return postprocess_generated_items(extracted, chunk, seq_start)
            else:
                return heuristic_seed_generation(chunk, seq_start)
        except Exception as e:
            logger.debug("Falha na geração via LLM chunk %d: %s. Usando contingência heurística.", chunk_id, e)
            return heuristic_seed_generation(chunk, seq_start)
    else:
        return heuristic_seed_generation(chunk, seq_start)

    """
    Limpa, formata e enriquece os itens gerados com metadados do chunk.
    Garante ancoragem técnica para validação BM25 posterior.
    """
    valid_items: List[Dict[str, Any]] = []
    doc_clean = chunk["doc"].lower()
    sec_clean = chunk["section"].strip()
    title_terms = " ".join([t for t in re.findall(r"\w+", chunk["title"].lower()) if len(t) > 3][:3])

    for i, it in enumerate(items):
        question = (it.get("question") or "").strip()
        if not question or len(question) < 8:
            continue

        persona = it.get("persona", "novato")
        if persona not in ("novato", "veterano"):
            persona = random.choice(["novato", "veterano"])

        register = it.get("register", "giria")
        asr_error = bool(it.get("asr_error", False))
        is_multiturn = bool(it.get("is_multiturn", False))
        history = (it.get("history") or "").strip()

        # Normaliza golden_keywords
        raw_kw = it.get("golden_keywords", "")
        if isinstance(raw_kw, list):
            kw_str = " ".join(str(k) for k in raw_kw)
        else:
            kw_str = str(raw_kw).strip()

        # Garante que as palavras-chave incluam o doc e termos nucleares
        kw_tokens = [w for w in re.findall(r"[\w\.-]+", kw_str.lower()) if len(w) > 1]
        if doc_clean not in kw_tokens:
            kw_tokens.insert(0, doc_clean)
        if sec_clean and sec_clean not in kw_tokens and len(kw_tokens) < 6:
            kw_tokens.append(sec_clean)

        final_keywords = " ".join(kw_tokens[:6])

        item_id = f"synth-{doc_clean}-{chunk['id']}-{start_seq_id + i:04d}"
        valid_items.append({
            "id": item_id,
            "chunk_id": chunk["id"],
            "doc": chunk["doc"],
            "section": chunk["section"],
            "title": chunk["title"],
            "page": chunk["page"],
            "persona": persona,
            "register": register,
            "asr_error": asr_error,
            "is_multiturn": is_multiturn,
            "history": history,
            "question": question,
            "golden_keywords": final_keywords
        })

    return valid_items


def heuristic_seed_generation(chunk: Dict[str, Any], start_seq_id: int) -> List[Dict[str, Any]]:
    """
    Gerador determinístico estruturado de contingência local quando nenhum LLM estiver disponível.
    Produz falas coloquiais críveis baseadas nos substantivos e verbos do trecho normativo.
    """
    doc = chunk["doc"].lower()
    sec = chunk["section"]
    title = chunk["title"]
    text = chunk["text"]

    # Extrai termos mais frequentes com comprimento > 4
    words = [w.lower() for w in re.findall(r"[a-záéíóúâêîôûãõç]{4,}", text)]
    stopwords = {"para", "como", "pelo", "pela", "onde", "quando", "este", "esta", "deve", "devem", "serão", "sendo"}
    candidates = [w for w in words if w not in stopwords]
    freq: Dict[str, int] = {}
    for w in candidates:
        freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:5]
    terms_str = " ".join(top_words[:3])

    kw_gold = f"{doc} {sec} {title} {terms_str}"
    kw_clean = " ".join([w for w in re.findall(r"[\w\.-]+", kw_gold.lower()) if len(w) > 1][:6])

    items: List[Dict[str, Any]] = []

    # 1. Novato em gíria
    q1 = f"Ô companheiro, como é que fica o esquema de {title.lower()} aqui na obra? Tem que fazer antes de mexer?"
    items.append({
        "id": f"seed-{doc}-{chunk['id']}-{start_seq_id:04d}",
        "chunk_id": chunk["id"],
        "doc": chunk["doc"],
        "section": chunk["section"],
        "title": chunk["title"],
        "page": chunk["page"],
        "persona": "novato",
        "register": "giria",
        "asr_error": False,
        "is_multiturn": False,
        "history": "",
        "question": q1,
        "golden_keywords": kw_clean
    })

    # 2. Veterano com atalho / dúvida de campo
    q2 = f"A gente sempre fez sem essa frescura de {title.lower()}, mas o encarregado falou que a {doc.upper()} exige agora. Pode dispensar isso aí?"
    items.append({
        "id": f"seed-{doc}-{chunk['id']}-{start_seq_id+1:04d}",
        "chunk_id": chunk["id"],
        "doc": chunk["doc"],
        "section": chunk["section"],
        "title": chunk["title"],
        "page": chunk["page"],
        "persona": "veterano",
        "register": "giria",
        "asr_error": False,
        "is_multiturn": False,
        "history": "",
        "question": q2,
        "golden_keywords": kw_clean
    })

    # 3. Multiturno com erro ASR simulado
    q3 = f"E pra fazer esse procedimento de {title.lower()} precisa de ateramento temporario ou o disjunto ja segura?"
    items.append({
        "id": f"seed-{doc}-{chunk['id']}-{start_seq_id+2:04d}",
        "chunk_id": chunk["id"],
        "doc": chunk["doc"],
        "section": chunk["section"],
        "title": chunk["title"],
        "page": chunk["page"],
        "persona": "novato",
        "register": "giria",
        "asr_error": True,
        "is_multiturn": True,
        "history": f"Estamos verificando as condições de segurança da {doc.upper()} para início de manutenção preventiva.",
        "question": q3,
        "golden_keywords": kw_clean
    })

    return items


def test_endpoint_health(url: str, timeout: float = 3.0) -> bool:
    """Verifica se o endpoint está operacional."""
    try:
        r = requests.get(f"{url.rstrip('/')}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Geração sintética de pares de reescrita (chunk -> falas de operário).")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco SQLite de chunks")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Arquivo JSONL de saída")
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM_URL, help="URL base do endpoint vLLM")
    parser.add_argument("--vllm-model", type=str, default=DEFAULT_VLLM_MODEL, help="Nome do modelo no vLLM")
    parser.add_argument("--local-url", type=str, default=DEFAULT_LOCAL_URL, help="URL do llama-server local")
    parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL, help="Nome do modelo local")
    parser.add_argument("--local-fallback", action="store_true", help="Usa gerador determinístico/local se vLLM indisponível")
    parser.add_argument("--max-chunks", type=int, default=800, help="Número máximo de chunks a processar")
    parser.add_argument("--workers", type=int, default=4, help="Número de threads concorrentes para geração")
    parser.add_argument("--batch-checkpoint", type=int, default=50, help="Intervalo de chunks para log de checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limita a N chunks para teste rápido (0 = sem limite)")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db_path)

    chunks = load_prioritized_chunks(db_path, max_chunks=args.max_chunks)
    if args.limit > 0:
        chunks = chunks[:args.limit]
        logger.info("Limite de teste ativo: restringindo a %d chunks.", len(chunks))

    processed_chunk_ids, total_pairs_existing = load_existing_checkpoint(output_path)
    pending_chunks = [c for c in chunks if c["id"] not in processed_chunk_ids]
    logger.info("Total de chunks pendentes para geração: %d", len(pending_chunks))

    if not pending_chunks:
        logger.info("Todos os %d chunks já foram processados! Nada a fazer.", len(chunks))
        return

    # Testa disponibilidade do endpoint primário vLLM
    use_vllm = test_endpoint_health(args.vllm_url)
    active_url = args.vllm_url if use_vllm else args.local_url
    active_model = args.vllm_model if use_vllm else args.local_model
    endpoint_mode = "vLLM corporativo" if use_vllm else "Local llama-server / Fallback"

    if use_vllm:
        logger.info("Conectado com sucesso ao vLLM: %s (%s)", active_url, active_model)
    else:
        logger.warning("vLLM (%s) INDISPONÍVEL (VPN oscilando).", args.vllm_url)
        # Verifica se llama-server local está de pé
        if test_endpoint_health(args.local_url):
            logger.info("Detectado llama-server local ativo em %s. Utilizando para geração.", args.local_url)
        else:
            logger.warning("llama-server local não detectado em %s. Ativando gerador estruturado de contingência.", args.local_url)

    out_file = open(output_path, "a", encoding="utf-8")
    file_lock = threading.Lock()
    generated_count = total_pairs_existing
    chunks_done = 0
    t0 = time.time()

    def worker_task(chunk_info: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        idx, ch = chunk_info
        # Estimativa de seq_start baseada no índice
        items = process_single_chunk(
            ch,
            active_url,
            active_model,
            seq_start=idx * 4,
            force_fallback=args.local_fallback
        )
        return ch["id"], items

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            tasks = [executor.submit(worker_task, (i, chunk)) for i, chunk in enumerate(pending_chunks, 1)]
            for future in as_completed(tasks):
                chunk_id, items_to_save = future.result()
                with file_lock:
                    for item in items_to_save:
                        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                        generated_count += 1
                    out_file.flush()

                    chunks_done += 1
                    if chunks_done % args.batch_checkpoint == 0 or chunks_done == len(pending_chunks):
                        elapsed = time.time() - t0
                        speed = chunks_done / max(elapsed, 0.1)
                        logger.info(
                            "Progresso: [%d/%d chunks] | %d pares acumulados | %.2f chunks/s | Modo: %s",
                            chunks_done, len(pending_chunks), generated_count, speed, endpoint_mode
                        )
    finally:
        out_file.close()

    total_time = time.time() - t0
    logger.info("Geração finalizada com sucesso: %d pares em %s (tempo: %.1fs)",
                generated_count, output_path, total_time)


if __name__ == "__main__":
    main()
