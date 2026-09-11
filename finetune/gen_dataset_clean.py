#!/usr/bin/env python3
"""
gen_dataset_clean.py — Geração DESCONTAMINADA do dataset de reescrita (M3).

Corrige o vício de contaminação da trilha anterior (finetune/gen_dataset.py), onde
as falas sintéticas e as golden_keywords copiavam o vocabulário literal do chunk —
inflando artificialmente o recall zero-shot (o "92%" era contra o próprio dataset
invertido). Aqui:

  1. O prompt de geração PROÍBE reutilizar substantivos técnicos do trecho e exige
     paráfrase de leigo (fala de campo com gíria/ASR), registrando termos proibidos.
  2. Filtro de descontaminação pós-hoc: rejeita falas com sobreposição lexical alta
     (Jaccard de radicais > limiar) com o texto do chunk.
  3. NÃO grava golden_keywords derivadas do chunk. O alvo de treino (assistant) é
     gerado separadamente como termos técnicos CANÔNICOS da norma (não do chunk),
     de forma que o modelo aprenda a MAPEAR fala->norma, não a copiar.
  4. Cada linha carrega `generator` (procedência) e `contamination_score`.

O holdout de avaliação é SEMPRE `corpus/qa_pairs_v2.jsonl` (151 perguntas reais),
nunca visto em treino/val — garantido por construção (fonte distinta).

Uso:
    python3 finetune/gen_dataset_clean.py --vllm-url http://10.100.0.111:8005/v1 \
        --model Qwen/Qwen3.8-27B-FP8 --max-chunks 700 --out finetune/data/clean_raw.jsonl
"""

import argparse
import json
import logging
import re
import sqlite3
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gen_clean")

DEFAULT_DB = "corpus/index_hf_36nr.db"
DEFAULT_OUT = "finetune/data/clean_raw.jsonl"

# Prioridade regulamentar (5 críticas + tier 2/3), igual à trilha anterior para cobertura comparável
PRIORITY = (
    ["nr-10", "nr-06", "nr-35", "nr-12", "nr-18"]
    + ["nr-01", "nr-16", "nr-26", "nr-33"]
    + ["nr-17", "nr-11", "nr-07", "nr-09", "nr-20", "nr-05",
       "nr-38", "nr-13", "nr-19", "nr-24", "nr-22", "nr-31"]
)

PT_STOP = {
    "para", "como", "pelo", "pela", "onde", "quando", "este", "esta", "deve", "devem",
    "serao", "sendo", "que", "com", "dos", "das", "uma", "por", "nao", "sua", "seu",
    "aos", "nas", "nos", "num", "numa", "ser", "sao", "tem", "mais", "seus", "suas",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def stems(text: str, min_len: int = 4) -> Set[str]:
    """Radicais (6 chars) de palavras de conteúdo, sem acento — para medir sobreposição lexical."""
    toks = re.findall(r"[a-z]{%d,}" % min_len, strip_accents(text).lower())
    return {t[:6] for t in toks if t not in PT_STOP}


def contamination_jaccard(question: str, chunk_text: str) -> float:
    """Jaccard entre radicais da fala e do chunk. Alto = fala copiou o vocabulário do trecho."""
    qs, cs = stems(question), stems(chunk_text)
    if not qs:
        return 0.0
    return len(qs & cs) / len(qs)


def load_chunks(db: str, max_chunks: int) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    out: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for nr in PRIORITY:
        cur.execute("SELECT id, doc, section, title, text FROM chunks WHERE doc=? ORDER BY id", (nr,))
        for r in cur.fetchall():
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(dict(r))
            if len(out) >= max_chunks:
                con.close()
                logger.info("Carregados %d chunks priorizados", len(out))
                return out
    con.close()
    logger.info("Carregados %d chunks priorizados", len(out))
    return out


GEN_SYSTEM = (
    "Você gera dados de treino para um assistente de campo da CEMIG (setor elétrico).\n"
    "Recebe um trecho oficial de Norma Regulamentadora (NR) e produz falas REALISTAS de "
    "trabalhadores de campo cuja resposta está nesse trecho.\n\n"
    "REGRA DE OURO — DESCONTAMINAÇÃO (obrigatória):\n"
    "As falas NÃO PODEM reutilizar os substantivos técnicos do trecho. O trabalhador é LEIGO: "
    "descreve a SITUAÇÃO concreta com gíria de canteiro/subestação, sem saber o termo técnico. "
    "Ex.: em vez de 'desenergização' diga 'desligar o negócio'; em vez de 'aterramento temporário' "
    "diga 'aquele fio que joga pra terra'; em vez de 'vestimenta condutível' diga 'a roupa/camisa'.\n\n"
    "Para cada fala, forneça também 'canonical_terms': de 3 a 6 termos técnicos CANÔNICOS da norma "
    "que um especialista usaria para buscar a resposta (esses SIM podem ser técnicos e incluir o "
    "código da NR, ex.: 'nr-10'). Esses termos representam a REESCRITA IDEAL, não a fala.\n\n"
    "FORMATO: retorne ESTRITAMENTE um array JSON de 2 a 3 objetos:\n"
    "[{\n"
    '  "persona": "novato"|"veterano",\n'
    '  "register": "giria"|"formal",\n'
    '  "asr_error": true|false,\n'
    '  "question": "fala leiga do trabalhador (SEM jargão do trecho)",\n'
    '  "canonical_terms": "nr-XX termo1 termo2 termo3"\n'
    "}]"
)


def build_user(chunk: Dict[str, Any]) -> str:
    body = chunk["text"]
    if len(body) > 800:
        body = body[:800] + "..."
    # Lista de substantivos técnicos proibidos (amostra dos radicais do trecho) p/ reforçar a regra
    banned = sorted(list(stems(body)))[:18]
    return (
        f"Norma: {chunk['doc'].upper()}  Item: {chunk['section']}  Título: {chunk['title']}\n"
        f"Trecho normativo:\n{body}\n\n"
        f"Substantivos técnicos que a FALA deve EVITAR (use paráfrase leiga): {', '.join(banned)}\n\n"
        f"Gere o array JSON com 2 a 3 falas leigas + canonical_terms técnicos."
    )


def call_vllm(base_url: str, model: str, chunk: Dict[str, Any], timeout: float = 60.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": build_user(chunk)},
        ],
        "temperature": 0.5,
        "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def parse_array(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            text = m.group(1)
    try:
        p = json.loads(text)
        if isinstance(p, list):
            return p
        if isinstance(p, dict):
            return [p]
    except Exception:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e > s:
            try:
                p = json.loads(text[s:e + 1])
                if isinstance(p, list):
                    return p
            except Exception:
                pass
    return []


def canonicalize_terms(raw_kw: Any, doc: str) -> str:
    """Higieniza canonical_terms e garante o código da NR presente (mapeamento fala->norma)."""
    if isinstance(raw_kw, list):
        kw = " ".join(str(k) for k in raw_kw)
    else:
        kw = str(raw_kw or "")
    toks = [w for w in re.findall(r"[\w.\-]+", kw.lower()) if len(w) > 1]
    if doc.lower() not in toks:
        toks.insert(0, doc.lower())
    return " ".join(toks[:6])


def process_chunk(chunk: Dict[str, Any], base_url: str, model: str, contam_max: float) -> List[Dict[str, Any]]:
    try:
        raw = call_vllm(base_url, model, chunk)
    except Exception as e:
        logger.debug("Falha vLLM chunk %d: %s", chunk["id"], e)
        return []
    items = parse_array(raw)
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        q = (it.get("question") or "").strip()
        if len(q) < 12:
            continue
        contam = contamination_jaccard(q, chunk["text"])
        if contam > contam_max:
            continue  # descontaminação: fala copiou vocabulário demais
        canon = canonicalize_terms(it.get("canonical_terms"), chunk["doc"])
        out.append({
            "id": f"clean-{chunk['doc']}-{chunk['id']}-{i:02d}",
            "chunk_id": chunk["id"],
            "doc": chunk["doc"],
            "section": chunk["section"],
            "title": chunk["title"],
            "persona": it.get("persona", "novato"),
            "register": it.get("register", "giria"),
            "asr_error": bool(it.get("asr_error", False)),
            "question": q,
            "canonical_terms": canon,
            "contamination_score": round(contam, 3),
            "generator": model,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Geração descontaminada do dataset de reescrita (M3).")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--vllm-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--max-chunks", type=int, default=700)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--contam-max", type=float, default=0.35, help="Jaccard máx de radicais fala vs chunk.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint incremental (VPN oscila): retoma por chunk_id já gravado
    done_chunks: Set[int] = set()
    n_existing = 0
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done_chunks.add(json.loads(line)["chunk_id"])
                n_existing += 1
            except Exception:
                pass
        logger.info("Checkpoint: %d pares, %d chunks já feitos", n_existing, len(done_chunks))

    chunks = load_chunks(args.db, args.max_chunks)
    if args.limit:
        chunks = chunks[:args.limit]
    pending = [c for c in chunks if c["id"] not in done_chunks]
    logger.info("Chunks pendentes: %d", len(pending))
    if not pending:
        logger.info("Nada a fazer.")
        return

    lock = Lock()
    fh = open(out_path, "a", encoding="utf-8")
    total = n_existing
    dropped = 0
    done = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_chunk, c, args.vllm_url, args.model, args.contam_max): c for c in pending}
            for fut in as_completed(futs):
                items = fut.result()
                with lock:
                    for it in items:
                        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                        total += 1
                    fh.flush()
                    done += 1
                    if done % 25 == 0 or done == len(pending):
                        sp = done / max(time.time() - t0, 0.1)
                        logger.info("[%d/%d chunks] %d pares | %.2f ch/s", done, len(pending), total, sp)
    finally:
        fh.close()
    logger.info("Concluído: %d pares em %s (%.1fs)", total, out_path, time.time() - t0)


if __name__ == "__main__":
    main()
