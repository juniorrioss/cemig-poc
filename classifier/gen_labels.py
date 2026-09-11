#!/usr/bin/env python3
"""
gen_labels.py — FASE A: geração do dataset rotulado de falas coloquiais -> NR-alvo.

Gera ~2000 falas de operário/eletricista rotuladas com a(s) NR(s)-alvo via vLLM
corporativo (Qwen3.8-27B-FP8). Estratificação (brief Firstmate):
  - ~60% concentradas nas 9 NRs relevantes ao eletricista (ELECTRICIAN_NRS);
  - resto distribuído nas demais 27 NRs do corpus;
  - ~15% de casos AMBÍGUOS (rótulo duplo: duas NRs plausíveis);
  - ~10% FORA-DE-ESCOPO (rótulo 'nenhuma': RH, folga, dúvida pessoal).

Anti-contaminação (CRÍTICO): o holdout FIXO (151 reais de qa_pairs_v2.jsonl + 20 do
smoke_qa_20.jsonl) NUNCA é gerado aqui; as falas sintéticas são de vocabulário de campo,
independentes desses conjuntos. Registramos `generator` e `prompt_kind` em cada linha
para procedência. Salvamento incremental em JSONL (VPN oscila).

Uso:
    python3 classifier/gen_labels.py --target 2000 --workers 6 \
        --out classifier/data/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from nr_taxonomy import (
    CLASSES,
    ELECTRICIAN_NRS,
    NONE_LABEL,
    NR_ONELINE,
    NR_TITLES,
    normalize_nr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gen_labels")

DEFAULT_VLLM_URL = "http://10.100.0.111:8005/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"

# Pares ambíguos plausíveis no campo elétrico (rótulo duplo). A ordem [primária, secundária]
# reflete a NR mais provável primeiro, mas ambas são aceitas como acerto.
AMBIGUOUS_PAIRS: List[Tuple[str, str]] = [
    ("nr-10", "nr-35"),   # trabalho elétrico no poste/torre em altura
    ("nr-10", "nr-06"),   # EPI específico para eletricidade
    ("nr-10", "nr-16"),   # periculosidade de energia elétrica
    ("nr-35", "nr-18"),   # altura no canteiro de obra / andaime
    ("nr-35", "nr-06"),   # cinto/talabarte como EPI de altura
    ("nr-10", "nr-33"),   # trabalho elétrico em espaço confinado (câmara subterrânea)
    ("nr-12", "nr-10"),   # máquina com painel elétrico
    ("nr-06", "nr-26"),   # EPI e sinalização de segurança
    ("nr-16", "nr-15"),   # periculosidade vs insalubridade (adicional)
    ("nr-18", "nr-12"),   # betoneira/serra no canteiro (máquina em obra)
    ("nr-01", "nr-06"),   # ordem de serviço + EPI
    ("nr-33", "nr-06"),   # espaço confinado + EPI/respirador
]

SYSTEM_PROMPT = (
    "Você é um gerador de dados de treinamento. Produz falas curtas, coloquiais e "
    "realistas de trabalhadores brasileiros de campo (eletricistas, operários, "
    "instaladores), como se fossem transcrições de voz feitas no celular durante o "
    "serviço. As falas têm gíria regional, erros leves, hesitação ('ô', 'tipo', 'aí'), "
    "e NÃO usam jargão técnico de norma nem citam o número da NR. Cada fala é uma "
    "dúvida ou situação real de segurança do trabalho."
)


def build_user_prompt_single(nr: str, n: int) -> str:
    """Prompt para gerar N falas cuja resposta correta é UMA norma (nr)."""
    desc = NR_ONELINE[nr]
    return (
        f"Gere {n} falas DIFERENTES de trabalhador de campo cuja dúvida seja resolvida "
        f"pela norma sobre: {desc}.\n"
        "Regras:\n"
        "1. Cada fala é curta (1 a 3 frases), coloquial, como fala espontânea de voz.\n"
        "2. NUNCA cite o número da norma nem palavras como 'norma', 'NR', 'regulamentadora'.\n"
        "3. NÃO copie os termos técnicos da descrição; use o vocabulário do operário "
        "('fio', 'choque', 'poste', 'luva', 'cinto', 'andaime', 'buraco', etc.).\n"
        "4. Varie situações, ferramentas e contextos.\n"
        'Responda SOMENTE um array JSON de strings, ex.: ["fala 1", "fala 2"].'
    )


def build_user_prompt_ambiguous(a: str, b: str, n: int) -> str:
    """Prompt para gerar N falas onde DUAS normas (a, b) são plausíveis simultaneamente."""
    return (
        f"Gere {n} falas DIFERENTES de trabalhador de campo em situações que envolvem "
        f"AO MESMO TEMPO dois temas: (1) {NR_ONELINE[a]}; e (2) {NR_ONELINE[b]}.\n"
        "Regras:\n"
        "1. A situação deve genuinamente misturar os dois temas (ex.: trabalho elétrico "
        "feito no alto de um poste envolve eletricidade E altura).\n"
        "2. Fala curta, coloquial, de voz; NUNCA cite número de norma nem a palavra 'norma'.\n"
        "3. Use vocabulário de operário, não copie os termos da descrição.\n"
        'Responda SOMENTE um array JSON de strings.'
    )


def build_user_prompt_none(n: int) -> str:
    """Prompt para gerar N falas FORA do escopo das NRs de segurança do trabalho."""
    return (
        f"Gere {n} falas DIFERENTES de trabalhador de campo que NÃO são sobre segurança "
        "do trabalho nem sobre nenhuma norma técnica. Exemplos de temas: dúvida de "
        "pagamento/salário/hora extra, folga e férias, onde fica o almoço, problema com "
        "o chefe, wi-fi, previsão do tempo, futebol, como preencher ponto, reembolso de "
        "combustível, conversa fiada.\n"
        "Regras:\n"
        "1. Fala curta, coloquial, de voz, realista.\n"
        "2. NÃO mencione risco, EPI, choque, altura, máquina, nem qualquer tema de "
        "segurança do trabalho.\n"
        'Responda SOMENTE um array JSON de strings.'
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=25),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError)),
    reraise=True,
)
def call_vllm(base_url: str, model: str, user_prompt: str, timeout: float = 90.0) -> str:
    """Chama a API OpenAI-compatível com retry exponencial (resiliente a VPN)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,  # diversidade alta nas falas
        "top_p": 0.95,
        "max_tokens": 1400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content", "") or ""


def extract_string_array(raw: str) -> List[str]:
    """Extrai um array JSON de strings, tolerando cercas markdown e lixo ao redor."""
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            text = m.group(1)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    out: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # Alguns modelos devolvem {"fala": "..."}
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
    return out


def plan_batches(target: int) -> List[Dict[str, Any]]:
    """Monta o plano de lotes estratificado (~60% eletricista, 15% ambíguo, 10% none)."""
    n_none = round(target * 0.10)
    n_amb = round(target * 0.15)
    n_single = target - n_none - n_amb
    n_elec = round(n_single * 0.60 / 0.75)  # dentro dos single: ~60% do total geral
    # Simplificação estável: 60% do total nas 9 elétricas, restante single nas outras 27
    n_elec = round(target * 0.60)
    n_other = n_single - n_elec
    if n_other < 0:
        n_elec += n_other
        n_other = 0

    batches: List[Dict[str, Any]] = []
    per_call = 12  # falas por chamada (bom equilíbrio diversidade x custo)

    # 1) Falas single nas 9 NRs do eletricista (distribuição uniforme)
    _spread_single(batches, ELECTRICIAN_NRS, n_elec, per_call)
    # 2) Falas single nas demais 27 NRs
    others = [c for c in CLASSES if c not in ELECTRICIAN_NRS]
    _spread_single(batches, others, n_other, per_call)
    # 3) Ambíguas (rótulo duplo)
    _spread_ambiguous(batches, n_amb, per_call)
    # 4) Fora de escopo
    _spread_none(batches, n_none, per_call)

    random.shuffle(batches)
    return batches


def _spread_single(batches: List[Dict[str, Any]], nrs: List[str], n_total: int, per_call: int) -> None:
    if n_total <= 0 or not nrs:
        return
    base = n_total // len(nrs)
    rem = n_total % len(nrs)
    for i, nr in enumerate(nrs):
        count = base + (1 if i < rem else 0)
        while count > 0:
            k = min(per_call, count)
            batches.append({"kind": "single", "nr": nr, "n": k})
            count -= k


def _spread_ambiguous(batches: List[Dict[str, Any]], n_total: int, per_call: int) -> None:
    if n_total <= 0:
        return
    base = n_total // len(AMBIGUOUS_PAIRS)
    rem = n_total % len(AMBIGUOUS_PAIRS)
    for i, (a, b) in enumerate(AMBIGUOUS_PAIRS):
        count = base + (1 if i < rem else 0)
        while count > 0:
            k = min(per_call, count)
            batches.append({"kind": "ambiguous", "a": a, "b": b, "n": k})
            count -= k


def _spread_none(batches: List[Dict[str, Any]], n_total: int, per_call: int) -> None:
    count = n_total
    while count > 0:
        k = min(per_call, count)
        batches.append({"kind": "none", "n": k})
        count -= k


def run_batch(batch: Dict[str, Any], base_url: str, model: str) -> List[Dict[str, Any]]:
    """Executa um lote e retorna registros rotulados prontos para JSONL."""
    kind = batch["kind"]
    if kind == "single":
        prompt = build_user_prompt_single(batch["nr"], batch["n"])
        labels = [batch["nr"]]
        prompt_kind = f"single:{batch['nr']}"
    elif kind == "ambiguous":
        prompt = build_user_prompt_ambiguous(batch["a"], batch["b"], batch["n"])
        labels = [batch["a"], batch["b"]]
        prompt_kind = f"ambiguous:{batch['a']}+{batch['b']}"
    elif kind == "none":
        prompt = build_user_prompt_none(batch["n"])
        labels = [NONE_LABEL]
        prompt_kind = "none"
    else:
        return []

    raw = call_vllm(base_url, model, prompt)
    utterances = extract_string_array(raw)
    records: List[Dict[str, Any]] = []
    for utt in utterances:
        # Descarta falas que vazam o número da norma (contaminação de vocabulário)
        if kind != "none" and re.search(r"\bnr[\s-]?\d", utt.lower()):
            continue
        records.append(
            {
                "text": utt,
                "labels": labels,
                "primary": labels[0],
                "kind": kind,
                "prompt_kind": prompt_kind,
                "generator": model,
            }
        )
    return records


def load_existing(out_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """Carrega registros já gerados (retomada) e um set de textos para deduplicação."""
    records: List[Dict[str, Any]] = []
    seen: set = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            records.append(r)
            seen.add(r["text"].strip().lower())
    return records, seen


def main() -> None:
    ap = argparse.ArgumentParser(description="FASE A: gera dataset rotulado de falas -> NR.")
    ap.add_argument("--target", type=int, default=2000, help="Nº total de falas desejadas.")
    ap.add_argument("--out", type=str, default="classifier/data/labels.jsonl")
    ap.add_argument("--base-url", type=str, default=DEFAULT_VLLM_URL)
    ap.add_argument("--model", type=str, default=DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-only", action="store_true", help="Só imprime o plano de lotes.")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batches = plan_batches(args.target)
    planned = sum(b["n"] for b in batches)
    logger.info("Plano: %d lotes, %d falas planejadas (alvo %d)", len(batches), planned, args.target)
    kinds = {}
    for b in batches:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + b["n"]
    logger.info("Distribuição: %s", kinds)
    if args.plan_only:
        return

    existing, seen = load_existing(out_path)
    logger.info("Registros existentes: %d (retomada)", len(existing))
    have = len(existing)

    fout = out_path.open("a", encoding="utf-8")
    written_since_flush = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_batch, b, args.base_url, args.model): b for b in batches}
            for i, fut in enumerate(as_completed(futs)):
                try:
                    recs = fut.result()
                except Exception as e:
                    logger.warning("Lote falhou (%s): %s", futs[fut], e)
                    continue
                for r in recs:
                    key = r["text"].strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    have += 1
                    written_since_flush += 1
                if written_since_flush >= 20:
                    fout.flush()
                    written_since_flush = 0
                    logger.info("Progresso: %d falas únicas | %d/%d lotes | %.0fs",
                                have, i + 1, len(batches), time.time() - t0)
    finally:
        fout.flush()
        fout.close()

    logger.info("Concluído: %d falas únicas em %s (%.0fs)", have, out_path, time.time() - t0)


if __name__ == "__main__":
    main()
