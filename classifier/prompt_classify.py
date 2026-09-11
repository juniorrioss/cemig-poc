#!/usr/bin/env python3
"""
prompt_classify.py — FASE B (LLM residente): classificação por prompt no LFM2.5-1.2B.

Ceticismo do capitão sobre 'responda só o número': MEDIR DE VERDADE o parsing e a
obediência, não assumir. Testa 3 formatos de saída no MESMO holdout real (151+20):
  - number  : "responda apenas o número da NR" (ex.: 10)
  - nrcode  : "responda no formato NR-XX" (ex.: NR-10)
  - json    : 'responda {"nr": XX}'

Para cada formato reporta:
  - acurácia top-1 (via parser robusto + normalize_nr);
  - taxa de saída NÃO-PARSEÁVEL (nenhuma norma extraível);
  - taxa de ALUCINAÇÃO (norma citada que não existe no corpus: nr-02, nr-27, nr-99...);
  - latência real por consulta (llama.cpp na 5070) e tokens gerados;
  - estimativa de latência no S24+ pela tabela do device-bench.

O prompt lista as 36 NRs + descrição de 1 linha (build_prompt_catalog).

Uso:
    python3 classifier/prompt_classify.py --base-url http://127.0.0.1:8090/v1 \
        --json-out classifier/data/prompt_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from data_utils import load_holdout
from nr_taxonomy import CLASSES, NR_TITLES, build_prompt_catalog, normalize_nr

_HERE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prompt_classify")

CATALOG = build_prompt_catalog()

BASE_SYSTEM = (
    "Você é um classificador de Normas Regulamentadoras (NRs) de segurança do trabalho.\n"
    "Dada a fala de um trabalhador de campo, identifique QUAL norma da lista abaixo trata do assunto.\n\n"
    "Lista de normas disponíveis:\n"
    f"{CATALOG}\n\n"
    "Se a fala não tratar de nenhuma dessas normas, responda 'nenhuma'.\n"
)

FORMAT_INSTRUCTION = {
    "number": "Responda APENAS com o número da norma (ex.: 10). Nada além do número.",
    "nrcode": "Responda APENAS no formato NR-XX (ex.: NR-10). Nada além disso.",
    "json": 'Responda APENAS com um JSON: {"nr": 10}. Nada além do JSON.',
}


def parse_output(raw: str, fmt: str) -> Tuple[Optional[str], bool]:
    """Extrai a norma da saída. Retorna (norma_normalizada|None, hallucinated).

    hallucinated=True quando o modelo cita um número de NR que NÃO existe no corpus
    (ex.: nr-02, nr-27, nr-99) — sinal de alucinação de norma inexistente.
    """
    if raw is None:
        return None, False
    text = raw.strip()
    low = text.lower()

    # 'nenhuma' explícito
    if re.search(r"\bnenhuma\b|\bnenhum\b|\bnone\b", low):
        return "nenhuma", False

    # Tenta JSON primeiro se for o formato pedido
    if fmt == "json":
        m = re.search(r'"nr"\s*:\s*"?(\d{1,2})"?', low)
        if m:
            return _resolve(int(m.group(1)))

    # Formato NR-XX
    m = re.search(r"\bnr[\s._-]*(\d{1,2})\b", low)
    if m:
        return _resolve(int(m.group(1)))

    # Número puro (primeiro inteiro 1-38 plausível)
    m = re.search(r"\b(\d{1,2})\b", low)
    if m:
        return _resolve(int(m.group(1)))

    return None, False


def _resolve(num: int) -> Tuple[Optional[str], bool]:
    code = f"nr-{num:02d}"
    if code in NR_TITLES:
        return code, False
    # Número plausível mas norma inexistente no corpus -> alucinação
    if 1 <= num <= 99:
        return None, True
    return None, False


def call_classify(base_url: str, model: str, question: str, fmt: str, timeout: float = 60.0) -> Tuple[str, int, float]:
    """Chama o llama-server. Retorna (conteúdo, tokens_gerados, latência_s)."""
    system = BASE_SYSTEM + FORMAT_INSTRUCTION[fmt]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Fala do trabalhador: {question}"},
        ],
        "temperature": 0.0,
        "max_tokens": 24,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "") or ""
    ntok = int(data.get("usage", {}).get("completion_tokens", 0))
    return content.strip(), ntok, dt


def eval_format(base_url: str, model: str, holdout: List[Dict[str, Any]], fmt: str) -> Dict[str, Any]:
    """Avalia um formato de saída no holdout completo."""
    golds = [h["gold"] for h in holdout]
    top1 = 0
    unparseable = 0
    hallucinated = 0
    lat_list: List[float] = []
    tok_list: List[int] = []
    samples: List[Dict[str, Any]] = []

    for i, h in enumerate(holdout):
        try:
            content, ntok, dt = call_classify(base_url, model, h["text"], fmt)
        except Exception as e:
            logger.warning("Falha em %s (%s): %s", h["id"], fmt, e)
            unparseable += 1
            continue
        pred, hall = parse_output(content, fmt)
        lat_list.append(dt)
        tok_list.append(ntok)
        if hall:
            hallucinated += 1
        if pred is None:
            unparseable += 1
        elif pred == h["gold"]:
            top1 += 1
        if i < 6:
            samples.append({"q": h["text"][:70], "gold": h["gold"], "raw": content[:40], "pred": pred})

    n = len(holdout)
    import statistics
    return {
        "format": fmt,
        "top1_acc": round(100 * top1 / n, 2),
        "unparseable_rate": round(100 * unparseable / n, 2),
        "hallucination_rate": round(100 * hallucinated / n, 2),
        "lat_ms_median_5070": round(statistics.median(lat_list) * 1000, 1) if lat_list else None,
        "lat_ms_p90_5070": round(statistics.quantiles(lat_list, n=10)[-1] * 1000, 1) if len(lat_list) > 10 else None,
        "tokens_median": int(statistics.median(tok_list)) if tok_list else None,
        "tokens_max": max(tok_list) if tok_list else None,
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FASE B: classificação por prompt no LFM2.5-1.2B residente.")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", type=str, default="local")
    ap.add_argument("--formats", type=str, default="number,nrcode,json")
    ap.add_argument("--json-out", type=str, default=str(_HERE / "data" / "prompt_results.json"))
    ap.add_argument("--limit", type=int, default=0, help="Limita nº de perguntas (smoke).")
    args = ap.parse_args()

    holdout = load_holdout()
    if args.limit:
        holdout = holdout[: args.limit]
    logger.info("Holdout: %d perguntas | modelo residente LFM2.5-1.2B (QAD Q4_0)", len(holdout))

    results = []
    for fmt in args.formats.split(","):
        fmt = fmt.strip()
        if not fmt:
            continue
        logger.info("Avaliando formato '%s'...", fmt)
        r = eval_format(args.base_url, args.model, holdout, fmt)
        results.append(r)
        logger.info(
            "  %-7s top1=%.1f%% naoparse=%.1f%% alucin=%.1f%% lat_med=%sms tok_med=%s",
            fmt, r["top1_acc"], r["unparseable_rate"], r["hallucination_rate"],
            r["lat_ms_median_5070"], r["tokens_median"],
        )

    print("\n" + "=" * 96)
    print(f"  FASE B — PROMPT-CLASSIFY LFM2.5-1.2B (holdout {len(holdout)}) — ceticismo 'só o número'")
    print("=" * 96)
    print(f"| {'Formato':<8} | {'Top-1':>6} | {'NãoParse':>8} | {'Alucin':>7} | {'Lat med(ms)':>11} | {'Tok med':>7} |")
    print("|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 13 + "|" + "-" * 9 + "|")
    for r in results:
        print(f"| {r['format']:<8} | {r['top1_acc']:>5.1f}% | {r['unparseable_rate']:>7.1f}% | "
              f"{r['hallucination_rate']:>6.1f}% | {str(r['lat_ms_median_5070']):>11} | {str(r['tokens_median']):>7} |")
    print("=" * 96)

    out = {"n_holdout": len(holdout), "model": "LFM2.5-1.2B-Instruct-QAD-Q4_0", "results": results}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Métricas salvas em %s", args.json_out)


if __name__ == "__main__":
    main()
