#!/usr/bin/env python3
"""
expand_docs.py — Etapa 1: expansão de documento (offline) via vLLM do capitão.

Para CADA um dos 2202 chunks do índice de produção, o LLM (Qwen3.8-27B-FP8) gera:
  - 3-5 falas coloquiais de operário/eletricista que aquele chunk responderia;
  - sinônimos leigos/regionais dos termos técnicos presentes no chunk.

O objetivo é fechar o ABISMO LEXICAL (operário: "cinto folgado no poste" vs norma:
"talabarte, ancoragem") indexando a expansão num campo FTS5 próprio com peso calibrável.

ANTI-CONTAMINAÇÃO (crítico, ordem do capitão):
  - o prompt é GENÉRICO por chunk; NUNCA vê as 151 perguntas do holdout;
  - a geração parte só do texto normativo do chunk (doc/section/title/text).

HIGIENE:
  - inferência não-interativa com timeout por chamada;
  - checkpoint incremental em JSONL (append) — a VPN oscila;
  - procedência (modelo, endpoint, timestamp, prompt-hash) gravada no cabeçalho.

Uso:
    python3 retrieval3/expand_docs.py --db corpus/index_hf_36nr.db \
        --out retrieval3/data/expansions.jsonl --workers 6
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sqlite3
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent

# Prompt GENÉRICO por chunk — jamais referencia perguntas do holdout.
SYSTEM_PROMPT = (
    "Você é um instrutor de segurança do trabalho que traduz o texto formal das Normas "
    "Regulamentadoras (NRs) para a linguagem REAL de campo de eletricistas, operários e "
    "trabalhadores brasileiros (incluindo gírias e termos regionais).\n"
    "Dado um TRECHO de uma NR, sua tarefa é gerar material de INDEXAÇÃO para busca, em duas partes:\n"
    "1. PERGUNTAS: de 4 a 6 perguntas curtas, coloquiais e diretas que um trabalhador leigo faria "
    "em campo (voz), e cuja resposta esteja NESTE trecho. Use a fala do dia a dia, não o juridiquês. "
    "Ex.: 'posso trabalhar sem desligar a energia?', 'o cinto tá folgado, e agora?'.\n"
    "2. SINONIMOS: liste os termos técnicos do trecho seguidos de seus equivalentes leigos/gírias. "
    "Ex.: 'talabarte = cinto/corda de segurança; ancoragem = ponto de amarrar; desenergização = desligar a energia'.\n"
    "Regras: seja fiel ao trecho, não invente obrigações que não estão nele, não cite números de itens, "
    "responda SOMENTE em JSON válido no formato: "
    '{"perguntas": ["...", "..."], "sinonimos": ["termo = leigo", "..."]}'
)

USER_TEMPLATE = (
    "TRECHO da {doc_upper} (seção {section} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Gere o JSON de indexação (perguntas coloquiais + sinônimos leigos)."
)

_PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + USER_TEMPLATE).encode()).hexdigest()[:12]

_write_lock = threading.Lock()


def call_vllm(base_url: str, model: str, system: str, user: str,
              max_tokens: int = 512, timeout: int = 120) -> str:
    """Chamada não-interativa ao endpoint OpenAI-compatível; thinking desligado."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def parse_expansion(raw: str) -> Optional[Dict[str, List[str]]]:
    """Extrai o JSON {perguntas, sinonimos} da resposta do LLM (tolerante a cercas)."""
    if not raw:
        return None
    txt = raw.strip()
    # Remove cercas markdown
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1] if "```" in txt[3:] else txt.strip("`")
        if txt.lstrip().startswith("json"):
            txt = txt.lstrip()[4:]
    # Isola o primeiro objeto JSON
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(txt[start:end + 1])
    except json.JSONDecodeError:
        return None
    perg = obj.get("perguntas") or obj.get("questions") or []
    sino = obj.get("sinonimos") or obj.get("sinônimos") or obj.get("synonyms") or []
    if not isinstance(perg, list):
        perg = []
    if not isinstance(sino, list):
        sino = []
    perg = [str(p).strip() for p in perg if str(p).strip()]
    sino = [str(s).strip() for s in sino if str(s).strip()]
    if not perg and not sino:
        return None
    return {"perguntas": perg, "sinonimos": sino}


def load_done(out_path: Path) -> set:
    """IDs já processados (checkpoint incremental)."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_chunks(db_path: str) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, doc, section, title, text FROM chunks ORDER BY id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(_HERE.parent / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--out", default=str(_HERE / "data" / "expansions.jsonl"))
    ap.add_argument("--base-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="0 = todos; >0 = smoke")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.db)
    if args.limit:
        chunks = chunks[:args.limit]
    done = load_done(out_path)
    todo = [c for c in chunks if c["id"] not in done]
    print(f"Chunks: {len(chunks)} | já feitos: {len(done)} | a fazer: {len(todo)}")

    # Cabeçalho de procedência (só uma vez, se arquivo novo)
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# " + json.dumps({
                "artifact": "document_expansions",
                "task": "poc-retrieval-v3",
                "model": args.model,
                "endpoint": args.base_url,
                "prompt_hash": _PROMPT_HASH,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "db": Path(args.db).name,
            }, ensure_ascii=False) + "\n")

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    ok = fail = 0

    def work(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user = USER_TEMPLATE.format(
            doc_upper=c["doc"].upper(), section=c["section"],
            title=c["title"], text=c["text"][:2000],
        )
        for attempt in range(3):
            try:
                raw = call_vllm(args.base_url, args.model, SYSTEM_PROMPT, user,
                                max_tokens=args.max_tokens, timeout=args.timeout)
                exp = parse_expansion(raw)
                if exp:
                    return {"id": c["id"], "doc": c["doc"], "section": c["section"], **exp}
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            if rec:
                with _write_lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                el = time.time() - t0
                rate = i / el
                eta = (len(todo) - i) / rate if rate else 0
                print(f"  {i}/{len(todo)} ok={ok} fail={fail} "
                      f"{rate:.1f}/s ETA {eta/60:.1f}min", flush=True)

    fh.close()
    print(f"Concluído: ok={ok} fail={fail} em {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
