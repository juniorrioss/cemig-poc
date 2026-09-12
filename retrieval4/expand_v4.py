#!/usr/bin/env python3
"""
expand_v4.py — FRENTE B: expansão de documento MAIS RICA (aprofundar, não redescobrir).

A v3 já gera 4-6 perguntas coloquiais + sinônimos leigos por chunk (maior alavanca da POC).
A v4 APROFUNDA a mesma ideia, com material NOVO e complementar por chunk:
  - "verbalizacoes": +6-8 formas ADICIONAIS de o operário falar do mesmo assunto,
     variando registro (pergunta indireta, ordem no rádio, desabafo, forma truncada);
  - "asr": variantes com ERRO típico de transcrição (Whisper Base) — troca fonética
     ('talabarte'->'talabart', 'desenergizar'->'des energizar'), aglutinação, número por extenso;
  - "sinonimos2": sinônimos/gírias regionais ADICIONAIS além dos da v3.

Fica num campo FTS5 próprio (`expansion2`) para medir o INCREMENTO isolado sobre a v3.

ANTI-CONTAMINAÇÃO (TRAVA 1): prompt GENÉRICO por chunk; jamais vê o holdout; parte só do
texto normativo. HIGIENE: não-interativo, timeout, retry, checkpoint incremental, paralelo
(ThreadPoolExecutor), procedência no cabeçalho.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python expand_v4.py --db ../corpus/index_hf_36nr.db \
     --out data/expansions_v4.jsonl --workers 8

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
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

SYSTEM_PROMPT = (
    "Você é um instrutor de segurança do trabalho que prepara MATERIAL DE BUSCA para um app "
    "offline usado por eletricistas e operários brasileiros em campo. Dado um TRECHO de uma "
    "Norma Regulamentadora, você gera texto que ajuda a casar a FALA REAL do trabalhador "
    "(por voz, transcrita) com este trecho. Gere três blocos:\n"
    "1. VERBALIZACOES: 6 a 8 formas DIFERENTES de um trabalhador leigo abordar o assunto deste "
    "trecho, variando o registro: pergunta direta, pergunta indireta (rodeia o assunto), ordem "
    "curta no rádio, desabafo/reclamação, frase truncada. Fala do dia a dia, sem juridiquês.\n"
    "2. ASR: 4 a 6 variantes das mesmas ideias JÁ COM ERRO de transcrição automática de voz "
    "(troca fonética, palavra colada, acento errado, número por extenso). "
    "Ex.: 'talabarte'->'talabart', 'desenergizar'->'des energizar', 'NR dez'.\n"
    "3. SINONIMOS2: sinônimos e gírias regionais ADICIONAIS dos termos técnicos do trecho.\n"
    "Regras: seja fiel ao trecho, não invente obrigação inexistente, não cite número de item, "
    "não repita exemplos óbvios. Responda SOMENTE JSON válido: "
    '{"verbalizacoes": ["..."], "asr": ["..."], "sinonimos2": ["termo = leigo"]}'
)

USER_TEMPLATE = (
    "TRECHO da {doc_upper} (seção {section} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Gere o JSON de indexação (verbalizacoes + asr + sinonimos2)."
)

_PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + USER_TEMPLATE).encode()).hexdigest()[:12]
_write_lock = threading.Lock()


def call_vllm(base_url: str, model: str, system: str, user: str,
              max_tokens: int = 640, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.5, "top_p": 0.95, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def parse_expansion(raw: str) -> Optional[Dict[str, List[str]]]:
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1] if "```" in txt[3:] else txt.strip("`")
        if txt.lstrip().startswith("json"):
            txt = txt.lstrip()[4:]
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        obj = json.loads(txt[s:e + 1])
    except json.JSONDecodeError:
        return None
    def lst(*keys):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        return []
    verb = lst("verbalizacoes", "verbalizações", "verbalizations")
    asr = lst("asr", "asr_variants", "variantes")
    sin = lst("sinonimos2", "sinônimos2", "sinonimos", "synonyms")
    if not verb and not asr and not sin:
        return None
    return {"verbalizacoes": verb, "asr": asr, "sinonimos2": sin}


def load_done(out_path: Path) -> set:
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
    ap.add_argument("--out", default=str(_HERE / "data" / "expansions_v4.jsonl"))
    ap.add_argument("--base-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.db)
    if args.limit:
        chunks = chunks[:args.limit]
    done = load_done(out_path)
    todo = [c for c in chunks if c["id"] not in done]
    print(f"Chunks: {len(chunks)} | já feitos: {len(done)} | a fazer: {len(todo)}")

    if not out_path.exists():
        out_path.write_text("# " + json.dumps({
            "artifact": "document_expansions_v4", "task": "poc-retrieval-v4",
            "model": args.model, "endpoint": args.base_url, "prompt_hash": _PROMPT_HASH,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "db": Path(args.db).name,
        }, ensure_ascii=False) + "\n", encoding="utf-8")

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    ok = fail = 0

    def work(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user = USER_TEMPLATE.format(doc_upper=c["doc"].upper(), section=c["section"],
                                    title=c["title"], text=c["text"][:2000])
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
            if i % 50 == 0:
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate if rate else 0
                print(f"  {i}/{len(todo)} ok={ok} fail={fail} {rate:.1f}/s ETA {eta/60:.1f}min", flush=True)
    fh.close()
    print(f"Concluído: ok={ok} fail={fail} em {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
