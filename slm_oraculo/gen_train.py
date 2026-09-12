#!/usr/bin/env python3
"""
gen_train.py — Dataset SFT de DESTILAÇÃO: estilo ACIONÁVEL do 27B, filtrado pela RÉGUA.

Desenho (ordem do capitão):
  1. Para cada chunk-fonte elegível (fora do holdout, fora das NRs reservadas), o 27B gera
     uma PERGUNTA coloquial de eletricista + a lista de FATOS OBRIGATÓRIOS do trecho
     (mesma ideia do bench/regua/build_gabarito, agora sobre o pool de treino).
  2. O 27B, em ORÁCULO (recebe o trecho-fonte), gera a RESPOSTA no estilo acionável
     (prompt `numeros`): ação concreta + número quando existir + citação inline.
  3. FILTRO PELA PRÓPRIA RÉGUA: só entra no SFT o par cuja resposta do 27B for APROVADA
     pela régua honesta (cobertura>=0.5, sem tautologia, SEM ALUCINAÇÃO). É a mesma régua
     que mede o teto — garante que o alvo do treino é resposta que a régua chancela.

MURALHAS anti-contaminação (reusa finetune2/common):
  - MURALHA 1: chunk-ouro do holdout nunca é fonte nem distrator;
  - MURALHA 2: NR-33/16/26 reservadas (OOD estrutural), 0 chunks no treino;
  - MURALHA 3: filtro lexical Jaccard<0.4 da PERGUNTA vs holdout inteiro.

Todo 27B roda no servidor do juiz (nenhum token novo na Spark). Paralelo, checkpoint
incremental, procedência por linha. Comentários PT-BR; código em inglês.

Uso (classifier/.venv; juiz default 10.100.0.111:8005):
  ../classifier/.venv/bin/python gen_train.py --per-chunk 2 --out data/train_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Import por caminho de arquivo p/ evitar o choque de nome entre finetune2/common.py e
# bench/regua/common.py (ambos se chamam "common").
import importlib.util  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ft = _load_module("ft2_common", _ROOT / "finetune2" / "common.py")
DEFAULT_VLLM_MODEL = _ft.DEFAULT_VLLM_MODEL
DEFAULT_VLLM_URL = _ft.DEFAULT_VLLM_URL
RESERVED_NRS = _ft.RESERVED_NRS
SYNTHESIS_SYSTEM_PROMPT = _ft.SYNTHESIS_SYSTEM_PROMPT
LexicalFilter = _ft.LexicalFilter
call_vllm = _ft.call_vllm
extract_json = _ft.extract_json
holdout_gold_chunk_ids = _ft.holdout_gold_chunk_ids
load_chunks = _ft.load_chunks

sys.path.insert(0, str(_ROOT / "bench" / "regua"))
sys.path.insert(0, str(_ROOT / "bench" / "prompt_teto"))
from prompts import NUMEROS  # noqa: E402 (prompt de síntese acionável)
from ruler import deterministic_pass  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402

# Prompt de geração de PERGUNTA + FATOS a partir do trecho.
QGEN_SYSTEM = (
    "Você prepara dados de treino para um assistente de segurança do trabalho que responde "
    "eletricistas por voz. Recebe UM trecho de Norma Regulamentadora. Produza um JSON com:\n"
    "1. \"pergunta\": pergunta CURTA e coloquial, como um eletricista brasileiro leigo "
    "perguntaria em voz no serviço — linguagem do dia a dia, SEM os termos técnicos do "
    "trecho, SEM citar número de item nem nome da norma.\n"
    "2. \"fatos\": lista de 2 a 5 FATOS OBRIGATÓRIOS e atômicos que qualquer resposta correta "
    "à pergunta precisa conter, extraídos SÓ do trecho (valores, ações, condições).\n"
    "Responda SOMENTE o objeto JSON."
)
QGEN_USER = ("TRECHO ({doc}, item {sec} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\n"
             "Gere o JSON com \"pergunta\" (do jeito do eletricista) e \"fatos\" (2 a 5, atômicos).")


def format_context_single(c: Dict[str, Any]) -> str:
    """Contexto de síntese igual ao AskPipeline (1 trecho-ouro)."""
    return f"[1] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}"


_lock = Lock()


def gen_one(chunk: Dict[str, Any], seed_tag: int, url: str, model: str,
            lf: LexicalFilter, threshold: float) -> Optional[Dict[str, Any]]:
    """Gera (pergunta, fatos) -> resposta 27B em oráculo -> filtra pela régua."""
    # 1) pergunta + fatos
    try:
        raw = call_vllm([{"role": "system", "content": QGEN_SYSTEM},
                         {"role": "user", "content": QGEN_USER.format(
                             doc=chunk["doc"], sec=chunk["section"],
                             title=chunk["title"], text=chunk["text"][:1800])}],
                        url=url, model=model, temperature=0.8, top_p=0.95,
                        max_tokens=512, thinking=False)
    except Exception:
        return None
    obj = extract_json(raw)
    if not obj or "pergunta" not in obj or "fatos" not in obj:
        return None
    question = " ".join(str(obj["pergunta"]).strip().strip('"').split())
    facts = [str(f).strip() for f in obj["fatos"] if str(f).strip()]
    if len(question) < 8 or len(facts) < 2:
        return None
    # MURALHA 3: filtro lexical da pergunta vs holdout.
    with _lock:
        ok = lf.accept(question, meta={"src_chunk_id": chunk["id"], "doc": chunk["doc"]})
    if not ok:
        return None

    # 2) resposta do 27B em ORÁCULO, estilo acionável (numeros).
    user = (f"Contexto normativo consultado:\n{format_context_single(chunk)}\n\n"
            f"Pergunta do eletricista:\n{question}")
    try:
        answer = call_vllm([{"role": "system", "content": NUMEROS},
                            {"role": "user", "content": user}],
                           url=url, model=model, temperature=0.0,
                           max_tokens=384, thinking=False)
    except Exception:
        return None
    answer = answer.strip()
    if len(answer) < 20:
        return None

    # 3) FILTRO PELA RÉGUA (a mesma que mede o teto).
    r = deterministic_pass(question, answer, facts, threshold)
    if r.needs_judge or (r.approved and not r.empty):
        try:
            jr = judge_disambiguate(question, answer, r.facts_ambiguous, chunk["text"])
            confirmed = set()
            amb_low = {a.lower(): a for a in r.facts_ambiguous}
            for fp in jr.get("fatos_presentes", []):
                key = fp.strip().lower()
                if key in amb_low:
                    confirmed.add(amb_low[key])
                else:
                    for al, orig in amb_low.items():
                        if al.startswith(key[:12]) or key.startswith(al[:12]):
                            confirmed.add(orig)
            r.facts_present = list(set(r.facts_present) | confirmed)
            r.hallucination = bool(jr.get("alucinacao", False))
        except Exception:
            pass
    coverage = len(r.facts_present) / max(1, len(facts))
    approved = coverage >= threshold and not r.tautology and not r.hallucination and not r.empty
    if not approved:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "src_chunk_id": chunk["id"], "src_doc": chunk["doc"], "src_section": chunk["section"],
            "question": question, "facts": facts, "coverage": round(coverage, 3),
            "seed_tag": seed_tag, "gen_model": model,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", type=int, default=2, help="tentativas de par por chunk")
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--url", default=DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--target", type=int, default=0, help="para cedo ao atingir N aprovados (0=todos)")
    ap.add_argument("--out", default=str(_HERE / "data" / "train_sft.jsonl"))
    ap.add_argument("--discards", default=str(_HERE / "data" / "train_lexical_discards.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks()
    gold = holdout_gold_chunk_ids()
    eligible = [c for c in chunks if c["id"] not in gold
                and c["doc"] not in RESERVED_NRS and len(c["text"]) >= args.min_len]
    rng.shuffle(eligible)
    # Prioriza seções numéricas (Q&A de campo melhores que anexos de despejo).
    _num = re.compile(r"^\d+(?:\.\d+)*$")
    eligible.sort(key=lambda c: 0 if _num.match(str(c["section"]).strip()) else 1)

    tasks: List[tuple] = []
    for c in eligible:
        for k in range(args.per_chunk):
            tasks.append((c, k))
    print(f"chunks elegíveis: {len(eligible)} | tarefas (per-chunk={args.per_chunk}): {len(tasks)}",
          flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                try:
                    m = json.loads(line)["meta"]
                    done_keys.add((m["src_chunk_id"], m.get("seed_tag", 0)))
                except Exception:
                    pass
        print(f"checkpoint: {len(done_keys)} pares já existentes", flush=True)
    else:
        out_path.write_text("# " + json.dumps({
            "artifact": "sft_distill_regua_filtered", "task": "poc-slm-oraculo",
            "model": args.model, "threshold": args.threshold, "reserved_nrs": RESERVED_NRS,
            "created_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n",
            encoding="utf-8")

    lf = LexicalFilter()
    pending = [t for t in tasks if (t[0]["id"], t[1]) not in done_keys]
    fh = out_path.open("a", encoding="utf-8")
    kept = len(done_keys)
    tried = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, c, k, args.url, args.model, lf, args.threshold): (c["id"], k)
                for c, k in pending}
        for fut in as_completed(futs):
            tried += 1
            rec = None
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec is not None:
                with _lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    kept += 1
            if tried % 50 == 0:
                rate = tried / max(1e-9, time.time() - t0)
                print(f"  tried={tried}/{len(pending)} kept={kept} "
                      f"lex_disc={len(lf.discards)} {rate:.1f}/s", flush=True)
            if args.target and kept >= args.target:
                break
    fh.close()
    lf.dump_discards(Path(args.discards))
    print(f"SFT distill pronto: {kept} pares aprovados | descartes lexicais {len(lf.discards)} "
          f"| {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
