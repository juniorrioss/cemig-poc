#!/usr/bin/env python3
"""
gen_sft.py — Dataset SFT (FASE A) por inversão de chunks (síntese de campo do LFM2.5-2.6B).

Para cada chunk-fonte (1 correto), amostra 1 DISTRATOR plausível (mesma NR se houver, senão NR
próxima) e pede ao vLLM 27B:
  1. uma PERGUNTA coloquial de eletricista (linguagem de campo, sem termos técnicos do trecho);
  2. a RESPOSTA-OURO: 2–4 frases, prosa corrida, citação EXATA da norma+item, baseada SÓ no
     chunk correto (o distrator existe para ensinar o modelo a ignorar contexto irrelevante).

MURALHAS anti-contaminação aplicadas aqui:
  - MURALHA 1: chunk-ouro do holdout NUNCA é usado como fonte nem como distrator;
  - MURALHA 2: chunks das NRs reservadas (33/16/26) são excluídos (viram OOD estrutural);
  - MURALHA 3: cada PERGUNTA gerada passa pelo LexicalFilter (Jaccard<0.4 vs holdout);
               descartes registrados com procedência em data/sft_lexical_discards.json.

Saída: data/sft.jsonl — formato ChatML {messages:[system,user,assistant], meta:{...procedência}}.
Cada linha carrega procedência completa (chunk fonte, distrator, jaccard, modelo, timestamp).

Comentários em PT-BR; código em inglês.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (DATA_DIR, DEFAULT_VLLM_MODEL, DEFAULT_VLLM_URL, RESERVED_NRS,
                    SYNTHESIS_SYSTEM_PROMPT, LexicalFilter, call_vllm, extract_json,
                    holdout_gold_chunk_ids, load_chunks)

GEN_SYSTEM = (
    "Você gera dados de TREINO para um assistente de segurança do trabalho que responde "
    "eletricistas por voz. Recebe UM trecho correto de uma Norma Regulamentadora (a FONTE) e "
    "UM trecho distrator (irrelevante). Produza um JSON com dois campos:\n"
    "1. \"pergunta\": uma pergunta CURTA e coloquial, como um eletricista brasileiro leigo "
    "perguntaria em voz alta no serviço — com linguagem do dia a dia, SEM usar os termos "
    "técnicos do trecho e SEM citar número de item ou nome da norma.\n"
    "2. \"resposta\": a resposta técnica correta baseada EXCLUSIVAMENTE no trecho-FONTE, em "
    "português brasileiro, prosa corrida, NO MÁXIMO 4 frases, SEM markdown/listas, citando a "
    "norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...'). Ignore o "
    "trecho distrator; ele existe só para confundir.\n"
    "Responda SOMENTE o objeto JSON, sem preâmbulo."
)

USER_TEMPLATE = (
    "TRECHO-FONTE ({src_doc}, item {src_sec} — {src_title}):\n\"\"\"\n{src_text}\n\"\"\"\n\n"
    "TRECHO-DISTRATOR ({dis_doc}, item {dis_sec} — {dis_title}):\n\"\"\"\n{dis_text}\n\"\"\"\n\n"
    "Gere o JSON com \"pergunta\" (coloquial, do jeito do eletricista) e \"resposta\" "
    "(baseada só no TRECHO-FONTE, 2 a 4 frases, citando {src_doc_up}, item {src_sec})."
)

_NR_RE = re.compile(r"nr-?\d{1,2}", re.IGNORECASE)


def clean_line(s: str) -> str:
    return " ".join(s.strip().strip('"').split())


def pick_distractor(src: Dict[str, Any], pool: List[Dict[str, Any]],
                    same_nr: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Distrator plausível: preferir mesma NR (seção diferente); senão qualquer outro chunk."""
    cands = [c for c in same_nr if c["id"] != src["id"]]
    if cands:
        return rng.choice(cands)
    other = [c for c in pool if c["doc"] != src["doc"]]
    return rng.choice(other) if other else None


def valid_answer(resp: str, src_doc: str) -> bool:
    """A resposta-ouro deve citar a norma correta e ser prosa curta sem markdown."""
    if not resp or len(resp) < 20:
        return False
    if any(tok in resp for tok in ("- ", "* ", "#", "```", "\n1.", "\n2.")):
        return False
    # deve citar a NR-fonte (permite variações nr-10 / NR-10 / NR 10)
    doc_num = src_doc.replace("nr-", "").replace("nr", "").strip()
    low = resp.lower()
    if f"nr-{doc_num}" in low or f"nr {doc_num}" in low or f"nr{doc_num}" in low:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="alvo de pares SFT")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-len", type=int, default=200, help="tamanho mínimo do texto do chunk-fonte")
    ap.add_argument("--url", default=DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    ap.add_argument("--out", default=str(DATA_DIR / "sft.jsonl"))
    ap.add_argument("--discards", default=str(DATA_DIR / "sft_lexical_discards.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = sem limite (usa --n)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks()
    gold_ids = holdout_gold_chunk_ids()

    # MURALHA 1 + 2: exclui chunk-ouro do holdout e NRs reservadas do pool de FONTES.
    eligible = [c for c in chunks
                if c["id"] not in gold_ids
                and c["doc"] not in RESERVED_NRS
                and len(c["text"]) >= args.min_len]
    # Pool de distratores: também sem chunk-ouro do holdout (evita vazamento por distrator),
    # mas pode incluir qualquer NR não-reservada.
    distractor_pool = [c for c in chunks
                       if c["id"] not in gold_ids and c["doc"] not in RESERVED_NRS]
    by_nr: Dict[str, List[Dict[str, Any]]] = {}
    for c in distractor_pool:
        by_nr.setdefault(c["doc"], []).append(c)

    # Prioriza chunks com item numérico (ex.: 10.5.1) — geram Q&A de campo melhores que
    # tabelas de anexo; anexos entram como preenchimento até o alvo (mistura realista).
    _num_sec = re.compile(r"^\d+(?:\.\d+)*$")
    rng.shuffle(eligible)
    eligible.sort(key=lambda c: 0 if _num_sec.match(str(c["section"]).strip()) else 1)
    target = args.limit if args.limit else args.n
    n_numeric = sum(1 for c in eligible if _num_sec.match(str(c["section"]).strip()))
    print(f"chunks elegíveis (fonte): {len(eligible)} (numéricos={n_numeric}) | alvo: {target} pares",
          flush=True)

    lf = LexicalFilter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint incremental: reaproveita fontes já geradas.
    done_src: set = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                try:
                    done_src.add(json.loads(line)["meta"]["src_chunk_id"])
                except Exception:
                    pass
        print(f"checkpoint: {len(done_src)} pares já existentes", flush=True)

    if not out_path.exists():
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# " + json.dumps({
                "artifact": "sft_synthesis", "task": "poc-dpo-sintese", "phase": "A",
                "model": args.model, "endpoint": args.url, "seed": args.seed,
                "reserved_nrs": RESERVED_NRS, "created_utc": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    kept = len(done_src)
    tried = 0
    rej_answer = 0
    for src in eligible:
        if kept >= target:
            break
        if src["id"] in done_src:
            continue
        tried += 1
        dis = pick_distractor(src, distractor_pool, by_nr.get(src["doc"], []), rng)
        if dis is None:
            continue
        user = USER_TEMPLATE.format(
            src_doc=src["doc"], src_sec=src["section"], src_title=src["title"],
            src_text=src["text"][:1800], src_doc_up=src["doc"].upper(),
            dis_doc=dis["doc"], dis_sec=dis["section"], dis_title=dis["title"],
            dis_text=dis["text"][:1000])
        try:
            raw = call_vllm([{"role": "system", "content": GEN_SYSTEM},
                             {"role": "user", "content": user}],
                            url=args.url, model=args.model, temperature=0.7,
                            max_tokens=640, thinking=False)
        except Exception as e:  # noqa: BLE001
            print(f"  falha chunk {src['id']}: {e}", flush=True)
            continue
        obj = extract_json(raw)
        if not obj or "pergunta" not in obj or "resposta" not in obj:
            continue
        question = clean_line(str(obj["pergunta"]))
        answer = clean_line(str(obj["resposta"]))
        if len(question) < 8:
            continue
        if not valid_answer(answer, src["doc"]):
            rej_answer += 1
            continue
        # MURALHA 3: filtro lexical da PERGUNTA vs holdout.
        if not lf.accept(question, meta={"src_chunk_id": src["id"], "doc": src["doc"]}):
            continue

        user_content = (f"Contexto normativo consultado:\n"
                        f"[1] ({dis['doc']} - {dis['section']} - {dis['title']}):\n{dis['text']}\n\n"
                        f"[2] ({src['doc']} - {src['section']} - {src['title']}):\n{src['text']}\n\n"
                        f"Pergunta do eletricista:\n{question}")
        rec = {
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "src_chunk_id": src["id"], "src_doc": src["doc"], "src_section": src["section"],
                "distractor_chunk_id": dis["id"], "distractor_doc": dis["doc"],
                "question": question,
                "max_jaccard_holdout": round(lf.max_overlap(question)[0], 4),
                "gen_model": args.model, "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        kept += 1
        if kept % 25 == 0:
            rate = kept / max(1e-9, (time.time() - t0))
            print(f"  kept={kept}/{target} tried={tried} rej_ans={rej_answer} "
                  f"lex_disc={len(lf.discards)} {rate:.2f}/s", flush=True)

    fh.close()
    lf.dump_discards(Path(args.discards))
    print(f"SFT pronto: {kept} pares | descartes lexicais: {len(lf.discards)} "
          f"| rej_resposta: {rej_answer} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
