#!/usr/bin/env python3
"""
gen_data.py — PARTE 1 do SFT v2: fabrica os dados em 4 FAMÍLIAS (o coração do v2).

Todas as famílias têm o MESMO formato conversacional {"messages": [system, user, assistant]}
com o SYNTHESIS_SYSTEM_PROMPT_V2 (que PERMITE a recusa útil). Cada par é FILTRADO por régua:
só entra o que o 27B produziu E foi aprovado no critério da família.

  (a) ORACULO (~50%): chunk-ouro presente; alvo = resposta acionável do 27B (prompt numeros),
      APROVADA pela régua honesta (cobertura>=0.5, sem tautologia, SEM alucinação). == v1.
  (b) RECUSA (~20%): pergunta emparelhada com chunk de OUTRA norma; alvo = recusa honesta e
      ÚTIL (diz que não está na norma consultada + o que faltou). Aprovação = juiz de recusa.
  (c) PARCIAL (~20%): chunk da norma certa com o dado-chave REMOVIDO; alvo = responder o que
      dá + declarar o que não dá, sem inventar. Aprovação = juiz de recusa (parcial).
  (d) DISTRATOR (~10%): chunk da MESMA norma, seção vizinha que NÃO responde (caso camiseta);
      alvo = não atribuir o item errado; recusar/delimitar. Aprovação = juiz de recusa.

MURALHAS anti-contaminação (reuso finetune2/common):
  1. chunk-ouro do holdout nunca é fonte nem distrator (via eligible_chunks);
  2. NR-33/16/26 reservadas (0 chunks — OOD estrutural);
  3. filtro lexical Jaccard<0.4 da PERGUNTA vs holdout inteiro.
Split adicional (lição do v1): chunks de VAL/refusal-test ficam FORA do treino (split_chunks).

Higiene: paralelo (>=16 workers), checkpoint incremental por linha, procedência, não-interativo.
Comentários PT-BR; código em inglês.

Uso (classifier/.venv; juiz 27B default 10.100.0.111:8005):
  ../classifier/.venv/bin/python gen_data.py --target-oracle 1600 --target-recusa 640 \
      --target-parcial 640 --target-distrator 320 --out data/train_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import common_v2 as C

_HERE = Path(__file__).resolve().parent
_lock = Lock()


def gen_qa(chunk: Dict[str, Any], url: str, model: str) -> Optional[Tuple[str, List[str]]]:
    """Gera (pergunta coloquial, fatos obrigatórios) a partir do trecho (professor 27B)."""
    try:
        raw = C.call_vllm([{"role": "system", "content": C.QGEN_SYSTEM},
                           {"role": "user", "content": C.QGEN_USER.format(
                               doc=chunk["doc"], sec=chunk["section"],
                               title=chunk["title"], text=chunk["text"][:1800])}],
                          url=url, model=model, temperature=0.8, top_p=0.95,
                          max_tokens=512, thinking=False)
    except Exception:
        return None
    obj = C.extract_json(raw)
    if not obj or "pergunta" not in obj or "fatos" not in obj:
        return None
    question = " ".join(str(obj["pergunta"]).strip().strip('"').split())
    facts = [str(f).strip() for f in obj["fatos"] if str(f).strip()]
    if len(question) < 8 or len(facts) < 2:
        return None
    return question, facts


def teach(system: str, context: str, question: str, url: str, model: str,
          max_tokens: int, temperature: float) -> Optional[str]:
    """Professor 27B gera o alvo (resposta ideal) para o (contexto, pergunta)."""
    user = C.TEACHER_USER.format(context=context, question=question) \
        if system in (C.TEACHER_RECUSA, C.TEACHER_PARCIAL, C.TEACHER_DISTRATOR) \
        else C.build_user(context, question)
    try:
        ans = C.call_vllm([{"role": "system", "content": system},
                           {"role": "user", "content": user}],
                          url=url, model=model, temperature=temperature,
                          max_tokens=max_tokens, thinking=False)
    except Exception:
        return None
    ans = ans.strip()
    return ans if len(ans) >= 20 else None


def make_example(system_train: str, context: str, question: str, answer: str,
                 family: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Monta o exemplo conversacional de treino (system v2 + user + assistant)."""
    return {
        "messages": [
            {"role": "system", "content": system_train},
            {"role": "user", "content": C.build_user(context, question)},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"family": family, **meta,
                 "created_utc": datetime.now(timezone.utc).isoformat()},
    }


def gen_oracle(chunk: Dict[str, Any], url: str, model: str, lf: C.LexicalFilter,
               threshold: float) -> Optional[Dict[str, Any]]:
    """Família (a): oráculo, alvo acionável do 27B FILTRADO pela régua honesta (== v1)."""
    qa = gen_qa(chunk, url, model)
    if not qa:
        return None
    question, facts = qa
    with _lock:
        if not lf.accept(question, meta={"src_chunk_id": chunk["id"], "fam": "oracle"}):
            return None
    context = C.format_context_single(chunk)
    answer = teach(C.NUMEROS, context, question, url, model, max_tokens=384, temperature=0.0)
    if not answer:
        return None
    # régua honesta (idêntica ao teto): cobertura + antitautologia + desempate/alucinação.
    r = C.deterministic_pass(question, answer, facts, threshold)
    if r.needs_judge or (r.approved and not r.empty):
        try:
            jr = C.judge_disambiguate(question, answer, r.facts_ambiguous, chunk["text"])
            amb_low = {a.lower(): a for a in r.facts_ambiguous}
            for fp in jr.get("fatos_presentes", []):
                key = fp.strip().lower()
                for al, orig in amb_low.items():
                    if key == al or al.startswith(key[:12]) or key.startswith(al[:12]):
                        r.facts_present.append(orig)
            r.facts_present = list(set(r.facts_present))
            r.hallucination = bool(jr.get("alucinacao", False))
        except Exception:
            pass
    coverage = len(r.facts_present) / max(1, len(facts))
    if not (coverage >= threshold and not r.tautology and not r.hallucination and not r.empty):
        return None
    return make_example(C.SYNTHESIS_SYSTEM_PROMPT_V2, context, question, answer, "oracle",
                        {"src_chunk_id": chunk["id"], "src_doc": chunk["doc"],
                         "src_section": chunk["section"], "question": question,
                         "facts": facts, "coverage": round(coverage, 3), "gen_model": model})


def gen_nonoracle(family: str, chunk: Dict[str, Any], pool: List[Dict[str, Any]],
                  url: str, model: str, lf: C.LexicalFilter, rng: random.Random
                  ) -> Optional[Dict[str, Any]]:
    """Famílias (b/c/d): contexto que NÃO responde; alvo = recusa/delimitação útil."""
    qa = gen_qa(chunk, url, model)
    if not qa:
        return None
    question, facts = qa
    with _lock:
        if not lf.accept(question, meta={"src_chunk_id": chunk["id"], "fam": family}):
            return None

    gold_text = chunk["text"]  # trecho-ouro real (referência p/ alucinação no juiz)
    removed = None
    if family == "recusa":
        wrong = C.pick_wrong_norm_chunk(chunk, pool, rng)
        if not wrong:
            return None
        given = C.format_context_single(wrong)
        teacher = C.TEACHER_RECUSA
        ctx_meta = {"wrong_chunk_id": wrong["id"], "wrong_doc": wrong["doc"]}
    elif family == "distrator":
        dist = C.pick_same_norm_distractor(chunk, pool, facts, rng)
        if not dist:
            return None
        given = C.format_context_single(dist)
        teacher = C.TEACHER_DISTRATOR
        ctx_meta = {"distractor_chunk_id": dist["id"], "distractor_section": dist["section"]}
    elif family == "parcial":
        partial_text, removed = C.make_partial_context(chunk, facts)
        if removed is None:
            return None  # não deu p/ tirar o dado-chave -> não vira exemplo parcial
        pc = {**chunk, "text": partial_text}
        given = C.format_context_single(pc)
        teacher = C.TEACHER_PARCIAL
        ctx_meta = {"src_chunk_id": chunk["id"], "removed_sentence": removed[:200]}
    else:
        return None

    answer = teach(teacher, given, question, url, model, max_tokens=320, temperature=0.2)
    if not answer:
        return None

    # Aprovação pelo JUIZ da família (critério = recusar/delimitar CORRETO, não cobrir fatos).
    try:
        verdict = C.judge_nonoracle(family, question, given, answer, gold_text)
    except Exception:
        return None
    if not verdict or not C.approve_nonoracle(family, verdict):
        return None

    return make_example(C.SYNTHESIS_SYSTEM_PROMPT_V2, given, question, answer, family,
                        {"src_chunk_id": chunk["id"], "src_doc": chunk["doc"],
                         "src_section": chunk["section"], "question": question,
                         "facts": facts, "verdict": verdict, "gen_model": model, **ctx_meta})


def load_done(out_path: Path) -> Dict[str, int]:
    """Conta pares já existentes por família (checkpoint) e evita duplicar (chunk,family)."""
    done_by_fam: Dict[str, int] = {}
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                m = json.loads(s)["meta"]
                fam = m.get("family", "?")
                done_by_fam[fam] = done_by_fam.get(fam, 0) + 1
                done_keys.add((m.get("src_chunk_id"), fam))
            except Exception:
                pass
    return done_by_fam, done_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-oracle", type=int, default=1600)
    ap.add_argument("--target-recusa", type=int, default=640)
    ap.add_argument("--target-parcial", type=int, default=640)
    ap.add_argument("--target-distrator", type=int, default=320)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--out", default=str(_HERE / "data" / "train_v2.jsonl"))
    ap.add_argument("--discards", default=str(_HERE / "data" / "train_lexical_discards.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_chunks, valtest_chunks = C.split_chunks(seed=args.seed, min_len=args.min_len)
    print(f"chunks: train={len(train_chunks)} valtest={len(valtest_chunks)} (val/teste FORA do treino)",
          flush=True)

    targets = {"oracle": args.target_oracle, "recusa": args.target_recusa,
               "parcial": args.target_parcial, "distrator": args.target_distrator}
    print(f"alvos por família: {targets} (total {sum(targets.values())})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_by_fam, done_keys = load_done(out_path)
    if not out_path.exists():
        out_path.write_text("# " + json.dumps({
            "artifact": "sft_v2_4family", "task": "poc-sft-v2", "model": args.model,
            "threshold": args.threshold, "seed": args.seed, "targets": targets,
            "reserved_nrs": C.RESERVED_NRS,
            "system_prompt_v2": C.SYNTHESIS_SYSTEM_PROMPT_V2[:60] + "...",
            "created_utc": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(f"checkpoint: {done_by_fam} pares já existentes", flush=True)

    lf = C.LexicalFilter()
    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()

    # Overshoot de tarefas: cada tentativa pode falhar (régua/juiz reprova), então
    # geramos MAIS candidatos que o alvo. Oráculo aprova ~muito; recusa/distrator menos.
    overshoot = {"oracle": 1.6, "recusa": 3.0, "parcial": 3.5, "distrator": 4.0}

    def run_family(family: str, chunk_iter: List[Dict[str, Any]], worker_fn) -> int:
        target = targets[family]
        have = done_by_fam.get(family, 0)
        if have >= target:
            print(f"[{family}] já completo ({have}/{target})", flush=True)
            return have
        need = target - have
        n_tasks = int(need * overshoot[family]) + 20
        tasks = [c for c in chunk_iter if (c["id"], family) not in done_keys][:n_tasks]
        kept = have
        tried = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker_fn, c): c["id"] for c in tasks}
            for fut in as_completed(futs):
                tried += 1
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                if rec is not None:
                    with _lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        kept += 1
                if tried % 40 == 0:
                    rate = tried / max(1e-9, time.time() - t0)
                    print(f"  [{family}] tried={tried}/{len(tasks)} kept={kept}/{target} "
                          f"lex_disc={len(lf.discards)} {rate:.1f}/s", flush=True)
                if kept >= target:
                    # cancela o resto das tarefas desta família
                    for f in futs:
                        f.cancel()
                    break
        print(f"[{family}] final: {kept}/{target} (tried {tried})", flush=True)
        return kept

    # Ordem: oráculo (metade), depois famílias sem oráculo. Pools distintos por família.
    # Oráculo usa a cabeça (numéricas priorizadas); as famílias de recusa usam o resto
    # embaralhado para diversidade.
    rng.shuffle(train_chunks)
    train_num_first = sorted(train_chunks, key=lambda c: 0 if C._NUM_SEC.match(str(c["section"]).strip()) else 1)

    run_family("oracle", train_num_first,
               lambda c: gen_oracle(c, args.url, args.model, lf, args.threshold))
    run_family("recusa", train_chunks,
               lambda c: gen_nonoracle("recusa", c, train_chunks, args.url, args.model, lf, rng))
    run_family("parcial", train_chunks,
               lambda c: gen_nonoracle("parcial", c, train_chunks, args.url, args.model, lf, rng))
    run_family("distrator", train_chunks,
               lambda c: gen_nonoracle("distrator", c, train_chunks, args.url, args.model, lf, rng))

    fh.close()
    lf.dump_discards(Path(args.discards))
    final_by_fam, _ = load_done(out_path)
    print(f"\nSFT v2 dados prontos: {final_by_fam} | total {sum(final_by_fam.values())} "
          f"| descartes lexicais {len(lf.discards)} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
