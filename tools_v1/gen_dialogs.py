#!/usr/bin/env python3
"""
gen_dialogs.py — PASSO 1+2: fabrica os 4.500 diálogos multiturno em 5 famílias.

Arquitetura (EMENDA 2 do capitão, fechada): o 27B é gerador de CONTEÚDO, nunca de FORMATO.
Pedimos a ele só campos SEMÂNTICOS em JSON guiado (guided_json). A renderização no formato
nativo do LFM2.5 é NOSSA e determinística (render_jinja/apply_chat_template — paridade
byte-a-byte provada em test_render_parity). O treino salva os `messages` (dicts); o render
é feito no train (assistant_only_loss nativo via {% generation %}).

5 famílias (proporção do brief; total-alvo 4500 — emenda 1):
  (1) chamada_simples   ~35% (1575): pergunta -> tool_call(consulta,nr) -> tool -> resposta.
  (2) sem_ferramenta    ~15% ( 675): saudação/agradecimento/repetição/bom senso/fora de escopo.
  (3) multiturno_reuso  ~20% ( 900): 2º turno no MESMO assunto já em contexto -> NÃO chama.
  (4) multiturno_nova   ~15% ( 675): assunto MUDA -> chama de novo com consulta diferente.
  (5) recusa_apos_busca ~15% ( 675): tool retorna trecho errado/parcial -> recusa/delimita.

VALIDAÇÃO MECÂNICA por exemplo (PASSO 2, não confiar no juiz p/ o que dá p/ executar):
  - round-trip: renderiza a tool_call e re-parseia; consulta/nr voltam IDÊNTICOS (escape de
    aspas, acento, vírgula decimal '13,8 kV');
  - o 'nr' existe no corpus;
  - a CONSULTA recupera o chunk-ouro no índice v4 (top-K) — senão descarta/regera;
  - a resposta final passa pela régua honesta (famílias com resposta) OU pelo critério de
    recusa (família 5);
  - o alvo CITA o item da norma (decisão B) — exigido nas famílias que respondem.

MURALHAS (reuso finetune2/common via common_tools): holdout intocável, NR-33/16/26 reservadas,
filtro lexical Jaccard<0.4 vs holdout. Split de chunks train/valtest (sft_v2.split_chunks).

Higiene: paralelo, checkpoint incremental por linha, procedência, não-interativo.
Comentários PT-BR; identificadores em inglês.

Uso (classifier/.venv; juiz/gerador 27B default 10.100.0.111:8005):
  ../classifier/.venv/bin/python gen_dialogs.py --total 4500 --workers 24
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import common_tools as C
import render_jinja as RJ
import retrieval_check as RC
from render import parse_tool_calls
from tool_schema import (SYSTEM_PROMPT_TOOLS, TOOLS, format_tool_result,
                         make_assistant_message, make_system_message,
                         make_tool_call_message, make_tool_result_message,
                         make_user_message)

_HERE = Path(__file__).resolve().parent
_lock = Lock()

# dedup intra-dataset de perguntas (evita quase-duplicatas ao ciclar o pool de chunks).
_seen_questions: set = set()


def _norm_q(q: str) -> str:
    return " ".join(re.sub(r"[^\wçãáéíóúâêôàõ ]", "", (q or "").lower()).split())


def _register_question(q: str) -> bool:
    """True se a pergunta é nova (registra); False se já vista (quase-duplicata exata)."""
    key = _norm_q(q)
    if not key:
        return False
    with _lock:
        if key in _seen_questions:
            return False
        _seen_questions.add(key)
        return True

CITATION_RE = re.compile(r"nr[\s\-]?\d{1,2}", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Schemas guided_json (o 27B devolve SÓ semântica).
# ---------------------------------------------------------------------------
SCHEMA_QUERY = {
    "type": "object",
    "properties": {
        "consulta": {"type": "string"},
        "nr": {"type": ["string", "null"]},
    },
    "required": ["consulta", "nr"],
}
SCHEMA_NOTOOL = {
    "type": "object",
    "properties": {
        "categoria": {"type": "string"},
        "pergunta": {"type": "string"},
        "resposta": {"type": "string"},
    },
    "required": ["categoria", "pergunta", "resposta"],
}
SCHEMA_FOLLOWUP = {
    "type": "object",
    "properties": {
        "pergunta": {"type": "string"},
        "resposta": {"type": "string"},
    },
    "required": ["pergunta", "resposta"],
}

# ---------------------------------------------------------------------------
# Prompts do 27B (CONTEÚDO semântico apenas).
# ---------------------------------------------------------------------------
QUERY_SYSTEM = (
    "Você prepara dados para um assistente de campo que consulta Normas Regulamentadoras por "
    "voz. Recebe a fala CRUA de um eletricista e produz os argumentos de uma busca. Devolva "
    "JSON com:\n"
    "1. \"consulta\": a fala REESCRITA em termos técnicos de busca — troque a linguagem de "
    "campo pelos termos das NRs, adicione sinônimos técnicos úteis e NORMALIZE números "
    "(ex.: '13,8 kV' vire '13,8 kV 13.8 kV', 'treze mil e oitocentos volts' vire '13,8 kV'). "
    "Só palavras-chave, sem frase completa, sem citar número de item.\n"
    "2. \"nr\": a norma no formato 'NR-10' SOMENTE se a fala citar a norma ou se for "
    "inequívoca; caso contrário null.\n"
    "Responda SOMENTE o JSON."
)
QUERY_USER = "Fala do eletricista:\n{question}\n\nGere o JSON com \"consulta\" e \"nr\"."

QUERY_RETRY_SYSTEM = QUERY_SYSTEM + (
    "\nATENÇÃO: a consulta anterior NÃO recuperou o trecho certo. Torne a consulta MAIS "
    "específica com os termos técnicos exatos do assunto e mais sinônimos."
)

NOTOOL_SYSTEM = (
    "Você gera exemplos de treino para um assistente de campo da CEMIG que responde "
    "eletricistas por voz e tem uma ferramenta de busca em normas. Gere um caso em que o "
    "assistente NÃO deve usar a ferramenta. Devolva JSON com:\n"
    "1. \"categoria\": uma de {saudacao, agradecimento, repeticao, bom_senso, fora_de_escopo}.\n"
    "2. \"pergunta\": a fala do eletricista dessa categoria (curta, coloquial, brasileira). "
    "Para 'repeticao' peça para repetir/explicar melhor de forma genérica; para 'bom_senso' "
    "uma dúvida de senso comum que não precisa de norma; para 'fora_de_escopo' algo alheio a "
    "segurança do trabalho.\n"
    "3. \"resposta\": a resposta curta e cordial do assistente SEM citar norma nem inventar "
    "dado; no MÁXIMO 3 frases, prosa corrida, sem markdown. Para 'fora_de_escopo' recuse com "
    "gentileza e redirecione ao escopo de normas de segurança.\n"
    "Responda SOMENTE o JSON."
)
NOTOOL_USER = "Gere um caso da categoria '{categoria}'. Responda o JSON."

FOLLOWUP_REUSE_SYSTEM = (
    "Você gera dados para um assistente de campo por voz. O eletricista já recebeu uma "
    "resposta baseada NO TRECHO abaixo. Gere um SEGUNDO turno natural em que ele faz uma "
    "pergunta de acompanhamento sobre o MESMO assunto, que AINDA É RESPONDÍVEL só com esse "
    "trecho (não precisa de nova busca). Devolva JSON com:\n"
    "1. \"pergunta\": a pergunta de acompanhamento (curta, coloquial).\n"
    "2. \"resposta\": a resposta do assistente usando SÓ o trecho, citando a norma e o item, "
    "no MÁXIMO 4 frases, prosa corrida, sem markdown.\n"
    "Responda SOMENTE o JSON."
)
FOLLOWUP_REUSE_USER = ("TRECHO ({doc}, item {sec} — {title}):\n\"\"\"\n{text}\n\"\"\"\n\n"
                       "Primeira pergunta do eletricista foi: {q1}\nGere o JSON do 2º turno.")

FOLLOWUP_NEW_SYSTEM = (
    "Você gera dados para um assistente de campo por voz. O eletricista mudou de assunto. "
    "Gere uma transição natural para um NOVO tópico (o do segundo trecho). Devolva JSON com "
    "só \"pergunta\": a nova fala coloquial do eletricista sobre o assunto do 2º trecho "
    "(curta, sem termos técnicos, sem citar item)."
)
FOLLOWUP_NEW_USER = ("Assunto anterior: {q1}\n\nNOVO assunto — TRECHO ({doc}, item {sec} — "
                     "{title}):\n\"\"\"\n{text}\n\"\"\"\n\nGere o JSON com \"pergunta\".")

SCHEMA_NEWQ = {"type": "object", "properties": {"pergunta": {"type": "string"}},
               "required": ["pergunta"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gen_query(question: str, url: str, model: str, retry: bool = False
              ) -> Optional[Tuple[str, Optional[str]]]:
    """27B reescreve a fala em (consulta, nr). Vê SÓ a pergunta (== inferência)."""
    sysmsg = QUERY_RETRY_SYSTEM if retry else QUERY_SYSTEM
    obj = C.call_vllm_json(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content": QUERY_USER.format(question=question)}],
        SCHEMA_QUERY, url=url, model=model, temperature=0.7, max_tokens=256)
    if not obj or "consulta" not in obj:
        return None
    consulta = " ".join(str(obj["consulta"]).strip().split())
    if len(consulta) < 4:
        return None
    nr_raw = obj.get("nr")
    nr = None
    if nr_raw and str(nr_raw).lower() not in ("null", "none", ""):
        norm = C.normalize_nr(str(nr_raw))
        nr = C.nr_display(norm) if norm and norm != C.NONE_LABEL else None
    return consulta, nr


def valid_call(consulta: str, nr: Optional[str], corpus_nr_set: set) -> bool:
    """Round-trip do render + nr existe no corpus (PASSO 2, parte sintática)."""
    if nr is not None and nr.lower() not in corpus_nr_set:
        return False
    return RJ.roundtrip_ok(consulta, nr)


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer or ""))


def teach_answer(context: str, question: str, url: str, model: str) -> Optional[str]:
    """Alvo acionável do 27B (prompt NUMEROS da régua). == sft_v2.teach oráculo."""
    from common_v2 import NUMEROS, build_user  # reuso do v2
    user = build_user(context, question)
    ans = C.call_vllm([{"role": "system", "content": NUMEROS},
                       {"role": "user", "content": user}],
                      url=url, model=model, temperature=0.0, max_tokens=384, thinking=False)
    ans = (ans or "").strip()
    return ans if len(ans) >= 20 else None


def find_query_recovering_gold(question: str, gold_id: int, relevant_ids: List[int],
                               url: str, model: str, corpus_nr_set: set, topk: int
                               ) -> Optional[Tuple[str, Optional[str]]]:
    """Gera consulta que RECUPERA o chunk-ouro no v4 (1 retry com hint). Senão None."""
    for retry in (False, True):
        qr = gen_query(question, url, model, retry=retry)
        if not qr:
            continue
        consulta, nr = qr
        if not valid_call(consulta, nr, corpus_nr_set):
            continue
        if RC.recovers_gold(consulta, gold_id, topk=topk, relevant_ids=relevant_ids):
            return consulta, nr
    return None


# ---------------------------------------------------------------------------
# Famílias
# ---------------------------------------------------------------------------
def fam_simples(chunk: Dict[str, Any], url: str, model: str, lf, threshold: float,
                corpus_nr_set: set, topk: int) -> Optional[Dict[str, Any]]:
    """(1) pergunta -> tool_call -> tool -> resposta acionável citando o item."""
    qa = _gen_qa(chunk, url, model)
    if not qa:
        return None
    question, facts = qa
    with _lock:
        if not lf.accept(question, meta={"src": chunk["id"], "fam": "chamada_simples"}):
            return None
    if not _register_question(question):
        return None
    qr = find_query_recovering_gold(question, chunk["id"], [chunk["id"]], url, model,
                                    corpus_nr_set, topk)
    if not qr:
        return None
    consulta, nr = qr
    context = f"[1] ({chunk['doc']} - {chunk['section']} - {chunk['title']}):\n{chunk['text']}"
    answer = teach_answer(context, question, url, model)
    if not answer or not has_citation(answer):
        return None
    gold_text = chunk["text"]
    if not C.approve_answer(question, answer, facts, gold_text, threshold):
        return None
    msgs = [make_system_message(), make_user_message(question),
            make_tool_call_message(consulta, nr),
            make_tool_result_message(format_tool_result([chunk])),
            make_assistant_message(answer)]
    return _wrap(msgs, "chamada_simples",
                 {"src_chunk_id": chunk["id"], "doc": chunk["doc"], "section": chunk["section"],
                  "question": question, "consulta": consulta, "nr": nr, "facts": facts})


def fam_sem_ferramenta(cat: str, url: str, model: str) -> Optional[Dict[str, Any]]:
    """(2) saudação/agradecimento/repetição/bom senso/fora de escopo -> responde SEM chamar."""
    obj = C.call_vllm_json(
        [{"role": "system", "content": NOTOOL_SYSTEM},
         {"role": "user", "content": NOTOOL_USER.format(categoria=cat)}],
        SCHEMA_NOTOOL, url=url, model=model, temperature=0.9, max_tokens=256)
    if not obj:
        return None
    question = " ".join(str(obj.get("pergunta", "")).strip().split())
    answer = str(obj.get("resposta", "")).strip()
    if len(question) < 3 or len(answer) < 5:
        return None
    # não pode citar norma (não buscou) nem conter tool_call
    if has_citation(answer):
        return None
    msgs = [make_system_message(), make_user_message(question),
            make_assistant_message(answer)]
    return _wrap(msgs, "sem_ferramenta", {"categoria": cat, "question": question})


def fam_reuso(chunk: Dict[str, Any], url: str, model: str, lf, threshold: float,
              corpus_nr_set: set, topk: int) -> Optional[Dict[str, Any]]:
    """(3) turno1 chamada; turno2 followup no MESMO assunto -> NÃO chama, reusa contexto."""
    base = fam_simples(chunk, url, model, lf, threshold, corpus_nr_set, topk)
    if not base:
        return None
    q1 = base["meta"]["question"]
    obj = C.call_vllm_json(
        [{"role": "system", "content": FOLLOWUP_REUSE_SYSTEM},
         {"role": "user", "content": FOLLOWUP_REUSE_USER.format(
             doc=chunk["doc"], sec=chunk["section"], title=chunk["title"],
             text=chunk["text"][:1600], q1=q1)}],
        SCHEMA_FOLLOWUP, url=url, model=model, temperature=0.8, max_tokens=384)
    if not obj:
        return None
    q2 = " ".join(str(obj.get("pergunta", "")).strip().split())
    a2 = str(obj.get("resposta", "")).strip()
    if len(q2) < 4 or len(a2) < 15 or not has_citation(a2):
        return None
    with _lock:
        if not lf.accept(q2, meta={"src": chunk["id"], "fam": "multiturno_reuso_q2"}):
            return None
    msgs = base["messages"] + [make_user_message(q2), make_assistant_message(a2)]
    m = base["meta"]
    m.update({"q2": q2, "family": "multiturno_reuso"})
    return _wrap(msgs, "multiturno_reuso", m)


def fam_nova_busca(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any], url: str, model: str,
                   lf, threshold: float, corpus_nr_set: set, topk: int
                   ) -> Optional[Dict[str, Any]]:
    """(4) turno1 chamada (chunk A); turno2 NOVO assunto -> nova chamada (chunk B)."""
    base = fam_simples(chunk_a, url, model, lf, threshold, corpus_nr_set, topk)
    if not base:
        return None
    q1 = base["meta"]["question"]
    # nova pergunta coloquial sobre o assunto do chunk B
    obj = C.call_vllm_json(
        [{"role": "system", "content": FOLLOWUP_NEW_SYSTEM},
         {"role": "user", "content": FOLLOWUP_NEW_USER.format(
             q1=q1, doc=chunk_b["doc"], sec=chunk_b["section"], title=chunk_b["title"],
             text=chunk_b["text"][:1400])}],
        SCHEMA_NEWQ, url=url, model=model, temperature=0.85, max_tokens=160)
    if not obj:
        return None
    q2 = " ".join(str(obj.get("pergunta", "")).strip().split())
    if len(q2) < 4:
        return None
    with _lock:
        if not lf.accept(q2, meta={"src": chunk_b["id"], "fam": "multiturno_nova_q2"}):
            return None
    facts_b = _gen_qa(chunk_b, url, model)  # fatos p/ a régua do 2º turno
    if not facts_b:
        return None
    _, facts2 = facts_b
    qr = find_query_recovering_gold(q2, chunk_b["id"], [chunk_b["id"]], url, model,
                                    corpus_nr_set, topk)
    if not qr:
        return None
    consulta2, nr2 = qr
    context_b = f"[1] ({chunk_b['doc']} - {chunk_b['section']} - {chunk_b['title']}):\n{chunk_b['text']}"
    a2 = teach_answer(context_b, q2, url, model)
    if not a2 or not has_citation(a2):
        return None
    if not C.approve_answer(q2, a2, facts2, chunk_b["text"], threshold):
        return None
    msgs = base["messages"] + [
        make_user_message(q2), make_tool_call_message(consulta2, nr2),
        make_tool_result_message(format_tool_result([chunk_b])),
        make_assistant_message(a2)]
    m = base["meta"]
    m.update({"q2": q2, "consulta2": consulta2, "nr2": nr2, "src_chunk_id_b": chunk_b["id"],
              "family": "multiturno_nova_busca"})
    return _wrap(msgs, "multiturno_nova_busca", m)


def fam_recusa(subfam: str, chunk: Dict[str, Any], pool: List[Dict[str, Any]], url: str,
               model: str, lf, rng: random.Random, corpus_nr_set: set) -> Optional[Dict[str, Any]]:
    """(5) tool retorna trecho ERRADO/PARCIAL -> recusa/delimita (reuso dos juízes do v2).

    subfam in {recusa, parcial, distrator}. A CHAMADA é uma reescrita legítima da pergunta;
    o RESULTADO injetado é o trecho errado/parcial. O alvo é recusar/delimitar corretamente.
    """
    qa = _gen_qa(chunk, url, model)
    if not qa:
        return None
    question, facts = qa
    with _lock:
        if not lf.accept(question, meta={"src": chunk["id"], "fam": f"recusa_{subfam}"}):
            return None
    if not _register_question(question):
        return None
    # consulta legítima (não exigimos recuperar o ouro — o ponto é o resultado ruim)
    qr = gen_query(question, url, model)
    if not qr:
        return None
    consulta, nr = qr
    if not valid_call(consulta, nr, corpus_nr_set):
        return None

    gold_text = chunk["text"]
    if subfam == "recusa":
        wrong = C.V2.pick_wrong_norm_chunk(chunk, pool, rng)
        if not wrong:
            return None
        given_chunk = wrong
        teacher = C.V2.TEACHER_RECUSA
    elif subfam == "distrator":
        dist = C.V2.pick_same_norm_distractor(chunk, pool, facts, rng)
        if not dist:
            return None
        given_chunk = dist
        teacher = C.V2.TEACHER_DISTRATOR
    else:  # parcial
        partial_text, removed = C.V2.make_partial_context(chunk, facts)
        if removed is None:
            return None
        given_chunk = {**chunk, "text": partial_text}
        teacher = C.V2.TEACHER_PARCIAL

    given = f"[1] ({given_chunk['doc']} - {given_chunk['section']} - {given_chunk['title']}):\n{given_chunk['text']}"
    from common_v2 import TEACHER_USER
    ans = C.call_vllm([{"role": "system", "content": teacher},
                       {"role": "user", "content": TEACHER_USER.format(
                           context=given, question=question)}],
                      url=url, model=model, temperature=0.2, max_tokens=320, thinking=False)
    ans = (ans or "").strip()
    if len(ans) < 20:
        return None
    try:
        verdict = C.V2.judge_nonoracle(subfam, question, given, ans, gold_text)
    except Exception:
        return None
    if not verdict or not C.V2.approve_nonoracle(subfam, verdict):
        return None
    msgs = [make_system_message(), make_user_message(question),
            make_tool_call_message(consulta, nr),
            make_tool_result_message(format_tool_result([given_chunk])),
            make_assistant_message(ans)]
    return _wrap(msgs, "recusa_apos_busca",
                 {"subfam": subfam, "src_chunk_id": chunk["id"], "doc": chunk["doc"],
                  "given_chunk_id": given_chunk["id"], "question": question,
                  "consulta": consulta, "nr": nr, "verdict": verdict})


def _gen_qa(chunk: Dict[str, Any], url: str, model: str):
    """Reusa o gerador de pergunta+fatos do v2 (professor 27B)."""
    from gen_data import gen_qa  # sft_v2.gen_data
    return gen_qa(chunk, url, model)


def _wrap(msgs: List[Dict[str, Any]], family: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = {"family": family, **meta, "created_utc": datetime.now(timezone.utc).isoformat()}
    return {"messages": msgs, "meta": meta}


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def load_done(out_path: Path):
    by_fam: Dict[str, int] = {}
    keys = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                d = json.loads(s)
                fam = d["meta"].get("family", "?")
                by_fam[fam] = by_fam.get(fam, 0) + 1
                keys.add((d["meta"].get("src_chunk_id"), fam))
                # preload das perguntas já usadas (dedup ao resumir do checkpoint)
                for qk in ("question", "q2"):
                    q = d["meta"].get(qk)
                    if q:
                        _seen_questions.add(_norm_q(q))
            except Exception:
                pass
    return by_fam, keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=5, help="top-K do índice v4 p/ aceitar a consulta")
    ap.add_argument("--url", default=C.DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=C.DEFAULT_VLLM_MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--out", default=str(_HERE / "data" / "dialogs.jsonl"))
    ap.add_argument("--discards", default=str(_HERE / "data" / "lexical_discards.json"))
    args = ap.parse_args()

    # proporção das 5 famílias
    frac = {"chamada_simples": 0.35, "sem_ferramenta": 0.15, "multiturno_reuso": 0.20,
            "multiturno_nova_busca": 0.15, "recusa_apos_busca": 0.15}
    targets = {k: round(v * args.total) for k, v in frac.items()}
    # ajusta arredondamento
    diff = args.total - sum(targets.values())
    targets["chamada_simples"] += diff
    print(f"alvos por família: {targets} (total {sum(targets.values())})", flush=True)

    rng = random.Random(args.seed)
    train_chunks, valtest_chunks = C.V2.split_chunks(seed=args.seed, min_len=args.min_len)
    print(f"chunks: train={len(train_chunks)} valtest={len(valtest_chunks)} "
          f"(val/teste FORA do treino)", flush=True)
    corpus_nr_set = set(C.corpus_nrs())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_by_fam, done_keys = load_done(out_path)
    if not out_path.exists():
        out_path.write_text("# " + json.dumps({
            "artifact": "tools_v1_dialogs", "task": "poc-tools-1.2b", "model": args.model,
            "total_target": args.total, "targets": targets, "topk_v4": args.topk,
            "seed": args.seed, "reserved_nrs": C.RESERVED_NRS,
            "system_prompt": SYNTHESIS_HASH(),
            "created_utc": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(f"checkpoint: {done_by_fam} já existentes", flush=True)

    lf = C.LexicalFilter()
    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()

    overshoot = {"chamada_simples": 2.2, "sem_ferramenta": 1.4, "multiturno_reuso": 3.0,
                 "multiturno_nova_busca": 3.5, "recusa_apos_busca": 3.0}

    def emit(rec):
        if rec is None:
            return 0
        with _lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        return 1

    def run_family(family: str, task_gen, worker_fn):
        target = targets[family]
        have = done_by_fam.get(family, 0)
        if have >= target:
            print(f"[{family}] já completo ({have}/{target})", flush=True)
            return
        need = target - have
        tasks = task_gen(int(need * overshoot[family]) + 20)
        kept = have
        tried = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker_fn, t) for t in tasks]
            for fut in as_completed(futs):
                tried += 1
                try:
                    rec = fut.result()
                except Exception:
                    rec = None
                kept += emit(rec)
                if tried % 40 == 0:
                    rate = tried / max(1e-9, time.time() - t0)
                    print(f"  [{family}] tried={tried}/{len(tasks)} kept={kept}/{target} "
                          f"{rate:.1f}/s", flush=True)
                if kept >= target:
                    for f in futs:
                        f.cancel()
                    break
        print(f"[{family}] final: {kept}/{target} (tried {tried})", flush=True)

    # pools por família
    rng.shuffle(train_chunks)
    num_first = sorted(train_chunks,
                       key=lambda c: 0 if C.V2._NUM_SEC.match(str(c["section"]).strip()) else 1)

    def cycle(pool):
        """Cicla o pool de chunks p/ gerar N tarefas > |pool| (múltiplas perguntas por chunk,
        cada uma com fala/consulta diferente pela temperatura — a variável de maior variância
        segundo o capitão é justamente a consulta reescrita)."""
        def gen(n):
            return [pool[i % len(pool)] for i in range(n)]
        return gen

    # (1) chamada_simples
    run_family("chamada_simples",
               cycle(num_first),
               lambda c: fam_simples(c, args.url, args.model, lf, args.threshold,
                                     corpus_nr_set, args.topk))

    # (2) sem_ferramenta — categorias balanceadas
    cats = ["saudacao", "agradecimento", "repeticao", "bom_senso", "fora_de_escopo"]
    run_family("sem_ferramenta",
               lambda n: [cats[i % len(cats)] for i in range(n)],
               lambda cat: fam_sem_ferramenta(cat, args.url, args.model))

    # (3) multiturno_reuso
    run_family("multiturno_reuso",
               cycle(num_first),
               lambda c: fam_reuso(c, args.url, args.model, lf, args.threshold,
                                   corpus_nr_set, args.topk))

    # (4) multiturno_nova_busca — pares (A,B) de NRs distintas
    def pairs_new(n):
        out = []
        for i in range(n):
            a = num_first[i % len(num_first)]
            b = rng.choice([c for c in train_chunks if c["doc"] != a["doc"]])
            out.append((a, b))
        return out
    run_family("multiturno_nova_busca",
               pairs_new,
               lambda ab: fam_nova_busca(ab[0], ab[1], args.url, args.model, lf,
                                         args.threshold, corpus_nr_set, args.topk))

    # (5) recusa_apos_busca — subfamílias recusa/parcial/distrator
    subs = ["recusa", "parcial", "distrator"]
    run_family("recusa_apos_busca",
               lambda n: [(subs[i % len(subs)], train_chunks[i % len(train_chunks)])
                          for i in range(n)],
               lambda sc: fam_recusa(sc[0], sc[1], train_chunks, args.url, args.model,
                                     lf, rng, corpus_nr_set))

    fh.close()
    lf.dump_discards(Path(args.discards))
    final_by_fam, _ = load_done(out_path)
    print(f"\nDiálogos prontos: {final_by_fam} | total {sum(final_by_fam.values())} "
          f"| descartes lexicais {len(lf.discards)} | {(time.time()-t0)/60:.1f}min", flush=True)


def SYNTHESIS_HASH() -> str:
    import hashlib
    return hashlib.sha256(SYSTEM_PROMPT_TOOLS.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
