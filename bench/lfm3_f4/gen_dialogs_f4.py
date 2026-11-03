#!/usr/bin/env python3
"""
gen_dialogs_f4.py — dataset de tool-calling multiturno do F4 (6 famílias).

Receita tools_v1 adaptada ao LFM3: o 27B gera CONTEÚDO (JSON), a renderização
é 100% do apply_chat_template (XML <tool_call> nativo); fala CRUA como
argumento da 1ª busca (evidência tools_v1: reescrita piora); contexts do
retriever REAL (reuso dos assignments do data2x da F3b, k variável 3/5/8).

Famílias (alvo ~4.8k diálogos, 2-4 turnos):
  chamada_simples  1600  gold: user -> tool_call(fala crua) -> tool(top-k real)
                         -> síntese 27B (target do data2x) citando a norma.
  recusa           1050  absent: tool retorna contexto errado (erros genuínos
                         do retriever) -> recusa honesta (templates F3). ~22%.
  troca             700  2 QAs gold de NRs DIFERENTES: 2º turno muda de assunto
                         -> nova chamada (consulta nova = fala crua da 2ª).
  reuso             700  2º turno respondível com o contexto JÁ presente
                         (follow-up 27B ancorado no chunk-ouro) -> responde SEM
                         nova chamada.
  multihop          500  partial: hop1 retorna evidência PARCIAL -> modelo
                         reconhece insuficiência -> 2ª chamada com argumento
                         DIFERENTE (27B, validado: recupera o ouro no retriever
                         REAL) -> síntese 27B das DUAS evidências.
  sem_ferramenta    250  saudação/agradecimento/fora de escopo -> responde sem
                         chamar (negativos da decisão).

GATES mecânicos por exemplo: round-trip do render (common_f4.roundtrip_ok);
dedup Jaccard>=0.5 vs as 1536 do benchmark; dedup intra por pergunta; multihop
com 2º argumento != 1º E ouro de verdade no hop2.

Partição EVAL antes do treino (disjunta por qa_id): 120 gold (decisão+), 60
absent (recusa), 80 partial (multihop), 60 reuso, 60 troca, 60 notool.

Uso (classifier/.venv; 27B em 10.100.0.111:8005; llama-server :8399 no ar):
  classifier/.venv/bin/python bench/lfm3_f4/gen_dialogs_f4.py --stage assemble
  classifier/.venv/bin/python bench/lfm3_f4/gen_dialogs_f4.py --stage gen27b --workers 24
  classifier/.venv/bin/python bench/lfm3_f4/gen_dialogs_f4.py --stage build
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common_f4 as C  # noqa: E402
from tool_schema_f4 import (format_tool_result, make_assistant_message,  # noqa: E402
                            make_baked_system_message, make_tool_call_message,
                            make_tool_result_message, make_user_message)

# Quotas (train) e reservas (eval)
QUOTA = {"chamada_simples": 1600, "recusa": 1050, "troca": 700,
         "reuso": 700, "multihop": 500, "sem_ferramenta": 250}
RESERVE = {"gold": 240, "absent": 60, "partial": 80}   # eval, disjunto por qa_id
N_EVAL = {"decision_pos": 120, "reuse": 60, "switch": 60, "refusal": 60,
          "multihop": 60, "notool": 60}

# Recusa-padrão da F3 (bench/lfm3_f3/build_dataset.py — mesmos 4 templates;
# todos disparam looks_like_refusal da régua).
REFUSAL_TEMPLATES = [
    "Não encontrei essa informação nas normas recuperadas. Para confirmar, "
    "eu consultaria a {nr_hint} diretamente ou o responsável de segurança.",
    "Os trechos de norma recuperados não cobrem esse ponto. O caminho seguro "
    "é verificar a {nr_hint} na íntegra ou falar com o técnico de segurança.",
    "Isso não está nos trechos das normas que recuperei. Vale consultar a "
    "{nr_hint} ou o engenheiro de segurança antes de prosseguir.",
    "Essa resposta não está nas normas recuperadas. Recomendo conferir a "
    "{nr_hint} completa ou acionar o SESMT da empresa.",
]

SYS27 = ("Você é um assistente técnico de segurança do trabalho, especialista "
         "em Normas Regulamentadoras (NRs) brasileiras. Gera dados de treino. "
         "Responda sempre com um bloco JSON ao final.")

PROMPT_REFINE = """PERGUNTA DO TRABALHADOR:
{question}

O QUE A 1ª BUSCA RETORNOU (cobre só em parte / não cobre):
{partial_src}: {partial_text}

O QUE AINDA FALTA ENCONTRAR (referência da norma que responde): {ref}
TEMA do trecho que falta: {fato_hint}

Escreva TRÊS variantes de TERMOS DE BUSCA para uma SEGUNDA consulta ao
índice de normas, focadas no que FALTOU. Regras:
- cada variante com 3 a 9 palavras, termos técnicos da norma (não repita a
  pergunta inteira); varie o vocabulário entre as variantes.
- NÃO inclua o valor numérico/resposta em si; apenas termos para ENCONTRAR o trecho.
- Inclua o número da norma e do item/anexo da referência quando ajudar
  (ex.: "NR-18 18.10 anexo II").

Pense primeiro e emita APENAS o JSON final:
{{"consultas": ["<variante 1>", "<variante 2>", "<variante 3>"]}}"""

PROMPT_SYNTH2 = """TRECHO A ({src_a}) — encontrado na 1ª busca, cobre só em parte:
{text_a}

TRECHO B ({src_b}) — encontrado na 2ª busca, contém a resposta:
{text_b}

PERGUNTA DO TRABALHADOR:
{question}

Escreva a resposta ideal usando APENAS os dois trechos acima.
Regras:
- 2 a 5 frases, português do Brasil, tom direto de assistente de voz.
- Fiel aos trechos: preserve números, prazos e condições literalmente.
- Cite a norma e o item/seção de onde vem CADA informação usada
  (ex.: "NR-10 item 10.2.8"). Use os dois trechos quando os dois contribuírem.
- NÃO mencione "o trecho"/"a busca" — fale da norma.

Pense primeiro e emita APENAS o JSON final: {{"resposta": "<texto>"}}"""

PROMPT_FOLLOWUP = """TRECHO NORMATIVO ({src}):
{text}

CONVERSA JÁ OCORRIDA:
Trabalhador: {q1}
Assistente: {a1}

Escreva UMA pergunta de follow-up que o MESMO trabalhador faria em seguida,
respondível COM O MESMO TRECHO acima (outro detalhe do mesmo trecho), e a
resposta ideal.
Regras:
- Follow-up curto e coloquial (pode usar pronome: "e nesse caso...", "e se...").
- A resposta: 1 a 3 frases, cita a norma e o item, fiel ao trecho.
- O follow-up NÃO pode ser respondível só com a resposta anterior — tem de
  precisar do trecho (senão o aluno aprende a responder sem contexto).

Pense primeiro e emita APENAS o JSON final:
{{"followup": "<pergunta>", "resposta": "<resposta>"}}"""

PROMPT_NOTOOL = """Gere {n} interações CURTAS de um trabalhador de campo com um assistente de
voz de segurança do trabalho em que o assistente NÃO precisa consultar nenhuma
norma. Categorias (misture): saudação, agradecimento, pedido de repetição,
conversa geral/bom senso (sem dado normativo), fora de escopo (futebol, tempo,
pedido pessoal).
Contexto deste lote (varie fortemente o vocabulário e as situações em torno
disto; perguntas DIFERENTES entre si): {theme}.
Regras das respostas: 1-2 frases, educadas, sem citar norma; nas fora de
escopo, diga que só ajuda com segurança do trabalho.
Pense primeiro e emita APENAS o JSON final:
[{{"categoria": "...", "pergunta": "...", "resposta": "..."}}, ...]"""

NOTOOL_THEMES = [
    "início de turno na usina, equipe chegando",
    "fim de expediente, despedidas e agradecimentos",
    "almoço no canteiro de obras, conversa informal",
    "deslocamento na caminhonete entre torres de transmissão",
    "dia de chuva, espera no alojamento",
    "manutenção em subestação, pausa para café",
    "novato no primeiro dia pedindo orientações gerais",
    "veterano contando história, conversa fiada",
    "ligação cortada, pedidos de repetir/falar mais alto",
    "celular com pouca bateria, mensagens curtas",
    "sábado de jogo do Cruzeiro e do Atlético, papo de futebol",
    "calor de 35 graus no campo, reclamações do tempo",
    "pedidos pessoais (receita, música, loteria, horóscopo)",
    "dúvidas de bom senso sem norma (hidratação, descanso)",
    "elogios e feedback sobre o próprio assistente",
    "teste de microfone e frases sem sentido",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def nr_hint(ref: str, nr: int) -> str:
    return ref if ref else f"NR-{nr}"


def tool_cycle(question: str, chunks: List[Dict[str, Any]],
               final: Optional[str]) -> List[Dict[str, Any]]:
    """user -> tool_call(fala crua) -> tool -> [assistant final]."""
    msgs = [make_user_message(question),
            make_tool_call_message(question, C.detect_nr(question)),
            make_tool_result_message(format_tool_result(chunks))]
    if final is not None:
        msgs.append(make_assistant_message(final))
    return msgs


def ctx_chunks(chunks_db: Dict[int, Dict], ids: List[int]) -> List[Dict[str, Any]]:
    return [chunks_db[i] for i in ids if i in chunks_db]


def containment(fato_tokens: set, text: str) -> float:
    if not fato_tokens:
        return 0.0
    ct = set(C.norm_tokens(text))
    return len(fato_tokens & ct) / len(fato_tokens)


def is_gold(chunk: Dict[str, Any], nr: int, fato_tokens: set) -> bool:
    """Mesmo critério do retrieve_f3 (doc igual + containment >= 0.70)."""
    doc = str(chunk["doc"]).lower()
    return doc in (f"nr-{nr:02d}", f"nr-{nr}") and containment(fato_tokens, chunk["text"]) >= 0.70


# ---------------------------------------------------------------------------
# STAGE assemble — particiona pools e monta as famílias sem 27B
# ---------------------------------------------------------------------------
def stage_assemble() -> None:
    C.ensure_dirs()
    rng = random.Random(C.SEED)
    assigns = C.load_assignments()
    targets = C.load_targets()
    contam = C.ContaminationFilter()

    def usable(a: Dict) -> bool:
        if contam.contaminated(a["question"]):
            return False
        if a["case"] in ("gold", "partial") and a["qa_id"] not in targets:
            return False
        return True

    pools = {case: [a for a in assigns if a["case"] == case and usable(a)]
             for case in ("gold", "absent", "partial")}
    for k, v in pools.items():
        rng.shuffle(v)
    print({k: len(v) for k, v in pools.items()})

    # reserva de eval (disjunta por qa_id)
    ev = {c: pools[c][:RESERVE[c]] for c in pools}
    tr = {c: pools[c][RESERVE[c]:] for c in pools}

    # --- train: chamada_simples / recusa / troca / reuso -------------------
    gold = tr["gold"]
    plan: Dict[str, Any] = {}
    i = 0
    plan["chamada_simples"] = gold[i:i + QUOTA["chamada_simples"]]; i += QUOTA["chamada_simples"]
    # multi-turno usa só k<=5 (orçamento de tokens; registrado no README)
    rest = gold[i:]
    small_k = [a for a in rest if a["k"] <= 5]
    plan["troca_a"] = small_k[:QUOTA["troca"]]
    plan["troca_b"] = small_k[QUOTA["troca"]:2 * QUOTA["troca"]]
    plan["reuso"] = small_k[2 * QUOTA["troca"]:2 * QUOTA["troca"] + QUOTA["reuso"]]
    plan["recusa"] = tr["absent"][:QUOTA["recusa"]]
    # multihop: POOL COMPLETO dos partial de treino (o gate do retriever real
    # descarta muito; a quota é aplicada no build)
    plan["multihop"] = [a for a in tr["partial"] if a["k"] <= 5]
    # pares de troca com NRs diferentes
    pairs = []
    for a, b in zip(plan["troca_a"], plan["troca_b"]):
        if a["nr"] != b["nr"]:
            pairs.append((a, b))
    plan["troca_pairs"] = pairs[:QUOTA["troca"]]
    print(f"[assemble] troca pares NR!=NR: {len(plan['troca_pairs'])}")

    out = {
        "train": {k: [a["qa_id"] for a in v] for k, v in plan.items()
                  if k in ("chamada_simples", "recusa", "reuso", "multihop")},
        "train_troca_pairs": [(a["qa_id"], b["qa_id"]) for a, b in plan["troca_pairs"]],
        "eval": {c: [a["qa_id"] for a in v] for c, v in ev.items()},
    }
    (C.SYNTH / "plan.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    n = {k: len(v) for k, v in out["train"].items()}
    n["troca"] = len(out["train_troca_pairs"])
    print(f"[assemble] plan salvo: {n}; eval: " + str({c: len(v) for c, v in out['eval'].items()}))


# ---------------------------------------------------------------------------
# STAGE gen27b — conteúdo que precisa do 27B (+ validação no retriever real)
# ---------------------------------------------------------------------------
def stage_gen27b(workers: int) -> None:
    C.ensure_dirs()
    plan = json.loads((C.SYNTH / "plan.json").read_text(encoding="utf-8"))
    assigns = {a["qa_id"]: a for a in C.load_assignments()}
    targets = C.load_targets()
    units = C.load_units()
    chunks_db = C.load_chunks()
    contam = C.ContaminationFilter()

    from retriever_lazy import LazyRetrieverV4
    rtv = LazyRetrieverV4()

    # ---- multihop: refine (27B) -> retrieval real -> gold? -> synth2 (27B)
    # pré-filtro multihop: só vale tentar quando o chunk-ouro EXISTE no corpus
    # (containment>=0.70 em algum chunk da mesma NR) — ~24% dos partial não têm.
    from collections import defaultdict
    _by_doc: Dict[str, list] = defaultdict(list)
    for c in chunks_db.values():
        _by_doc[str(c["doc"]).lower()].append((set(C.norm_tokens(c["text"])), c["id"]))

    def gold_exists(nr: int, fato_tokens: set) -> bool:
        if not fato_tokens:
            return False
        for ct, _cid in _by_doc.get(f"nr-{nr:02d}", []) + _by_doc.get(f"nr-{nr}", []):
            if len(fato_tokens & ct) / len(fato_tokens) >= 0.70:
                return True
        return False

    def mh_worker(qa_id: str) -> Dict[str, Any]:
        a = assigns[qa_id]
        u = units[a["unit_id"]]
        fato_tokens = set(C.norm_tokens(u["fato"]))
        if not gold_exists(a["nr"], fato_tokens):
            return {"__discard__": True, "why": "no_gold_in_corpus"}
        anchor = a["anchor_passage"]
        src_a = f"{anchor['doc'].upper()} {anchor['section']} — {anchor['title']}"
        q1norm = " ".join(C.norm_tokens(a["question"]))
        # dica temática do que falta (primeiros termos do fato, sem o valor):
        fato_hint = " ".join(C.norm_tokens(u["fato"])[:10])
        det_query = f"{a['ref']} {fato_hint}"
        for attempt in range(2):
            v = C.gen_json(PROMPT_REFINE.format(
                question=a["question"], partial_src=src_a,
                partial_text=anchor["text"][:1200], ref=a["ref"],
                fato_hint=fato_hint),
                system=SYS27, temperature=0.5 + 0.2 * attempt, max_tokens=1800)
            cands = (v.get("consultas") if isinstance(v, dict) else
                     v if isinstance(v, list) else None) or []
            # candidata determinística por último (grátis; preferimos a do 27B
            # que é mais natural como argumento de modelo)
            cands = list(cands[:3]) + [det_query]
            q2 = gold_hit = hits = None
            for cand in cands:
                cq = str(cand).strip()
                if not cq or " ".join(C.norm_tokens(cq)) == q1norm:
                    continue
                chits = rtv.search_text(cq, top_k=5)
                gh = next((h for h in chits if is_gold(h, a["nr"], fato_tokens)), None)
                if gh:
                    q2, gold_hit, hits = cq, gh, chits
                    break
            if not gold_hit:
                continue
            # síntese das duas evidências
            sv = C.gen_json(PROMPT_SYNTH2.format(
                src_a=src_a, text_a=anchor["text"],
                src_b=f"{gold_hit['doc'].upper()} {gold_hit['section']} — {gold_hit['title']}",
                text_b=gold_hit["text"], question=a["question"]),
                system=SYS27, temperature=0.4, max_tokens=2048)
            resp = str(sv.get("resposta", "")).strip() if isinstance(sv, dict) else ""
            if len(resp) > 30 and "<texto" not in resp:
                return {"qa_id": qa_id, "consulta2": q2,
                        "hop2_ids": [h["id"] for h in hits],
                        "gold_hit_id": gold_hit["id"], "target2": resp}
        return {"__discard__": True}

    mh_ids = plan["train"]["multihop"] + plan["eval"]["partial"]
    res = C.parallel_map(mh_worker, mh_ids, max_workers=workers,
                         checkpoint_path=C.SYNTH / "multihop.ckpt.json",
                         key_fn=lambda x: x)
    mh_rows = [r for r in res.values()
               if isinstance(r, dict) and not r.get("__discard__") and "__error__" not in r]
    C.write_jsonl(C.SYNTH / "multihop.jsonl", mh_rows)
    print(f"[gen27b] multihop válidos: {len(mh_rows)}/{len(mh_ids)}")

    # ---- reuso: follow-up ancorado no chunk-ouro
    def fu_worker(qa_id: str) -> Dict[str, Any]:
        a = assigns[qa_id]
        gold_id = a["context_ids"][a["gold_position"]]
        gchunk = chunks_db[gold_id]
        src = f"{gchunk['doc'].upper()} {gchunk['section']} — {gchunk['title']}"
        for _ in range(3):
            v = C.gen_json(PROMPT_FOLLOWUP.format(
                src=src, text=gchunk["text"], q1=a["question"],
                a1=targets[qa_id]), system=SYS27, temperature=0.7,
                max_tokens=2000)
            if not isinstance(v, dict):
                continue
            fu = str(v.get("followup", "")).strip()
            resp = str(v.get("resposta", "")).strip()
            if (len(fu) > 10 and len(resp) > 30 and "<pergunta" not in fu
                    and not contam.contaminated(fu)):
                return {"qa_id": qa_id, "followup": fu, "resposta": resp}
        return {"__discard__": True}

    fu_ids = plan["train"]["reuso"] + plan["eval"]["gold"][:N_EVAL["reuse"]]
    res = C.parallel_map(fu_worker, fu_ids, max_workers=workers,
                         checkpoint_path=C.SYNTH / "followup.ckpt.json",
                         key_fn=lambda x: x)
    fu_rows = [r for r in res.values()
               if isinstance(r, dict) and not r.get("__discard__") and "__error__" not in r]
    C.write_jsonl(C.SYNTH / "followup.jsonl", fu_rows)
    print(f"[gen27b] followups válidos: {len(fu_rows)}/{len(fu_ids)}")

    # ---- sem_ferramenta: lotes de 25
    need = QUOTA["sem_ferramenta"] + N_EVAL["notool"]
    def nt_worker(batch_id: int) -> List[Dict[str, Any]]:
        # lotes de 10: com 25 o thinking + itens estouram o max_tokens e o
        # JSON final trunca (lotes voltavam vazios)
        theme = NOTOOL_THEMES[batch_id % len(NOTOOL_THEMES)]
        v = C.gen_json(PROMPT_NOTOOL.format(n=10, theme=theme), system=SYS27,
                       temperature=0.9, max_tokens=5000)
        rows = []
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and it.get("pergunta") and it.get("resposta"):
                    if not contam.contaminated(it["pergunta"]):
                        rows.append({"categoria": str(it.get("categoria", "")),
                                     "pergunta": str(it["pergunta"]).strip(),
                                     "resposta": str(it["resposta"]).strip()})
        return rows

    n_batches = (need + 9) // 10 + 8
    res = C.parallel_map(nt_worker, list(range(n_batches)), max_workers=min(workers, 8),
                         checkpoint_path=C.SYNTH / "notool.ckpt.json",
                         key_fn=lambda x: str(x))
    nt_rows, seen = [], set()
    for batch in res.values():
        if not isinstance(batch, list):
            continue
        for r in batch:
            key = " ".join(C.norm_tokens(r["pergunta"]))
            if key and key not in seen:
                seen.add(key)
                nt_rows.append(r)
    C.write_jsonl(C.SYNTH / "notool.jsonl", nt_rows)
    print(f"[gen27b] notool únicos: {len(nt_rows)} (precisa {need})")


# ---------------------------------------------------------------------------
# STAGE build — monta diálogos finais (train/val) + suites de eval
# ---------------------------------------------------------------------------
def stage_build() -> None:
    C.ensure_dirs()
    rng = random.Random(C.SEED + 7)
    plan = json.loads((C.SYNTH / "plan.json").read_text(encoding="utf-8"))
    assigns = {a["qa_id"]: a for a in C.load_assignments()}
    targets = C.load_targets()
    units = C.load_units()
    chunks_db = C.load_chunks()
    mh = {r["qa_id"]: r for r in C.read_jsonl(C.SYNTH / "multihop.jsonl")}
    fu = {r["qa_id"]: r for r in C.read_jsonl(C.SYNTH / "followup.jsonl")}
    nt = C.read_jsonl(C.SYNTH / "notool.jsonl")

    sys_msg = make_baked_system_message()
    rows: List[Dict[str, Any]] = []
    seen_q: set = set()
    stats = collections.Counter()

    def dedup_q(q: str) -> bool:
        key = " ".join(C.norm_tokens(q))
        if not key or key in seen_q:
            return False
        seen_q.add(key)
        return True

    def push(family: str, qa_id: str, msgs: List[Dict[str, Any]],
             meta: Optional[Dict] = None) -> None:
        full = [sys_msg] + msgs
        if not C.roundtrip_ok(full):
            stats[f"drop_rt_{family}"] += 1
            return
        rows.append({"messages": full, "family": family, "qa_id": qa_id,
                     **(meta or {})})
        stats[family] += 1

    # chamada_simples
    for qa_id in plan["train"]["chamada_simples"]:
        a = assigns[qa_id]
        if not dedup_q(a["question"]):
            continue
        push("chamada_simples", qa_id,
             tool_cycle(a["question"], ctx_chunks(chunks_db, a["context_ids"]),
                        targets[qa_id]), {"k": a["k"], "nr": a["nr"]})

    # recusa
    for qa_id in plan["train"]["recusa"]:
        a = assigns[qa_id]
        if not dedup_q(a["question"]):
            continue
        tgt = rng.choice(REFUSAL_TEMPLATES).format(nr_hint=nr_hint(a["ref"], a["nr"]))
        push("recusa", qa_id,
             tool_cycle(a["question"], ctx_chunks(chunks_db, a["context_ids"]), tgt),
             {"k": a["k"], "nr": a["nr"]})

    # troca (2 ciclos, NRs diferentes)
    for qa_a, qa_b in plan["train_troca_pairs"]:
        a, b = assigns[qa_a], assigns[qa_b]
        if not (dedup_q(a["question"]) and dedup_q(b["question"])):
            continue
        msgs = (tool_cycle(a["question"], ctx_chunks(chunks_db, a["context_ids"]), targets[qa_a])
                + tool_cycle(b["question"], ctx_chunks(chunks_db, b["context_ids"]), targets[qa_b]))
        push("troca", f"{qa_a}+{qa_b}", msgs, {"nr": a["nr"], "nr2": b["nr"]})

    # reuso (2º turno sem chamada)
    for qa_id in plan["train"]["reuso"]:
        if qa_id not in fu:
            stats["drop_fu_missing"] += 1
            continue
        a = assigns[qa_id]
        if not dedup_q(a["question"]):
            continue
        msgs = (tool_cycle(a["question"], ctx_chunks(chunks_db, a["context_ids"]), targets[qa_id])
                + [make_user_message(fu[qa_id]["followup"]),
                   make_assistant_message(fu[qa_id]["resposta"])])
        push("reuso", qa_id, msgs, {"nr": a["nr"]})

    # multihop (2 chamadas, 2º argumento diferente — gate re-verificado)
    n_mh = 0
    for qa_id in plan["train"]["multihop"]:
        if n_mh >= QUOTA["multihop"]:
            break
        if qa_id not in mh:
            continue
        a, m = assigns[qa_id], mh[qa_id]
        if " ".join(C.norm_tokens(m["consulta2"])) == " ".join(C.norm_tokens(a["question"])):
            stats["drop_mh_samearg"] += 1
            continue
        if not dedup_q(a["question"]):
            continue
        msgs = [make_user_message(a["question"]),
                make_tool_call_message(a["question"], C.detect_nr(a["question"])),
                make_tool_result_message(format_tool_result(
                    ctx_chunks(chunks_db, a["context_ids"]))),
                make_tool_call_message(m["consulta2"], f"NR-{a['nr']}"),
                make_tool_result_message(format_tool_result(
                    ctx_chunks(chunks_db, m["hop2_ids"]))),
                make_assistant_message(m["target2"])]
        push("multihop", qa_id, msgs, {"nr": a["nr"], "consulta2": m["consulta2"]})
        n_mh += 1

    # sem_ferramenta
    nt_train = nt[N_EVAL["notool"]:]      # os primeiros N_EVAL são da suite
    for i, r in enumerate(nt_train[:QUOTA["sem_ferramenta"]]):
        if not dedup_q(r["pergunta"]):
            continue
        push("sem_ferramenta", f"nt{i}",
             [make_user_message(r["pergunta"]), make_assistant_message(r["resposta"])],
             {"categoria": r["categoria"]})

    rng.shuffle(rows)
    # split 98/2 estratificado por família
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    train, val = [], []
    for fam, items in sorted(by_fam.items()):
        n_val = max(2, int(0.02 * len(items)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    C.write_jsonl(C.TRAIN / "train.jsonl", train)
    C.write_jsonl(C.TRAIN / "val.jsonl", val)

    stats_out = {"train": len(train), "val": len(val),
                 "by_family": {f: len(v) for f, v in by_fam.items()},
                 "drops": {k: v for k, v in stats.items() if k.startswith("drop")},
                 "seed": C.SEED}
    (C.TRAIN / "build_stats.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats_out, indent=2, ensure_ascii=False))

    # ---------------- suites de eval ----------------------------------------
    suites: Dict[str, List[Dict[str, Any]]] = {}

    # decisão: positivos = perguntas gold reservadas; negativos = notool eval
    pos = [{"id": f"pos_{q}", "question": assigns[q]["question"], "should_call": True,
            "qa_id": q, "nr": assigns[q]["nr"], "unit_id": assigns[q]["unit_id"],
            "ref": assigns[q]["ref"]}
           for q in plan["eval"]["gold"][N_EVAL["reuse"]:N_EVAL["reuse"] + N_EVAL["decision_pos"]]]
    neg = [{"id": f"neg{i}", "question": r["pergunta"], "should_call": False,
            "categoria": r["categoria"]}
           for i, r in enumerate(nt[:N_EVAL["notool"]])]
    suites["decision"] = pos + neg

    # reuso: histórico fixo (ciclo gold) + follow-up (não deveria chamar)
    reuse_items = []
    for q in plan["eval"]["gold"][:N_EVAL["reuse"]]:
        if q not in fu:
            continue
        a = assigns[q]
        reuse_items.append({
            "id": f"reuse_{q}", "qa_id": q, "nr": a["nr"],
            "history_question": a["question"], "history_ids": a["context_ids"],
            "history_answer": targets[q],
            "followup": fu[q]["followup"], "gold_answer": fu[q]["resposta"],
            "unit_id": a["unit_id"], "should_call": False})
    suites["reuse"] = reuse_items[:N_EVAL["reuse"]]

    # troca: histórico fixo + pergunta de OUTRA NR (deveria chamar de novo)
    sw, gold_ev = [], plan["eval"]["gold"]
    for qa_a, qa_b in zip(gold_ev[N_EVAL["reuse"] + N_EVAL["decision_pos"]:],
                          gold_ev[N_EVAL["reuse"]:]):
        a, b = assigns[qa_a], assigns[qa_b]
        if a["nr"] == b["nr"] or len(sw) >= N_EVAL["switch"]:
            continue
        sw.append({"id": f"switch_{qa_a}_{qa_b}", "qa_id": qa_b, "nr2": b["nr"],
                   "history_question": a["question"], "history_ids": a["context_ids"],
                   "history_answer": targets[qa_a],
                   "new_question": b["question"], "unit_id": b["unit_id"],
                   "ref": b["ref"], "should_call": True})
    suites["switch"] = sw

    # recusa: perguntas absent reservadas (retriever real vai errar de novo)
    suites["refusal"] = [
        {"id": f"ref_{q}", "qa_id": q, "question": assigns[q]["question"],
         "nr": assigns[q]["nr"], "ref": assigns[q]["ref"],
         "unit_id": assigns[q]["unit_id"],
         "origin_absent": assigns[q]["origin_absent"]}
        for q in plan["eval"]["absent"][:N_EVAL["refusal"]]]

    # multihop: partial reservados COM refine válido (o circuito completo é
    # medido ao vivo; consulta2 do 27B fica como referência, não como gabarito)
    mh_items = []
    for q in plan["eval"]["partial"]:
        if q not in mh or len(mh_items) >= N_EVAL["multihop"]:
            continue
        a = assigns[q]
        mh_items.append({"id": f"mh_{q}", "qa_id": q, "question": a["question"],
                         "nr": a["nr"], "ref": a["ref"], "unit_id": a["unit_id"],
                         "hop1_ids": a["context_ids"],
                         "ref_consulta2": mh[q]["consulta2"],
                         "gold_hit_id": mh[q]["gold_hit_id"]})
    suites["multihop"] = mh_items

    for name, items in suites.items():
        C.write_jsonl(C.EVAL / f"suite_{name}.jsonl", items)
    print("[build] suites:", {k: len(v) for k, v in suites.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["assemble", "gen27b", "build"])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.stage == "assemble":
        stage_assemble()
    elif args.stage == "gen27b":
        stage_gen27b(args.workers)
    else:
        stage_build()


if __name__ == "__main__":
    main()
