#!/usr/bin/env python3
"""
common_v2.py — Núcleo compartilhado do SFT v2 (4 famílias + varredura de rank).

O v1 (slm_oraculo/) provou que destilar o estilo acionável do 27B funciona em ORÁCULO
(Q4 45->58%). O diagnóstico do v1 (ordem do capitão) origina o v2:
  1. OVERFITTING: loss caiu monotônica enquanto a QUALIDADE PIOROU (ep1->ep3 alucinação
     25->30%). Sem validation set. -> v2 treina 1 ÉPOCA e mede a RÉGUA na validação.
  2. O MODELO NÃO SABE RECUSAR: alucinação 25% em oráculo, 41-51% no mundo real. Treinado
     só com chunk-ouro. -> v2 fabrica dados COM e SEM oráculo.
  3. r=16 nunca foi testado com o MLP dentro (bug dos targets do v1). -> varre r 16/32/64.
  4. Camiseta rasgada: o v1 responde citando "vedado o uso de adornos pessoais" (item de
     joias, ERRADO) com convicção. -> família DISTRATOR PLAUSÍVEL ensina a não atribuir.

Este módulo concentra:
  - o system prompt de TREINO v2 (permite RECUSA ÚTIL, ao contrário do produção que manda
    a recusa preguiçosa "diga apenas: Não sei...");
  - os prompts do PROFESSOR (27B) para gerar o alvo de cada família;
  - os JUÍZES de aprovação das famílias de recusa (critério = recusar/delimitar CORRETO,
    não cobrir fatos — implementado e explicado aqui);
  - o SPLIT de chunks train/val/refusal-test (val e teste FORA do treino, seed fixa);
  - reuso das 3 muralhas anti-contaminação do finetune2/common.

Comentários em PT-BR; identificadores em inglês.
"""

from __future__ import annotations

import importlib.util
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# finetune2/common tem as muralhas + call_vllm + load_chunks (import por caminho para
# evitar o choque de nome com bench/regua/common).
_ft = _load_module("ft2_common", _ROOT / "finetune2" / "common.py")
DEFAULT_VLLM_MODEL = _ft.DEFAULT_VLLM_MODEL
DEFAULT_VLLM_URL = _ft.DEFAULT_VLLM_URL
RESERVED_NRS = _ft.RESERVED_NRS
LexicalFilter = _ft.LexicalFilter
call_vllm = _ft.call_vllm
extract_json = _ft.extract_json
holdout_gold_chunk_ids = _ft.holdout_gold_chunk_ids
load_chunks = _ft.load_chunks
_PROD_SYNTHESIS_PROMPT = _ft.SYNTHESIS_SYSTEM_PROMPT

# régua honesta (bench/regua)
sys.path.insert(0, str(_ROOT / "bench" / "regua"))
from ruler import deterministic_pass  # noqa: E402
from judge_regua import judge_disambiguate  # noqa: E402
from common import judge_call  # noqa: E402

# ---------------------------------------------------------------------------
# SYSTEM PROMPT DE TREINO v2 — permite a RECUSA ÚTIL (a diferença central do v2).
#
# É idêntico ao prompt de produção (AskPipeline SYNTHESIS_SYSTEM_PROMPT: voz, <=4 frases,
# sem markdown, cita norma, baseia-se só no contexto) EXCETO a última regra: em vez de
# mandar a recusa preguiçosa ("diga APENAS: Não sei..."), o v2 ensina a recusa HONESTA E
# ÚTIL — diz que a resposta não está nas normas consultadas E o que seria preciso verificar.
# TODAS as 4 famílias de treino usam ESTE prompt (consistência: o modelo vê um só system).
# ---------------------------------------------------------------------------
SYNTHESIS_SYSTEM_PROMPT_V2 = (
    "Você é o assistente técnico de campo da CEMIG. O eletricista OUVE sua resposta por voz, "
    "então seja curto e direto.\n"
    "REGRAS OBRIGATÓRIAS (nunca viole):\n"
    "- Responda em NO MÁXIMO 4 frases curtas.\n"
    "- PROIBIDO usar markdown, títulos, negrito, listas, bullets ou numeração. Escreva em prosa corrida.\n"
    "- Cite a norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...').\n"
    "- Baseie-se EXCLUSIVAMENTE no contexto normativo fornecido; não invente procedimentos, "
    "valores nem itens.\n"
    "- Se o contexto trouxer só PARTE da resposta, responda o que der e diga com franqueza o "
    "que não dá para afirmar com o que foi consultado.\n"
    "- Se o contexto NÃO responder à pergunta, não invente: diga que isso não está nas normas "
    "consultadas e, em uma frase, o que seria preciso verificar (a norma ou o assunto que "
    "trataria disso)."
)

# ---------------------------------------------------------------------------
# Prompt do PROFESSOR (27B) para gerar PERGUNTA + FATOS a partir de um trecho (reuso do v1).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Prompt do PROFESSOR — alvo ACIONÁVEL do ORÁCULO (reuso do v1: prompt `numeros`).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_ROOT / "bench" / "prompt_teto"))
from prompts import NUMEROS  # noqa: E402

# ---------------------------------------------------------------------------
# Prompts do PROFESSOR para as famílias SEM oráculo (o alvo é a RECUSA/DELIMITAÇÃO útil).
# São prompts de PROFESSOR (geram o alvo ideal); o exemplo de treino guarda como system o
# SYNTHESIS_SYSTEM_PROMPT_V2 e como assistant a saída aprovada do professor.
# ---------------------------------------------------------------------------
TEACHER_RECUSA = (
    "Você é um auditor técnico de Normas Regulamentadoras brasileiras. Um eletricista fez uma "
    "pergunta, mas o ÚNICO contexto disponível é de OUTRO assunto/norma e NÃO responde à "
    "pergunta. Produza a RESPOSTA IDEAL para o eletricista ouvir por voz:\n"
    "- Recuse com FRANQUEZA e UTILIDADE: diga que isso NÃO está na norma que foi consultada.\n"
    "- Diga, em uma frase, O QUE SERIA PRECISO consultar (a norma ou o assunto que trata do tema).\n"
    "- PROIBIDO inventar procedimento, valor ou item. PROIBIDO atribuir o conteúdo do contexto "
    "errado à pergunta. PROIBIDO a recusa preguiçosa vazia ('não sei' e ponto).\n"
    "- No MÁXIMO 3 frases curtas, prosa corrida, sem markdown."
)
TEACHER_PARCIAL = (
    "Você é um auditor técnico de Normas Regulamentadoras brasileiras. Um eletricista fez uma "
    "pergunta e o contexto é da norma CERTA, mas está INCOMPLETO: falta o dado-chave que a "
    "pergunta pede. Produza a RESPOSTA IDEAL para o eletricista ouvir por voz:\n"
    "- Responda o que DER para responder com o contexto disponível (cite a norma e o item que "
    "aparecem no contexto).\n"
    "- Declare EXPLICITAMENTE o que NÃO dá para afirmar com o que foi consultado (o dado que "
    "falta) — e NÃO invente esse dado.\n"
    "- PROIBIDO inventar o valor/dado ausente. No MÁXIMO 4 frases curtas, prosa corrida, sem markdown."
)
TEACHER_DISTRATOR = (
    "Você é um auditor técnico de Normas Regulamentadoras brasileiras. Um eletricista fez uma "
    "pergunta e o contexto é da MESMA norma, mas de uma seção VIZINHA que trata de OUTRO "
    "assunto e NÃO responde à pergunta. É uma armadilha: atribuir o item vizinho à pergunta "
    "seria um ERRO grave. Produza a RESPOSTA IDEAL para o eletricista ouvir por voz:\n"
    "- NÃO atribua o item do contexto à pergunta. Diga que o trecho encontrado trata de outro "
    "assunto e não responde diretamente ao que foi perguntado.\n"
    "- Diga, em uma frase, o que seria preciso verificar para responder de fato.\n"
    "- PROIBIDO inventar procedimento/valor. No MÁXIMO 3 frases curtas, prosa corrida, sem markdown."
)

TEACHER_USER = ("Contexto normativo consultado:\n{context}\n\n"
                "Pergunta do eletricista:\n{question}\n\n"
                "Escreva SOMENTE a resposta ideal (sem preâmbulo).")

# ---------------------------------------------------------------------------
# JUÍZ de aprovação das famílias SEM oráculo (critério = recusar/delimitar CORRETO).
#
# Ordem do capitão: "nas famílias de recusa, o critério de aprovação é RECUSAR
# CORRETAMENTE, não cobrir fatos". Este juiz 27B decide, com o contexto DADO ao modelo e o
# trecho-ouro REAL (para pegar alucinação), os eixos abaixo. A aprovação por família é
# computada em Python (transparente e auditável), não escondida no prompt.
# ---------------------------------------------------------------------------
JUDGE_NONORACLE_PROMPT = """Você é um auditor técnico de Normas Regulamentadoras brasileiras.
Avalie a RESPOSTA DO MODELO a um eletricista, sabendo que o CONTEXTO fornecido ao modelo
{context_desc}. Você também recebe o TRECHO-OURO real (que de fato responde à pergunta) só
para checar alucinação.

[PERGUNTA DO ELETRICISTA]
{question}

[CONTEXTO QUE FOI DADO AO MODELO]
{given_context}

[TRECHO-OURO REAL (referência para checar alucinação; o modelo NÃO o viu)]
{gold_text}

[RESPOSTA DO MODELO]
{answer}

Responda avaliando estes eixos (true/false):
- "recusou_ou_delimitou": a resposta reconhece honestamente que o contexto NÃO responde (ou
  responde só em parte) à pergunta, em vez de fingir que respondeu.
- "disse_o_que_faltou": a resposta indica o que seria preciso consultar / qual dado falta
  (recusa ÚTIL), em vez da recusa preguiçosa vazia.
- "atribuiu_item_errado": a resposta ATRIBUI à pergunta um item/assunto do contexto que na
  verdade não responde (ex.: responder sobre roupa rasgada citando "vedado adornos pessoais").
- "inventou_dado": a resposta afirma valor/procedimento/item que NÃO está no contexto dado
  nem é sustentado pelo trecho-ouro (alucinação).
- "respondeu_o_possivel": a resposta aproveita corretamente a parte do contexto que existe
  (relevante só quando o contexto responde parcialmente).

Retorne EXCLUSIVAMENTE um objeto JSON estrito:
{{"recusou_ou_delimitou": false, "disse_o_que_faltou": false, "atribuiu_item_errado": false, "inventou_dado": false, "respondeu_o_possivel": false, "justificativa": "curta"}}"""

_CONTEXT_DESC = {
    "recusa": "é de OUTRA norma/assunto e NÃO responde à pergunta",
    "parcial": "é da norma CERTA mas está INCOMPLETO (falta o dado-chave)",
    "distrator": "é da MESMA norma mas de uma seção VIZINHA que NÃO responde à pergunta",
}


def _parse_bool_json(raw: str, keys: List[str]) -> Optional[Dict[str, Any]]:
    obj = extract_json(raw)
    if not obj:
        return None
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = bool(obj.get(k, False))
    out["justificativa"] = str(obj.get("justificativa", ""))[:300]
    return out


_NONORACLE_KEYS = ["recusou_ou_delimitou", "disse_o_que_faltou",
                   "atribuiu_item_errado", "inventou_dado", "respondeu_o_possivel"]


def judge_nonoracle(family: str, question: str, given_context: str, answer: str,
                    gold_text: str) -> Optional[Dict[str, Any]]:
    """Juiz 27B das famílias sem oráculo. Retorna os eixos ou None em falha de parse."""
    prompt = JUDGE_NONORACLE_PROMPT.format(
        context_desc=_CONTEXT_DESC.get(family, "NÃO responde à pergunta"),
        question=question, given_context=given_context[:2500],
        gold_text=(gold_text or "(indisponível)")[:2500], answer=answer[:2000])
    raw = judge_call(prompt, max_tokens=512, temperature=0.0)
    return _parse_bool_json(raw, _NONORACLE_KEYS)


def approve_nonoracle(family: str, verdict: Dict[str, Any]) -> bool:
    """
    Critério de aprovação por família (explícito, ordem do capitão):
      - recusa   : recusou_ou_delimitou E disse_o_que_faltou E NÃO inventou E NÃO atribuiu errado
                   (recusa HONESTA E ÚTIL — não a preguiçosa).
      - distrator: recusou_ou_delimitou E NÃO atribuiu_item_errado E NÃO inventou
                   (o pecado capital aqui é atribuir o item vizinho, o caso da camiseta).
      - parcial  : respondeu_o_possivel E recusou_ou_delimitou (declarou o que faltou)
                   E NÃO inventou_dado (não preencheu o buraco com invenção).
    """
    v = verdict
    if family == "recusa":
        return (v["recusou_ou_delimitou"] and v["disse_o_que_faltou"]
                and not v["inventou_dado"] and not v["atribuiu_item_errado"])
    if family == "distrator":
        return (v["recusou_ou_delimitou"] and not v["atribuiu_item_errado"]
                and not v["inventou_dado"])
    if family == "parcial":
        return (v["respondeu_o_possivel"] and v["recusou_ou_delimitou"]
                and not v["inventou_dado"])
    return False


# ---------------------------------------------------------------------------
# Detecção de RECUSA (para a métrica taxa-de-recusa-correta da Parte 3).
# Barato + robusto: frases típicas de recusa/delimitação honesta.
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    r"n[aã]o (está|esta|consta|traz|aborda|responde|encontr|foi encontr)",
    r"n[aã]o (sei|posso (afirmar|orientar|garantir|responder|dizer))",
    r"n[aã]o (h[aá]|existe|disp)",
    r"nas normas consultad", r"no(s)? trecho(s)? consultad",
    r"n[aã]o est[aá] nas normas", r"n[aã]o trata (disso|desse|do que)",
    r"seria preciso (consultar|verificar)", r"trata de outro (assunto|tema)",
    r"n[aã]o (é|e) poss[ií]vel (afirmar|responder|orientar)",
    r"contexto (não|nao) (responde|traz|cobre|aborda)",
    r"fora do (que|assunto)", r"n[aã]o (cobre|contempla)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def looks_like_refusal(answer: str) -> bool:
    """Heurística determinística de recusa/delimitação (rápida; o juiz confirma no limite)."""
    a = (answer or "").strip()
    if len(a) < 4:
        return True  # resposta vazia = não respondeu
    return bool(_REFUSAL_RE.search(a))


JUDGE_ISREFUSAL_PROMPT = """Você audita respostas de um assistente de NRs. Diga apenas se a
RESPOSTA abaixo é uma RECUSA/DELIMITAÇÃO (o assistente reconhece que não pode responder à
pergunta com o contexto, ou que o contexto não responde) OU uma RESPOSTA EFETIVA (tenta
responder de fato a pergunta, dando procedimento/valor/orientação).

[PERGUNTA]
{question}

[RESPOSTA DO MODELO]
{answer}

Retorne EXCLUSIVAMENTE: {{"recusou": true}} se for recusa/delimitação, ou {{"recusou": false}}
se for uma tentativa efetiva de responder."""


def judge_is_refusal(question: str, answer: str) -> bool:
    """Confirmação do juiz 27B para casos ambíguos de recusa."""
    prompt = JUDGE_ISREFUSAL_PROMPT.format(question=question, answer=answer[:1500])
    raw = judge_call(prompt, max_tokens=64, temperature=0.0)
    obj = extract_json(raw)
    if obj is not None and "recusou" in obj:
        return bool(obj["recusou"])
    return looks_like_refusal(answer)


# ---------------------------------------------------------------------------
# Formatação de contexto — idêntica ao AskPipeline.kt.
# ---------------------------------------------------------------------------
def format_context_single(c: Dict[str, Any]) -> str:
    return f"[1] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}"


def build_user(context: str, question: str) -> str:
    return (f"Contexto normativo consultado:\n{context}\n\n"
            f"Pergunta do eletricista:\n{question}")


# ---------------------------------------------------------------------------
# SPLIT de chunks: train vs (val + refusal-test). val/teste FORA do treino (seed fixa).
# ---------------------------------------------------------------------------
_NUM_SEC = re.compile(r"^\d+(?:\.\d+)*$")
VALTEST_FRACTION = 0.14  # ~14% dos chunks elegíveis reservados p/ val + refusal-test


def eligible_chunks(min_len: int = 200) -> List[Dict[str, Any]]:
    """Chunks elegíveis: fora do holdout-ouro, fora das NRs reservadas, texto >= min_len."""
    chunks = load_chunks()
    gold = holdout_gold_chunk_ids()
    elig = [c for c in chunks if c["id"] not in gold
            and c["doc"] not in RESERVED_NRS and len(c["text"]) >= min_len]
    return elig


def split_chunks(seed: int = 17, min_len: int = 200
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retorna (train_chunks, valtest_chunks) DISJUNTOS por id. Prioriza seções numéricas no
    treino (Q&A de campo melhores que anexos de despejo). Seed fixa -> reprodutível.
    """
    elig = eligible_chunks(min_len)
    rng = random.Random(seed)
    rng.shuffle(elig)
    n_valtest = max(60, int(len(elig) * VALTEST_FRACTION))
    valtest = elig[:n_valtest]
    train = elig[n_valtest:]
    train.sort(key=lambda c: 0 if _NUM_SEC.match(str(c["section"]).strip()) else 1)
    return train, valtest


def pick_wrong_norm_chunk(src: Dict[str, Any], pool: List[Dict[str, Any]],
                          rng: random.Random) -> Optional[Dict[str, Any]]:
    """Escolhe um chunk de OUTRA norma (família recusa)."""
    others = [c for c in pool if c["doc"] != src["doc"]]
    return rng.choice(others) if others else None


def pick_same_norm_distractor(src: Dict[str, Any], pool: List[Dict[str, Any]],
                              facts: List[str], rng: random.Random
                              ) -> Optional[Dict[str, Any]]:
    """
    Escolhe um chunk da MESMA norma, seção diferente, com BAIXA sobreposição com os fatos
    (distrator plausível: mesma norma, assunto vizinho que não responde). O caso da camiseta.
    """
    from ruler import _content_tokens  # import tardio p/ evitar ciclo
    fact_toks: Set[str] = set()
    for f in facts:
        fact_toks |= set(_content_tokens(f))
    cands = [c for c in pool if c["doc"] == src["doc"] and c["id"] != src["id"]]
    if not cands:
        return None
    scored = []
    for c in cands:
        ctoks = set(_content_tokens(c["text"][:600]))
        overlap = len(fact_toks & ctoks) / max(1, len(fact_toks))
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0])  # menor sobreposição primeiro = distrator melhor
    # amostra entre os 5 menos sobrepostos p/ diversidade
    low = [c for _, c in scored[:max(1, min(5, len(scored)))]]
    return rng.choice(low)


def make_partial_context(chunk: Dict[str, Any], facts: List[str]
                         ) -> Tuple[str, Optional[str]]:
    """
    Constrói contexto PARCIAL do chunk certo: remove a(s) sentença(s) que carregam o
    dado-chave (maior sobreposição com os fatos). Retorna (texto_parcial, sentença_removida).
    Se não sobrar contexto útil, retorna (texto_original, None).
    """
    from ruler import _content_tokens
    text = chunk["text"]
    # segmenta em sentenças de forma barata
    sents = re.split(r"(?<=[.;:])\s+(?=[A-ZÀ-Ú0-9])", text)
    if len(sents) < 3:
        return text, None
    fact_toks: Set[str] = set()
    for f in facts:
        fact_toks |= set(_content_tokens(f))
    # pontua cada sentença pela sobreposição com os fatos; remove a de maior escore
    scored = []
    for i, s in enumerate(sents):
        stoks = set(_content_tokens(s))
        # bônus se a sentença tem número (o dado-chave costuma ser numérico)
        has_num = bool(re.search(r"\d", s))
        score = len(fact_toks & stoks) + (0.5 if has_num else 0)
        scored.append((score, i, s))
    scored.sort(reverse=True)
    top_score, top_i, top_sent = scored[0]
    if top_score <= 0:
        return text, None
    kept = [s for j, s in enumerate(sents) if j != top_i]
    partial = " ".join(kept).strip()
    if len(partial) < 120:  # sobrou pouco -> descarta a construção parcial
        return text, None
    return partial, top_sent
