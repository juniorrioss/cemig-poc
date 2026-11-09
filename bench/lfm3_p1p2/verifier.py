"""
verifier.py — CONTRATO do reward do P2 (mini-RLVR de groundedness) e régua
automática do P1 (aprendibilidade). Zero juiz: tudo verificável por regra.

Reward BINÁRIO (brief do capitão, princípio "recompensa dura primeiro" do
paper Rufus-Air):

  r = 1  se  expect == "answer" (ouro presente no contexto) E
             (a) cita o item-ouro: algum código de item da resposta casa com
                 um código-ouro (igual, ou filho mais específico: 18.8.6.1.2
                 casa com 18.8.6.1). Se o item não tem código (tabela/nota),
                 a condição é dispensada;
             (b) cobre >= X dos fatos do gabarito (match canônico DETERMINÍSTICO
                 da régua honesta: `bench/regua/ruler.py`, só fatos "presentes";
                 os ambíguos NÃO contam — sem juiz);
             (c) nenhum número fora do contexto: todo número da resposta existe
                 no contexto ou na pergunta (com normalização de vírgula/ponto,
                 milhar, número por extenso no contexto, e item-pai de código
                 citado no contexto);
             (d) não é recusa (a régua de avaliação conta recusa como reprovação);
  r = 1  se  expect == "refuse" (ouro ausente) E recusa explícita;
             v1: "recusa" = regex amplo `looks_like_refusal` em QUALQUER ponto;
             v2 (CONTRACT_VERSION=2, default): recusa EXPLÍCITA nas primeiras
             ~200 chars (`STRICT_REFUSAL_RE`: "não encontrei", "os trechos
             recuperados não cobrem", "não está nas normas"...). Motivo (run P2#1): com v1 a política aprendeu a
             RESPONDER e fechar com uma ressalva ("a norma não traz outra..."),
             que o regex amplo contava como recusa — recusa explícita no
             distrator caiu 185→40/344 e a alucinação subiu.
  r = 1  se  expect == "no_tool" E não emitiu <tool_call> E resposta não-vazia
             (só no P1, família sem_ferramenta do F4);
  r = 0  caso contrário. Truncada (sem EOS) => 0 sempre.

`expect == "partial"` (hedge intencional do F3) usa (b)+(c) sem (a)/(d).
`expect == "closed"` (âncora sem contexto) usa (a)+(b) sem (c) (não há contexto).

Comentários em português; identificadores em inglês.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

_REPO = Path(__file__).resolve().parents[2]
# no container os módulos viajam junto em ./deps (ruler.py, common_f0.py)
for _p in (_REPO / "bench" / "regua", _REPO / "bench" / "lfm3_f0",
           Path(__file__).resolve().parent / "deps"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ruler import _content_tokens, _fact_match_ratio, PRESENT_RATIO  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402

# limiar X de cobertura — calibrado no smoke (ver RESULTS.md §Calibração)
COVERAGE_X = 0.5

# mesmo regex de código de item do score_e2e da F5 (coluna "cita item-ouro")
ITEM_RE = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d{1,3}){1,5})(?![\w]|\.\d)")
# qualquer número: inteiro, decimal com vírgula/ponto, código pontuado
NUM_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*")

TOOL_CALL_RE = re.compile(r"<tool_call>|<\|tool_call_start\|>")

CONTRACT_VERSION = 2
REFUSAL_HEAD_CHARS = 200
# recusa EXPLÍCITA: declara que a informação não está no material recuperado
STRICT_REFUSAL_RE = re.compile(
    r"n[aã]o (encontrei|localizei|achei|foi (poss[ií]vel )?(encontr|localiz))"
    r"|n[aã]o (est[aá]|consta|aparece|h[aá] (essa|esta|tal) informa\w*)\s+(n[oa]s?|em) "
    r"(normas?|trechos?|contexto|material|documentos?|passage\w*)"
    # sujeito = MATERIAL RECUPERADO (trechos/contexto/normas recuperadas), nunca "a
    # norma" genérica: "a norma não traz outra exigência" é ressalva, não recusa
    r"|(trechos?|contexto|passage\w*|documentos?|material|"
    r"normas? (recuperad|consultad|fornecid|dispon)\w*)( de norma)?"
    r"( \w+){0,3} n[aã]o (cobre|cobrem|traz|trazem|cont[eé]m|cont[eê]m|abordam?|respondem?|"
    r"tratam?|mencionam?|informam?|especificam?|apresentam?|incluem?|definem?)"
    r"|n[aã]o (tenho|disponho de) (essa|esta|a) informa"
    r"|n[aã]o (é|e) poss[ií]vel (afirmar|responder|confirmar|determinar|informar)"
    r"|sem (essa|esta) informa\w* n[oa]s? (normas?|trechos?|contexto)",
    re.IGNORECASE)


def is_explicit_refusal(answer: str) -> bool:
    """Recusa explícita nas primeiras REFUSAL_HEAD_CHARS (contrato v2)."""
    return bool(STRICT_REFUSAL_RE.search((answer or "")[:REFUSAL_HEAD_CHARS]))


# ---------------------------------------------------------------------------
# números por extenso (PT-BR) — o texto das NRs escreve "0,8 m (oitenta
# centímetros)"; uma resposta "80 cm" NÃO é número inventado.
# ---------------------------------------------------------------------------
_UNITS = {"zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
          "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
          "dez": 10, "onze": 11, "doze": 12, "treze": 13, "catorze": 14,
          "quatorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17,
          "dezoito": 18, "dezenove": 19}
_TENS = {"vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50,
         "sessenta": 60, "setenta": 70, "oitenta": 80, "noventa": 90}
_HUNDREDS = {"cem": 100, "cento": 100, "duzentos": 200, "duzentas": 200,
             "trezentos": 300, "trezentas": 300, "quatrocentos": 400,
             "quatrocentas": 400, "quinhentos": 500, "quinhentas": 500,
             "seiscentos": 600, "seiscentas": 600, "setecentos": 700,
             "setecentas": 700, "oitocentos": 800, "oitocentas": 800,
             "novecentos": 900, "novecentas": 900}
_WORDNUM = {**_UNITS, **_TENS, **_HUNDREDS}


def _deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def word_numbers(text: str) -> Set[str]:
    """Valores de números escritos por extenso ("dois metros e noventa" -> 2, 90;
    "cento e vinte" -> 120; "mil" multiplica). Retorna strings canônicas."""
    toks = re.findall(r"[a-z]+", _deaccent(text.lower()))
    out: Set[str] = set()
    cur: Optional[int] = None
    for i, t in enumerate(toks):
        if t in _WORDNUM:
            v = _WORDNUM[t]
            cur = v if cur is None else cur + v
            out.add(str(v))
            out.add(str(cur))
        elif t == "mil":
            cur = (cur or 1) * 1000
            out.add(str(cur))
        elif t == "e" and cur is not None:
            continue                       # "cento e vinte": soma segue
        else:
            cur = None
    return out


def canon_forms(num: str) -> Set[str]:
    """Formas canônicas de um número textual: '0,80' -> {'0,80','0.80','0.8'};
    '1.000' -> {'1.000','1000'}; códigos '18.8.2' ficam como estão."""
    forms = {num}
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", num):          # milhar com ponto
        forms.add(num.replace(".", ""))
    if num.count(",") == 1 and "." not in num:            # decimal com vírgula
        dot = num.replace(",", ".")
        forms.add(dot)
        try:
            forms.add(_strip_float(dot))
        except ValueError:
            pass
    if num.count(".") == 1 and "," not in num and not re.fullmatch(r"\d{1,3}\.\d{3}", num):
        forms.add(_strip_float(num))                      # 0.80 -> 0.8
    if num.isdigit():
        forms.add(str(int(num)))                          # 05 -> 5
    return forms


def _strip_float(s: str) -> str:
    v = float(s)
    return ("%f" % v).rstrip("0").rstrip(".") if "." in s else str(int(v))


def numbers_in(text: str) -> List[str]:
    return NUM_RE.findall(text or "")


def allowed_numbers(*texts: str) -> Set[str]:
    """Conjunto canônico de números permitidos (contexto + pergunta)."""
    allowed: Set[str] = set()
    for t in texts:
        for n in numbers_in(t):
            allowed |= canon_forms(n)
            # partes de um código/decimal também são citáveis
            # ("18.8.2" permite "18.8" e "18"; "13,8" não permite "8" sozinho)
            if "." in n and "," not in n:
                parts = n.split(".")
                for k in range(1, len(parts)):
                    allowed.add(".".join(parts[:k]))
        allowed |= word_numbers(t)
    return allowed


def unsupported_numbers(answer: str, *contexts: str) -> List[str]:
    """Números da resposta que NÃO existem no contexto/pergunta."""
    allowed = allowed_numbers(*contexts)
    bad = []
    for n in numbers_in(answer):
        if not (canon_forms(n) & allowed):
            bad.append(n)
    return bad


# ---------------------------------------------------------------------------
# citação de item
# ---------------------------------------------------------------------------
def item_codes(text: str) -> Set[str]:
    return set(ITEM_RE.findall(text or ""))


def cites_gold_item(answer: str, gold_items: Iterable[str]) -> bool:
    gold = [g for g in gold_items if g]
    if not gold:
        return True                                       # dispensado
    got = item_codes(answer)
    return any(a == g or a.startswith(g + ".") for a in got for g in gold)


def fact_coverage(answer: str, facts: List[str]) -> float:
    """Cobertura determinística (só fatos claramente presentes, régua honesta)."""
    if not facts:
        return 0.0
    toks = _content_tokens(answer or "")
    hit = sum(1 for f in facts if _fact_match_ratio(f, toks) >= PRESENT_RATIO)
    return hit / len(facts)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    reward: int
    reason: str
    detail: Dict = field(default_factory=dict)


def strip_answer(raw: str) -> str:
    """Remove bloco <think></think> e tokens especiais finais."""
    t = raw or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    return t.replace("<|im_end|>", "").strip()


def verify(answer: str, item: Dict, stopped: bool = True,
           x: float = None) -> Verdict:
    """Aplica o contrato a UMA resposta. `item` precisa de: expect, facts,
    gold_items, context, question."""
    x = COVERAGE_X if x is None else x
    expect = item["expect"]
    a = strip_answer(answer)
    if not stopped:
        return Verdict(0, "truncated")
    if not a:
        return Verdict(0, "empty")
    has_tool = bool(TOOL_CALL_RE.search(a))
    refused = looks_like_refusal(a)

    if expect == "refuse":
        if CONTRACT_VERSION < 2:
            ok = refused and not has_tool
            return Verdict(int(ok), "refused" if ok else "answered_without_gold")
        if has_tool or not is_explicit_refusal(a):
            return Verdict(0, "hedged_refusal" if refused else "answered_without_gold")
        return Verdict(1, "refused")
    if expect == "no_tool":
        ok = not has_tool
        return Verdict(int(ok), "no_tool" if ok else "spurious_tool_call")
    if has_tool:
        return Verdict(0, "spurious_tool_call")

    facts = item.get("facts") or []
    cov = fact_coverage(a, facts)
    det = {"coverage": round(cov, 3)}
    if expect in ("answer", "closed"):
        cite = cites_gold_item(a, item.get("gold_items") or [])
        det["cite"] = cite
        if not cite:
            return Verdict(0, "no_gold_item", det)
    if expect == "answer" and refused:
        return Verdict(0, "undue_refusal", det)
    if cov < x:
        return Verdict(0, "low_coverage", det)
    if expect in ("answer", "partial"):
        bad = unsupported_numbers(a, item.get("context") or "", item.get("question") or "")
        det["bad_numbers"] = bad[:8]
        if bad:
            return Verdict(0, "number_outside_context", det)
    return Verdict(1, "ok", det)
