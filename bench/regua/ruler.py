"""
ruler.py — PASSO 2 da régua honesta: verificador binário + antitautologia.

A pergunta única (ordem do capitão): a resposta CONTÉM A INFORMAÇÃO PEDIDA?

Métrica central:
- cobertura_fatos = fração dos FATOS OBRIGATÓRIOS (gabarito_151) presentes na resposta.
  Casamento barato por termo/sinônimo (alta precisão); os fatos AMBÍGUOS (nem claramente
  presentes nem ausentes) vão a um desempate ÚNICO do juiz 27B por resposta.

Critério de aprovação:
- APROVADO = cobertura >= LIMIAR (default 0.5) E não-tautológico E sem alucinação
  que contradiga o trecho-ouro.
- CITAÇÃO NÃO ENTRA NO GATE (decisão do capitão) — é reportada só como coluna informativa.

Detector de tautologia (determinístico, obrigatório):
- resposta que só reordena os termos da PERGUNTA sem trazer NENHUM fato novo do gabarito
  -> REPROVA automaticamente. Implementado por sobreposição de n-gramas com a pergunta
  + ausência total de fatos do gabarito. Validado nos exemplos citados pelo capitão
  (ver bench/regua/test_tautology.py).

Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


def deaccent(text: str) -> str:
    """Remove acentos (NFD) preservando o texto base."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")

# Stopwords PT-BR (curta, foco em conectivos que não carregam fato).
STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "que", "em", "no", "na", "nos",
    "nas", "um", "uma", "uns", "umas", "para", "pra", "por", "com", "sem", "ao",
    "aos", "as", "os", "se", "sua", "seu", "suas", "seus", "ou", "como", "mais",
    "menos", "ser", "estar", "foi", "sao", "e", "ja", "so", "tem", "ter", "deve",
    "devem", "pode", "podem", "quando", "onde", "qual", "quais", "isso", "esse",
    "essa", "este", "esta", "isto", "aquele", "aquela", "lhe", "me", "te", "nos",
    "vos", "lhes", "meu", "minha", "dele", "dela", "num", "numa", "pelo", "pela",
    "aqui", "ali", "la", "entao", "tao", "muito", "pouco", "todo", "toda", "todos",
    "todas", "cada", "ate", "sobre", "entre", "apos", "antes", "depois", "conforme",
    "segundo", "item", "norma", "nr", "voce", "obrigatorio", "obrigatoria",
}

# Sinônimos/variantes de termos técnicos das NRs. Chave e valores são canônicos
# em minúsculas SEM acento; a presença de qualquer variante conta como o termo.
SYNONYMS: Dict[str, Set[str]] = {
    "desenergizacao": {"desenergizar", "desenergizada", "desligar", "desligamento",
                       "desligado", "seccionamento", "seccionar"},
    "paraquedista": {"paraquedas", "para-quedista", "para-quedas"},
    "trava-quedas": {"travaquedas", "trava", "antiqueda", "anti-queda"},
    "talabarte": {"talabartes"},
    "ancoragem": {"ancorar", "ancorado", "ancoradouro"},
    "cinturao": {"cinto", "cinturoes"},
    "aterramento": {"aterrar", "aterrado", "equipotencializacao", "equipotencial"},
    "epi": {"epis", "equipamento", "protecao", "individual"},
    "capacitacao": {"capacitar", "capacitado", "treinamento", "treinar", "treinado",
                    "curso", "reciclagem"},
    "sinalizacao": {"sinalizar", "sinalizado", "placa", "placas"},
    "permissao": {"pt", "autorizacao", "liberacao"},
    "analise": {"ar", "avaliacao"},
    "recusa": {"recusar", "interromper", "interrupcao", "parar", "paralisar"},
    "resgate": {"salvamento", "socorro"},
    "isolamento": {"isolacao", "isolar", "isolado", "isolante"},
    "tensao": {"voltagem", "volts", "v"},
}

# Índice reverso variante -> canônico, para expansão barata na tokenização.
_VARIANT_TO_CANON: Dict[str, str] = {}
for canon, variants in SYNONYMS.items():
    _VARIANT_TO_CANON[canon] = canon
    for v in variants:
        _VARIANT_TO_CANON[v] = canon

# Limiares de decisão determinística por fato.
PRESENT_RATIO = 0.60   # >= : fato claramente presente
ABSENT_RATIO = 0.20    # <= : fato claramente ausente; entre os dois -> ambíguo
COVERAGE_THRESHOLD = 0.50  # limiar de aprovação (calibrável)
TAUTOLOGY_QOVERLAP = 0.60  # sobreposição de conteúdo com a pergunta p/ suspeita de tautologia


def _tokens(text: str) -> List[str]:
    """Tokeniza: minúsculas, sem acento, só alfanumérico, aplica sinônimos."""
    norm = deaccent(text.lower())
    raw = re.findall(r"[a-z0-9]+", norm)
    out = []
    for t in raw:
        out.append(_VARIANT_TO_CANON.get(t, t))
    return out


def _content_tokens(text: str) -> List[str]:
    """Tokens de conteúdo: remove stopwords e ruído; mantém números e termos >=3."""
    out = []
    for t in _tokens(text):
        if t in STOPWORDS:
            continue
        if t.isdigit():
            out.append(t)
        elif len(t) >= 3:
            out.append(t)
    return out


def _stem_match(fact_tok: str, ans_set: Set[str]) -> bool:
    """Casamento por radical barato entre um token de fato e o conjunto da resposta."""
    if fact_tok in ans_set:
        return True
    if fact_tok.isdigit():
        return fact_tok in ans_set
    if len(fact_tok) < 4:
        return fact_tok in ans_set
    pref = fact_tok[:5]
    for a in ans_set:
        if a == fact_tok:
            return True
        if len(a) >= 4 and (a.startswith(pref) or fact_tok.startswith(a[:5])):
            return True
    return False


def _fact_match_ratio(fact: str, ans_tokens: List[str]) -> float:
    """Fração de tokens de conteúdo do fato presentes na resposta (0..1)."""
    ftoks = _content_tokens(fact)
    if not ftoks:
        return 0.0
    ans_set = set(ans_tokens)
    hit = sum(1 for ft in ftoks if _stem_match(ft, ans_set))
    return hit / len(ftoks)


@dataclass
class RulerResult:
    """Resultado da régua para uma (resposta, gabarito)."""
    coverage: float = 0.0
    n_facts: int = 0
    facts_present: List[str] = field(default_factory=list)
    facts_absent: List[str] = field(default_factory=list)
    facts_ambiguous: List[str] = field(default_factory=list)
    tautology: bool = False
    hallucination: bool = False
    approved: bool = False
    needs_judge: bool = False
    citation_present: bool = False  # coluna informativa (NÃO entra no gate)
    empty: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


CITATION_RE = re.compile(r"nr[-\s]?\d{1,2}", re.IGNORECASE)


def deterministic_pass(question: str, answer: str, facts: List[str],
                       threshold: float = COVERAGE_THRESHOLD) -> RulerResult:
    """Passe determinístico: cobertura barata + tautologia + necessidade de desempate."""
    r = RulerResult(n_facts=len(facts))
    ans = (answer or "").strip()
    r.citation_present = bool(CITATION_RE.search(ans))

    if not ans or len(ans) < 4:
        r.empty = True
        r.facts_absent = list(facts)
        return r

    ans_tokens = _content_tokens(ans)

    for f in facts:
        ratio = _fact_match_ratio(f, ans_tokens)
        if ratio >= PRESENT_RATIO:
            r.facts_present.append(f)
        elif ratio <= ABSENT_RATIO:
            r.facts_absent.append(f)
        else:
            r.facts_ambiguous.append(f)

    n = max(1, len(facts))
    lower_cov = len(r.facts_present) / n
    upper_cov = (len(r.facts_present) + len(r.facts_ambiguous)) / n

    # Detector de tautologia (determinístico):
    # a resposta é quase toda reordenação dos termos da pergunta E não traz
    # NENHUM fato determinístico do gabarito.
    q_content = set(_content_tokens(question))
    a_content = _content_tokens(ans)
    if a_content and q_content:
        overlap = sum(1 for t in a_content if t in q_content) / len(a_content)
    else:
        overlap = 0.0
    r.detail["q_overlap"] = round(overlap, 3)
    r.detail["lower_cov"] = round(lower_cov, 3)
    r.detail["upper_cov"] = round(upper_cov, 3)

    no_det_fact = len(r.facts_present) == 0
    if no_det_fact and overlap >= TAUTOLOGY_QOVERLAP:
        r.tautology = True

    # Precisa de desempate do juiz? Só quando a resposta PODE ser aprovada:
    # (cobertura superior alcança o limiar) e há ambíguos, ou a cobertura baixa
    # já cruza o limiar (aí basta checar alucinação).
    could_pass = (upper_cov >= threshold) and not r.tautology
    r.needs_judge = could_pass

    # Decisão preliminar (será revista pelo juiz se needs_judge).
    r.coverage = lower_cov
    r.approved = (lower_cov >= threshold) and not r.tautology
    return r
