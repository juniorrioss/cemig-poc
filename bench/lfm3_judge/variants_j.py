"""
variants_j.py — variantes sintéticas de qualidade CONHECIDA (verdade por construção).

Evolui o `vela_judge/variants.py` do jaopredos/eval-vela. Fragilidades do original
(docs/relatorio_projeto_vela_judge.md + inspeção do variants.jsonl publicado) e correção:
  - poucas variantes DIFÍCEIS: wrong_number 13, wrong_item 9 em 549 (o AUROC alto vinha
    de gold×off_topic/refusal, triviais). Aqui: amostra maior (N perguntas) e defeitos
    sutis com regex mais amplo (números por extenso, 'NR 10', itens 10.2.8.1);
  - novas variantes de defeito que o nosso pipeline produz de verdade: `negation`
    (inverte uma obrigação), `swap_obligation` (deve -> pode/é facultativo),
    `tautology` (repete a pergunta sem fato — o defeito "oco" da régua honesta),
    `refusal` em VÁRIAS redações (o original tinha 1 frase fixa = viés de memorização);
  - `extra_true` (gold + frase VERDADEIRA de outra evidência da mesma pergunta) para
    separar "acrescentou algo" de "alucinou" (o original só tinha acréscimo falso);
  - `off_topic` da MESMA norma (mais difícil que gold aleatório de outra norma).
Contexto das variantes = evidência-ouro (oráculo), como no original, para isolar o eixo.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

REFUSALS = [
    "Não encontrei essa informação nos documentos disponíveis.",
    "Isso não está nos trechos das normas que recuperei. Vale consultar o engenheiro de segurança antes de prosseguir.",
    "Os trechos fornecidos não trazem essa informação, então não consigo responder com segurança.",
    "Não sei responder com base nas normas disponíveis.",
]
HALLUCINATION_BANK = [
    "Adicionalmente, a empresa deve renovar esse procedimento junto ao órgão regional do MTE a cada 24 meses.",
    "Cabe destacar que o não cumprimento desse item gera multa automática de 3 salários mínimos por trabalhador exposto.",
    "Vale ressaltar que essa exigência só se aplica a estabelecimentos com mais de 500 empregados.",
    "Além disso, é obrigatória a certificação do responsável técnico por entidade internacional reconhecida pelo Inmetro.",
    "Essa medida deve ser revalidada por auditoria externa independente a cada 6 meses.",
    "Soma-se a isso a exigência de comunicação prévia ao sindicato da categoria com 30 dias de antecedência.",
    "Essa obrigação se estende também aos terceirizados, que devem portar certificado próprio emitido pela CIPA.",
    "Essa exigência substitui integralmente a necessidade de treinamento presencial.",
]
NUM_WORDS = {"um": "dois", "uma": "duas", "dois": "três", "duas": "três", "três": "cinco",
             "quatro": "seis", "cinco": "oito", "seis": "dez", "dez": "quinze", "doze": "vinte",
             "quinze": "trinta", "trinta": "sessenta", "sessenta": "noventa"}
NR_ITEM_RE = re.compile(r"NR[-\s]?\d{1,2}\b|\b\d{1,2}(?:\.\d{1,2}){1,4}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
SENT_RE = re.compile(r"(?<=[.!?;])\s+")
OBLIG_SWAPS = [
    (r"\bdeve(m)? ser\b", lambda m: "pode" + ("m" if m.group(1) else "") + " ser, a critério do empregador,"),
    (r"\bdever[áa]o?\b", lambda m: "poderá"),
    (r"\bé obrigatóri[oa]\b", lambda m: "é facultativo"),
    (r"\bdeve(m)?\b", lambda m: "pode" + ("m" if m.group(1) else "")),
]
NEG_SWAPS = [
    (r"\bnão deve(m)?\b", lambda m: "deve" + ("m" if m.group(1) else "")),
    (r"\bé proibid[oa]\b", lambda m: "é permitido"),
    (r"\bé vedad[oa]\b", lambda m: "é permitido"),
    (r"\bdeve(m)?\b", lambda m: "não deve" + ("m" if m.group(1) else "")),
    (r"\bé permitid[oa]\b", lambda m: "é proibido"),
    (r"\bé obrigatóri[oa]\b", lambda m: "não é necessário"),
]


def _sub_first(patterns, text: str) -> Optional[str]:
    for pat, fn in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return text[:m.start()] + fn(m) + text[m.end():]
    return None


def _has_gloss(gold: str, end: int) -> bool:
    """Número seguido de extenso entre parênteses ('180 (cento e oitenta)'): trocar só o
    dígito deixaria pista inconsistente (o original do eval-vela já evitava)."""
    near = gold[end:end + 20]
    i = near.find("(")
    if i < 0:
        return False
    j = gold.find(")", end + i)
    return j > 0 and not re.search(r"\d", gold[end + i + 1:j])


def wrong_number(gold: str, rng: random.Random) -> Optional[str]:
    items = [m.span() for m in NR_ITEM_RE.finditer(gold)]
    cands = [m for m in NUMBER_RE.finditer(gold)
             if not any(a <= m.start() < b for a, b in items) and not _has_gloss(gold, m.end())]
    if cands:
        m = rng.choice(cands)
        v = float(m.group().replace(",", "."))
        nv = v + rng.choice([-1, 1]) * max(1, round(v * rng.uniform(0.3, 0.8)))
        if nv <= 0 or nv == v:
            nv = v * 2 + 1
        s = str(int(nv)) if v == int(v) and nv == int(nv) else f"{nv:.2f}".rstrip("0").rstrip(".").replace(".", ",")
        return gold[:m.start()] + s + gold[m.end():]
    for w, nw in NUM_WORDS.items():  # número por extenso
        m = re.search(rf"\b{w}\b", gold)
        inside_paren = m and gold.rfind("(", 0, m.start()) > gold.rfind(")", 0, m.start())
        if m and w not in ("um", "uma") and not inside_paren:
            return gold[:m.start()] + nw + gold[m.end():]
    return None


def wrong_item(gold: str, rng: random.Random) -> Optional[str]:
    ms = list(NR_ITEM_RE.finditer(gold))
    if not ms:
        return None
    m = rng.choice(ms)
    o = m.group()
    if o.upper().startswith("NR"):
        d = re.search(r"\d+", o).group()
        n = str(rng.choice([x for x in range(1, 38) if x != int(d)]))
        new = o.replace(d, n)
    else:
        parts = o.split(".")
        i = rng.randrange(len(parts))
        parts[i] = str(rng.choice([x for x in range(1, 10) if str(x) != parts[i]]))
        new = ".".join(parts)
    return gold[:m.start()] + new + gold[m.end():]


def tautology(question: str) -> str:
    q = re.sub(r"^\W*(é\.\.\.|então|olha|tipo|queria saber( assim)?,?|uma dúvida:?)\s*", "", question.strip(), flags=re.I)
    q = q.rstrip("?").strip()
    return f"Sobre a sua pergunta — {q[0].lower() + q[1:]} — é preciso seguir o que a norma determina e respeitar as exigências de segurança aplicáveis."


def build_variants(rows: List[Dict[str, Any]], n: int, seed: int,
                   paraphrases: Dict[str, str]) -> List[Dict[str, Any]]:
    """rows: linhas do benchmark (id, question, answer, evidence[], source_file, facts...)."""
    rng0 = random.Random(seed)
    # prioriza os ids que têm paráfrase do eval-vela (paráfrase humana-LLM validada),
    # completa com amostra estratificada aleatória.
    with_para = [r for r in rows if r["id"] in paraphrases]
    rest = [r for r in rows if r["id"] not in paraphrases]
    rng0.shuffle(rest)
    chosen = (with_para + rest)[:n]
    by_norm: Dict[str, List[str]] = {}
    for r in rows:
        by_norm.setdefault(r["source_file"].split("_")[0].split("-atual")[0], []).append(r["answer"])
    out = []
    for r in chosen:
        rng = random.Random(f"{seed}:{r['id']}")
        gold = r["answer"]
        ev_texts = [e["text"] for e in r["evidence"] if e.get("text")]
        evidence = "\n\n".join(ev_texts)
        sents = [s for s in SENT_RE.split(gold.strip()) if s]
        norm = r["source_file"].split("_")[0].split("-atual")[0]
        same_norm = [g for g in by_norm.get(norm, []) if g != gold]
        extra = None
        gold_toks = set(re.findall(r"\w+", gold.lower()))
        for e in ev_texts:  # frase VERDADEIRA da evidência pouco presente no gold
            for s in SENT_RE.split(e):
                st = set(re.findall(r"\w+", s.lower()))
                if len(s) > 60 and st and len(st & gold_toks) / len(st) < 0.5:
                    extra = s.strip()
                    break
            if extra:
                break
        v = {
            "gold": gold,
            "paraphrase": paraphrases.get(r["id"]),
            "wrong_number": wrong_number(gold, rng),
            "wrong_item": wrong_item(gold, rng),
            "negation": _sub_first(NEG_SWAPS, gold),
            "swap_obligation": _sub_first(OBLIG_SWAPS, gold),
            "incomplete": " ".join(sents[:max(1, len(sents) // 2)]) if len(sents) >= 2 else None,
            "hallucination_added": gold.rstrip() + " " + rng.choice(HALLUCINATION_BANK),
            "extra_true": (gold.rstrip() + " " + extra) if extra else None,
            "refusal": rng.choice(REFUSALS),
            "tautology": tautology(r["question"]),
            "off_topic": rng.choice(same_norm) if same_norm else None,
        }
        for vt, ans in v.items():
            if not ans or (vt != "gold" and ans.strip() == gold.strip()):
                continue
            out.append({
                "uid": f"V:{r['id']}:{vt}", "set": "V", "qid": r["id"], "base_qa_id": r["base_qa_id"],
                "variant": vt, "question": r["question"], "gold": gold, "evidence": evidence,
                "context": f"[Arquivo: {r['source_file']}]\n{evidence}", "answer": ans,
                "facts": r.get("facts") or [], "vocabulary_level": r["vocabulary_level"],
                "question_type": r["question_type"], "source_file": r["source_file"],
            })
    return out
