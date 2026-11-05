"""
rubric_j.py — rubrica de DECISÃO adaptada à régua honesta da POC (eixos portáveis).

Evolução da rubrica do jaopredos/eval-vela (11 eixos para comparar retrievers) para o
NOSSO cenário: julgar RESPOSTAS do pipeline com as perguntas da régua honesta —
"a resposta contém a informação pedida?", alucinação, recusa, citação (informativa).

ASSIMETRIA DOCUMENTADA (adendo do capitão, verificado no código dos dois modelos):
  - pplx-decider-v1.1-27b (`autojev.types.Question`) só aceita choice | noul | score —
    NÃO existe `span`. A cabeça é um readout [255,5120] sobre códigos de opção.
  - vela3-flash aceita choice | noul | score | span (span adapter + head próprios).
  => Os eixos abaixo são TODOS choice/noul/score e rodam IDÊNTICOS nos dois deciders.
     `span` entra só no Vela, como eixo EXTRA (spans 'unsupported' da resposta vs
     trecho-ouro, e evidência citável), e é reportado à parte (não entra no kappa).

Cada eixo: tipo, state (campos do item), paráfrases (>=2 quando o eixo decide o gate,
para medir spread — fragilidade apontada no relatório do eval-vela), mapeamento da saída
para nota em [0,1] (1 = bom) e quais variantes sintéticas devem pontuar alto/baixo.

Correções de fragilidades do eval-vela incorporadas:
  (1) `refusal` não é mais "boa" em correctness por acidente: o eixo `verdict` tem opção
      explícita de recusa e a régua trata recusa como não-aprovada (igual ao Qwen);
  (2) cobertura por FATO (noul por fato do gabarito_384) em vez de choice de 4 níveis
      com valores arbitrários — é a mesma pergunta binária da régua honesta, o decision
      model responde p(fato presente) e a cobertura é a média (sem threshold mágico);
  (3) alucinação é julgada contra o TRECHO-OURO (evidência), não contra o contexto
      recuperado (o eval-vela misturava: grounding vs contexto de retriever ruim
      penalizava a resposta pelo erro do retriever);
  (4) variantes difíceis e numerosas (ver variants_j.py), não só gold/off_topic triviais.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Opções do veredito (ordem = do melhor para o pior). Valores = nota esperada.
VERDICT_OPTIONS = {
    "correta": ("A resposta traz a informação pedida e está de acordo com a referência", 1.0),
    "parcialmente_correta": ("Traz parte do que foi pedido, ou um detalhe secundário errado", 0.5),
    "incorreta": ("O fato principal está errado, ausente, ou a resposta não responde ao que foi perguntado", 0.0),
    "recusa": ("A resposta diz que não sabe ou que a informação não está disponível, sem responder", 0.0),
}

AXES: Dict[str, Dict[str, Any]] = {
    # --- Gate principal (≈ approved da régua honesta) ---------------------------------
    "verdict": {
        "type": "choice",
        "state": ("pergunta", "referencia", "trecho_oficial", "resposta"),
        "paraphrases": [
            "Compare a RESPOSTA com a REFERÊNCIA e o TRECHO OFICIAL da norma. A resposta dá ao trabalhador a informação pedida, corretamente?",
            "Avalie a resposta como um auditor de Normas Regulamentadoras: ela está correta frente à referência e ao trecho oficial?",
        ],
        "options": VERDICT_OPTIONS,
        "good": ["gold", "paraphrase", "extra_true"],
        "bad": ["wrong_number", "wrong_item", "negation", "off_topic", "refusal",
                "tautology", "hallucination_added", "swap_obligation"],
    },
    # --- Alucinação contra o trecho-ouro (régua: 'contradiz a norma ou inventa') -----
    "hallucination": {
        "type": "noul",
        "invert": True,  # p(sim)=alucina -> nota = 1 - p
        "state": ("pergunta", "trecho_oficial", "resposta"),
        "paraphrases": [
            "A resposta afirma algo que CONTRADIZ o trecho oficial da norma ou inventa uma exigência, número ou procedimento que o trecho não ampara? Omitir informação não conta.",
            "Existe na resposta alguma afirmação errada ou inventada em relação ao trecho oficial (número, prazo, item, obrigação)? Ignore omissões.",
        ],
        "good": ["gold", "paraphrase", "extra_true", "incomplete", "refusal", "tautology"],
        "bad": ["wrong_number", "wrong_item", "negation", "hallucination_added", "swap_obligation"],
    },
    # --- Recusa (diagnóstico, cruza com 'ouro no contexto') --------------------------
    "refusal": {
        "type": "noul",
        "invert": True,  # p(sim)=recusou -> nota de 'respondeu' = 1 - p
        "state": ("pergunta", "resposta"),
        "paraphrases": [
            "A resposta se recusa a responder ou diz que a informação não está disponível, em vez de responder à pergunta?",
        ],
        "good": ["gold", "paraphrase", "extra_true", "wrong_number", "wrong_item", "negation",
                 "hallucination_added", "off_topic", "incomplete", "swap_obligation"],
        "bad": ["refusal"],
    },
    # --- Relevância --------------------------------------------------------------------
    "answers_question": {
        "type": "noul",
        "state": ("pergunta", "resposta"),
        "paraphrases": ["A resposta trata diretamente do que foi perguntado e traz alguma informação nova além de repetir a pergunta?"],
        "good": ["gold", "paraphrase", "extra_true", "wrong_number", "wrong_item", "incomplete"],
        "bad": ["off_topic", "refusal", "tautology"],
    },
    # --- Fundamentação no contexto RECUPERADO (diagnóstico do retriever x leitor) -----
    "grounded": {
        "type": "noul",
        "state": ("contexto", "resposta"),
        "paraphrases": ["Todas as afirmações técnicas da resposta estão apoiadas nos trechos do contexto fornecido?"],
        "good": ["gold", "paraphrase", "extra_true", "incomplete"],
        "bad": ["hallucination_added", "wrong_number", "wrong_item", "negation", "swap_obligation", "off_topic"],
    },
    # --- Citação (INFORMATIVA, fora do gate — decisão do capitão) ---------------------
    "citation": {
        "type": "noul",
        "state": ("pergunta", "trecho_oficial", "resposta"),
        "paraphrases": ["A resposta cita explicitamente a norma (NR) ou o item que fundamenta a informação?"],
        "good": [], "bad": [],
    },
    # --- Contexto suficiente: avalia o RETRIEVER, sem resposta (tarefa extra) ---------
    "context_sufficient": {
        "type": "noul",
        "state": ("pergunta", "referencia", "contexto"),
        "paraphrases": [
            "O contexto recuperado contém a informação necessária para chegar à resposta de referência?",
            "Usando apenas o contexto recuperado, é possível dar a resposta de referência?",
        ],
        "good": [], "bad": [],
    },
}

# Pergunta por FATO (cobertura da régua honesta): 1 noul por fato do gabarito.
FACT_QUESTION = ("A resposta comunica ao trabalhador este fato (mesmo com outras palavras)? "
                 "FATO: {fact}")

# Conjuntos de eixos por uso.
ANSWER_AXES = ["verdict", "hallucination", "refusal", "answers_question", "grounded", "citation"]
CONTEXT_AXES = ["context_sufficient"]


def build_state(axis: str, item: Dict[str, Any]) -> Dict[str, str]:
    """State JSON com nomes PT-BR (o decider trata state como dado, não instrução)."""
    m = {
        "pergunta": item["question"],
        "referencia": item.get("gold") or "",
        "trecho_oficial": (item.get("evidence") or "")[:3000],
        "resposta": (item.get("answer") or "")[:2500],
        "contexto": (item.get("context") or "")[:7000],
    }
    return {k: m[k] for k in AXES[axis]["state"]}


def axis_rows(axis: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Linhas {state, question} de um eixo (uma por paráfrase)."""
    cfg = AXES[axis]
    state = build_state(axis, item)
    rows = []
    for i, instr in enumerate(cfg["paraphrases"]):
        if cfg["type"] == "choice":
            q = {"type": "choice", "instructions": instr,
                 "criteria": {k: v[0] for k, v in cfg["options"].items()}}
        elif cfg["type"] == "noul":
            q = {"type": "noul", "instructions": instr}
        else:
            q = {"type": "score", "instructions": instr, "criteria": cfg["criteria"]}
        rows.append({"key": f"{item['uid']}|{axis}|p{i}", "state": state, "question": q})
    return rows


def fact_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    state = {"pergunta": item["question"], "resposta": (item.get("answer") or "")[:2500]}
    return [{"key": f"{item['uid']}|fact|{j}", "state": state,
             "question": {"type": "noul", "instructions": FACT_QUESTION.format(fact=f)}}
            for j, f in enumerate(item.get("facts") or [])]


def span_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """EIXO EXTRA, só vela3-flash: spans 'unsupported' da resposta contra o TRECHO-OURO
    (rótulo nativo do span adapter, sem instructions — usa a formulação treinada)."""
    if not item.get("answer"):
        return []
    state = {"context": f"Pergunta: {item['question']}\n\nContexto:\n{(item.get('evidence') or '')[:3000]}",
             "answer": (item.get("answer") or "")[:2500]}
    return [{"key": f"{item['uid']}|span_unsupported|p0", "state": state,
             "question": {"type": "span", "criteria": {"unsupported": None}}}]


def read_span_unsupported(ans: Dict[str, Any], answer_text: str) -> float | None:
    """Nota = 1 - fração de caracteres da resposta marcados como 'unsupported'."""
    if ans is None or ans.get("error") or not answer_text:
        return None
    iv = sorted((s["start"], s["end"]) for s in ans.get("spans", []) if s["end"] > s["start"])
    tot, cs, ce = 0, None, None
    for a, b in iv:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            tot += ce - cs
            cs, ce = a, b
    if cs is not None:
        tot += ce - cs
    return 1.0 - min(1.0, tot / len(answer_text[:2500]))


def read_answer(axis: str, ans: Dict[str, Any]) -> float | None:
    """Saída bruta do decider -> nota [0,1] (1 = bom)."""
    if ans is None or ans.get("error"):
        return None
    cfg = AXES.get(axis)
    if axis == "fact":
        return float(ans["noul"])
    if cfg["type"] == "noul":
        p = float(ans["noul"])
        return 1.0 - p if cfg.get("invert") else p
    if cfg["type"] == "choice":
        probs = ans["probabilities"]
        tot = sum(probs.get(k, 0.0) for k in cfg["options"])
        return sum(probs.get(k, 0.0) * v[1] for k, v in cfg["options"].items()) / max(tot, 1e-12)
    raise ValueError(axis)
