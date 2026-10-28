"""
common_f0.py — utilitários compartilhados do baseline LFM3-1.7B (F0).

Fonte de dados: UFGCEMIGONA/benchmark_mobile, config normas_regulamentadoras
(1536 queries = 384 base_qa_id x 4 vocabulários). Usa SEMPRE chunks_512.

Células do protocolo (ordem do capitão):
  - closed_book : nenhum contexto (régua do "antes" da injeção de conhecimento)
  - oraculo     : só a passagem correct (teto de síntese)
  - ranking     : correct + 4 incorrect, ordem embaralhada com seed fixa POR id
  - distrator   : só os 4 incorrect (recusa correta vs alucinação)

Este módulo roda tanto na Spark (geração) quanto localmente (scoring):
só stdlib + pyarrow.

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

GLOBAL_SEED = 42
CELLS = ("closed_book", "oraculo", "ranking", "distrator")
VOCABS = ("clean", "technical", "operational", "lay")

# System prompt FIXO em todas as células (registrado no PREFLIGHT/README).
SYSTEM_PROMPT = (
    "Você é um assistente técnico de segurança do trabalho, especialista em "
    "Normas Regulamentadoras (NRs) brasileiras. Responda sempre em português do "
    "Brasil, de forma objetiva e curta. Quando houver contexto normativo "
    "fornecido, baseie sua resposta exclusivamente nele e cite a fonte "
    "(norma ou documento). Se o contexto fornecido não contiver a informação "
    "necessária, diga isso claramente em vez de inventar. Quando não houver "
    "contexto, responda com seu conhecimento das normas, citando a norma "
    "quando souber."
)


def load_rows(parquet_path: str | Path) -> List[Dict[str, Any]]:
    """Carrega as 1536 linhas do parquet normas_regulamentadoras/train."""
    import pyarrow.parquet as pq
    rows = pq.read_table(str(parquet_path)).to_pylist()
    assert len(rows) == 1536, f"esperava 1536 linhas, veio {len(rows)}"
    return rows


def get_chunks(row: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extrai {correct:[1], incorrect:[4]} de chunks_512 (lista com 1 struct)."""
    c = row["chunks_512"]
    c = c[0] if isinstance(c, list) else c
    assert len(c["correct"]) == 1 and len(c["incorrect"]) == 4, row["id"]
    return c


def shuffled_order(row_id: str, n: int) -> List[int]:
    """Permutação determinística POR id (seed global + id), registrada na saída."""
    seed = f"{GLOBAL_SEED}:{row_id}"
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    order = list(range(n))
    rng.shuffle(order)
    return order


def build_cell_context(row: Dict[str, Any], cell: str) -> Dict[str, Any]:
    """
    Monta a lista de passagens da célula e a ordem embaralhada.
    Retorna {"passages": [{chunk_id, source_file, text, is_correct}], "order": [...]}.
    """
    if cell == "closed_book":
        return {"passages": [], "order": []}
    c = get_chunks(row)
    if cell == "oraculo":
        pool = [dict(p, is_correct=True) for p in c["correct"]]
    elif cell == "ranking":
        pool = ([dict(p, is_correct=True) for p in c["correct"]]
                + [dict(p, is_correct=False) for p in c["incorrect"]])
    elif cell == "distrator":
        pool = [dict(p, is_correct=False) for p in c["incorrect"]]
    else:
        raise ValueError(cell)
    order = shuffled_order(row["id"], len(pool)) if len(pool) > 1 else list(range(len(pool)))
    passages = [pool[i] for i in order]
    return {"passages": passages, "order": order}


def render_user(row: Dict[str, Any], passages: List[Dict[str, Any]]) -> str:
    """Conteúdo do turno user: contexto numerado (quando houver) + pergunta."""
    if not passages:
        return f"Pergunta:\n{row['question']}"
    parts = []
    for i, p in enumerate(passages, 1):
        src = p["source_file"].replace(".pdf", "")
        parts.append(f"[{i}] ({src}):\n{p['text']}")
    ctx = "\n\n".join(parts)
    return (f"Contexto normativo consultado:\n{ctx}\n\n"
            f"Pergunta:\n{row['question']}")


def build_messages(row: Dict[str, Any], cell: str) -> Dict[str, Any]:
    """Mensagens chat + metadados de auditoria (ordem do embaralhamento etc.)."""
    ctx = build_cell_context(row, cell)
    user = render_user(row, ctx["passages"])
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
        "shuffle_order": ctx["order"],
        "passage_ids": [p["chunk_id"] for p in ctx["passages"]],
        "correct_position": next(
            (i for i, p in enumerate(ctx["passages"]) if p["is_correct"]), None),
    }


# ---------------------------------------------------------------------------
# Separação thinking/resposta (LFM3 é modelo de raciocínio; template abre <think>).
# ---------------------------------------------------------------------------
def split_thinking(generated: str) -> Dict[str, str]:
    """Separa o rascunho <think>...</think> da resposta final."""
    if "</think>" in generated:
        think, _, answer = generated.partition("</think>")
        return {"thinking": think.replace("<think>", "").strip(),
                "answer": answer.strip()}
    # sem fechamento: tudo é thinking truncado, resposta vazia
    return {"thinking": generated.replace("<think>", "").strip(), "answer": ""}


# ---------------------------------------------------------------------------
# Detecção de recusa — portada de sft_v2/common_v2.py (padrões validados na POC).
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    r"n[aã]o (está|esta|consta|traz|aborda|responde|encontr|foi encontr)",
    r"n[aã]o (sei|posso (afirmar|orientar|garantir|responder|dizer))",
    r"n[aã]o (h[aá]|existe|disp)",
    r"nas normas consultad", r"no(s)? trecho(s)? consultad",
    r"n[aã]o est[aá] nas normas", r"n[aã]o trata (disso|desse|do que)",
    r"seria preciso (consultar|verificar)", r"trata de outro (assunto|tema)",
    r"n[aã]o (é|e) poss[ií]vel (afirmar|responder|orientar)",
    r"contexto (não|nao) (responde|traz|cobre|aborda|cont[eé]m)",
    r"fora do (que|assunto)", r"n[aã]o (cobre|contempla)",
    r"contexto fornecido n[aã]o", r"n[aã]o cont[eé]m (a|essa|esta) informa",
]
REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def looks_like_refusal(answer: str) -> bool:
    """Heurística determinística de recusa/delimitação."""
    a = (answer or "").strip()
    if len(a) < 4:
        return True
    return bool(REFUSAL_RE.search(a))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
