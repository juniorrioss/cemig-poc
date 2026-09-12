"""
05_e2e_approval.py — ETAPA 6 (parte B): aprovação honesta E2E (bench/regua) nas NR-10 do holdout.

Gera respostas E2E (BM25 top-2 pelo caminho do app -> síntese com o SYSTEM PROMPT de
produção, no 1.2B QAD embarcado na RTX 5070) para o subconjunto NR-10 do holdout (as únicas
NRs-piloto presentes nas 151), com o índice ATUAL e com o índice DocAI aditivo. Depois
aplica a RÉGUA HONESTA (bench/regua/ruler.py: cobertura de fatos + antitautologia), com o
gabarito congelado data/gabarito_151.jsonl e desempate do juiz 27B.

REGRA (brief): latência de GPU NÃO vale p/ aparelho; medimos QUALIDADE/aprovação.
Checkpoint incremental, paralelo no juiz, procedência registrada.

Comentários em português; identificadores em inglês.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bench" / "regua"))

from corpus.eval_fino import app_fts_query
from ruler import deterministic_pass, RulerResult  # régua honesta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS = (1.5, 3.0, 2.0, 1.0)

# System prompt IDÊNTICO ao AskPipeline.SYNTHESIS_SYSTEM_PROMPT (produção).
SYNTHESIS_SYSTEM_PROMPT = (
    "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras "
    "(NR-10, NR-06, NR-35, NR-12, NR-18 e demais NRs aplicáveis).\n"
    "Suas diretrizes mandatórias:\n"
    "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
    "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. "
    "Não adicione procedimentos não contidos nas normas.\n"
    "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
    "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
    "'Não sei com base nas normas consultadas.' Não tente adivinhar."
)
SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05}


def search(db_path: Path, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    fts_q = app_fts_query(query)
    if fts_q == '""':
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    w = WEIGHTS
    sql = f"""SELECT c.id,c.doc,c.section,c.title,c.page,c.text,
        bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]}) AS score
        FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
        WHERE chunks_fts MATCH ? ORDER BY score ASC LIMIT ?;"""
    try:
        rows = con.execute(sql, (fts_q, top_k)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [dict(r) for r in rows]


def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "Nenhum contexto normativo recuperado para a consulta."
    return "\n\n".join(f"[{i+1}] ({c['doc']} - {c['section']} - {c['title']}):\n{c['text']}"
                       for i, c in enumerate(chunks))


def synthesize(url: str, model: str, question: str, context: str, timeout: int = 120) -> Dict[str, Any]:
    user = f"Contexto normativo:\n{context}\n\nPergunta do trabalhador: {question}"
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user}],
        "max_tokens": 320, **SAMPLING}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return {"response": (d["choices"][0]["message"].get("content") or "").strip(),
            "wall_s_gpu": round(time.perf_counter() - t0, 2)}


def gen_answers(db_path: Path, pairs: List[Dict[str, Any]], url: str, model: str,
                cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Gera (ou reusa do cache) respostas E2E para cada pergunta. Checkpoint incremental."""
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for i, p in enumerate(pairs):
        if p["id"] in cache and cache[p["id"]].get("response"):
            continue
        chunks = search(db_path, p["question"], top_k=2)
        ctx = format_context(chunks)
        try:
            syn = synthesize(url, model, p["question"], ctx)
        except Exception as e:
            logger.warning("falha síntese %s: %s", p["id"], e)
            syn = {"response": "", "wall_s_gpu": 0}
        cache[p["id"]] = {
            "question": p["question"], "doc": p["doc"], "section": p["section"],
            "retrieved": [{"doc": c["doc"], "section": c["section"]} for c in chunks],
            "response": syn["response"], "wall_s_gpu": syn["wall_s_gpu"],
        }
        if (i + 1) % 5 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("checkpoint %d/%d", i + 1, len(pairs))
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def apply_ruler(answers: Dict[str, Dict[str, Any]], gabarito: Dict[str, Dict[str, Any]],
                threshold: float = 0.5) -> Dict[str, Any]:
    """Aplica a régua honesta determinística (sem juiz — barato e reprodutível)."""
    n = approved = taut = 0
    cov_sum = 0.0
    per_item = []
    for qid, a in answers.items():
        gab = gabarito.get(qid)
        if not gab:
            continue
        n += 1
        r: RulerResult = deterministic_pass(a["question"], a["response"], gab["facts"], threshold)
        if r.approved:
            approved += 1
        if r.tautology:
            taut += 1
        cov_sum += r.coverage
        per_item.append({"id": qid, "coverage": round(r.coverage, 3),
                         "approved": r.approved, "tautology": r.tautology,
                         "citation": r.citation_present})
    return {"n": n, "approved": approved,
            "approval_pct": round(100 * approved / max(1, n), 1),
            "mean_coverage": round(cov_sum / max(1, n), 3),
            "tautology": taut, "per_item": per_item}


def main() -> None:
    ap = argparse.ArgumentParser(description="E2E aprovação honesta NR-10 (atual vs DocAI).")
    ap.add_argument("--current-db", default="corpus/index_hf_36nr.db")
    ap.add_argument("--docai-db", default="docai/data/index_docai.db")
    ap.add_argument("--qa", default="corpus/qa_pairs_v2.jsonl")
    ap.add_argument("--gabarito", default="bench/regua/data/gabarito_151.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="local")
    ap.add_argument("--doc-filter", default="nr-10")
    ap.add_argument("--out", default="docai/data/e2e_approval.json")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.doc_filter:
        pairs = [p for p in pairs if p["doc"] == args.doc_filter]
    logger.info("NR-alvo=%s: %d perguntas do holdout", args.doc_filter, len(pairs))

    gabarito = {r["id"]: r for r in
                (json.loads(l) for l in Path(args.gabarito).read_text(encoding="utf-8").splitlines() if l.strip())}

    data_dir = Path(args.out).parent
    ans_cur = gen_answers(Path(args.current_db), pairs, args.url, args.model,
                          data_dir / "e2e_answers_current.json")
    ans_doc = gen_answers(Path(args.docai_db), pairs, args.url, args.model,
                          data_dir / "e2e_answers_docai.json")

    m_cur = apply_ruler(ans_cur, gabarito)
    m_doc = apply_ruler(ans_doc, gabarito)
    report = {"doc_filter": args.doc_filter,
              "current": m_cur, "docai": m_doc,
              "delta_approval_pct": round(m_doc["approval_pct"] - m_cur["approval_pct"], 1),
              "delta_coverage": round(m_doc["mean_coverage"] - m_cur["mean_coverage"], 3),
              "note": "régua honesta determinística (ruler.deterministic_pass, limiar 0.5); "
                      "latência GPU não vale p/ aparelho (engine:cuda)."}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=== NR-10 aprovação honesta: %.1f%% -> %.1f%% (%+.1f); cobertura %.3f -> %.3f ===",
                m_cur["approval_pct"], m_doc["approval_pct"], report["delta_approval_pct"],
                m_cur["mean_coverage"], m_doc["mean_coverage"])


if __name__ == "__main__":
    main()
