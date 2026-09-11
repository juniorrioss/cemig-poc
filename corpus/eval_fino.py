"""
eval_fino.py — Avaliação Fase 1 (M3): retrieval no índice FINO com rewrite zero-shot.

Mede recall@1/2/3/5 e MRR nas 151 perguntas coloquiais reais (qa_pairs_v2.jsonl —
conjunto NÃO-contaminado) sobre o índice de chunking fino (index_hf_36nr_fino.db).

O rewrite zero-shot é gerado pelo MESMO modelo de produção (LFM2.5-1.2B) via
llama-server local, replicando fielmente o REWRITE_SYSTEM_PROMPT do AskPipeline.kt.
As reescritas são cacheadas em disco (JSON) para reprodutibilidade e para não
depender do servidor em re-execuções.

Estratégias de mitigação de dominância avaliadas:
  - raw          : pergunta bruta (voz) OR, sem rewrite
  - rw_or        : rewrite zero-shot, OR, sem boost
  - rw_boost     : rewrite zero-shot + boost de campo doc/section (bm25 5,5,2,1)
  - rw_normboost : rewrite + boost suave (soma) da norma detectada no rewrite
  - rw_2layer    : índice 2 camadas (5 NRs críticas priorizadas + resto penalizado)

Uso:
    python3 -m corpus.eval_fino --db corpus/index_hf_36nr_fino.db \
        --qa corpus/qa_pairs_v2.jsonl --cache corpus/data/rewrites_zeroshot.json
"""

import argparse
import json
import logging
import re
import sqlite3
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from corpus.build_index import format_fts_query
from corpus.eval_retrieval import EvalMetrics, check_hit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Replica fiel do REWRITE_SYSTEM_PROMPT de AskPipeline.kt (produção)
REWRITE_SYSTEM_PROMPT = (
    "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n"
    "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n"
    "Dado o histórico da conversa e a nova fala do trabalhador, gere de 3 a 6 palavras-chave técnicas para busca.\n"
    "Diretrizes mandatórias:\n"
    "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n"
    "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n"
    "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n"
    "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."
)

# Conjunto das 5 NRs críticas priorizadas na estratégia de 2 camadas
CORE_5_NRS = {"nr-06", "nr-10", "nr-12", "nr-18", "nr-35"}

# Stopwords idênticas ao Fts5Retriever.kt (produção) — inclui ruído de domínio
APP_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos",
    "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas",
    "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", "foram",
    "ha", "há", "isso", "isto", "ja", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo",
    "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nao", "não", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
    "se", "seja", "sem", "so", "só", "sua", "suas", "seu", "seus", "tambem", "também",
    "te", "tem", "têm", "temos", "ter", "teu", "teus", "tua", "tuas", "um", "uma",
    "voce", "você", "voces", "vocês",
    "norma", "normas", "regulamentar", "regulamentares", "regulamentaria", "regulamentarias",
    "seguranca", "segurança", "trabalho", "item", "artigo",
}


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def app_fts_query(raw_query: str) -> str:
    """Replica fielmente Fts5Retriever.sanitizeFts5Query do app (stem 6 + prefixo* + OR)."""
    normalized = _strip_accents(raw_query).lower()
    raw_tokens = re.findall(r'[\w.-]+', normalized)
    filtered = [t for t in raw_tokens if t not in APP_STOPWORDS and len(t) > 2]
    if not filtered:
        filtered = [t for t in raw_tokens if len(t) > 2]
    if not filtered:
        return '""'
    parts = []
    for tok in filtered:
        stem = tok[:6] if len(tok) > 6 else tok
        if "-" in stem or "." in stem:
            parts.append(f'"{stem}"*')
        else:
            parts.append(f'{stem}*')
    return " OR ".join(parts)


# Prompt few-shot melhorado: força código de NR + substantivos técnicos em uma linha
FEWSHOT_SYSTEM_PROMPT = (
    "Você converte a fala de um eletricista/operário em uma consulta de busca técnica nas Normas Regulamentadoras (NRs).\n"
    "Responda em UMA linha, SOMENTE com: o código da NR mais provável seguido de 4 a 6 substantivos técnicos.\n"
    "Não explique, não use bullets, não escreva frases. Formato: nr-XX termo1 termo2 termo3 termo4\n\n"
    "Exemplos:\n"
    "Fala: Tô na subestação e não dá pra desligar o circuito, posso usar só a luva de borracha?\n"
    "nr-10 proteção coletiva desenergização tensão segurança isolação\n"
    "Fala: Minha bota furou trabalhando na chuva, a empresa é obrigada a trocar?\n"
    "nr-06 EPI calçado fornecimento gratuito conservação substituição\n"
    "Fala: Vou subir num andaime de 8 metros, preciso de cinto?\n"
    "nr-35 trabalho altura cinto paraquedista ancoragem proteção queda\n"
)


def llm_rewrite(question: str, base_url: str, model: str = "local", timeout: int = 120, fewshot: bool = False, reasoning: bool = False) -> str:
    """Gera a reescrita zero-shot via llama-server (endpoint OpenAI-compatível)."""
    if fewshot:
        messages = [
            {"role": "system", "content": FEWSHOT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Fala: {question}\n"},
        ]
    else:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Fala do trabalhador: {question}\n\nPalavras-chave técnicas:"},
        ]
    # Modelos de raciocínio (LFM2.5-2.6B) precisam de orçamento maior p/ concluir o pensamento
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1200 if reasoning else 64,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    # Alguns modelos de raciocínio colocam a saída útil em reasoning_content
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return content


def clean_keywords(raw: str, fallback: str) -> str:
    """Higieniza a saída do rewrite em termos de busca (replica cleanKeywords do app)."""
    if not raw or not raw.strip():
        return fallback
    text = raw.strip()
    # Remove eventuais rótulos ou markdown
    text = re.sub(r'(?i)^(palavras-chave|termos|busca)\s*[:\-]\s*', '', text)
    text = text.replace("*", " ").replace("`", " ").replace("\n", " ")
    # Mantém apenas a primeira linha útil
    text = text.split("\n")[0].strip()
    if not text:
        return fallback
    return text


def build_rewrite_cache(
    qa_pairs: List[Dict[str, Any]],
    base_url: str,
    cache_path: Path,
    model: str = "local",
    fewshot: bool = False,
    reasoning: bool = False,
) -> Dict[str, str]:
    """Gera/carrega o cache de reescritas zero-shot com checkpoint incremental."""
    cache: Dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Cache de rewrite carregado: %d entradas em %s", len(cache), cache_path)

    dirty = False
    for i, p in enumerate(qa_pairs):
        qid = p["id"]
        if qid in cache and cache[qid]:
            continue
        try:
            raw = llm_rewrite(p["question"], base_url, model=model, fewshot=fewshot, reasoning=reasoning)
            cache[qid] = clean_keywords(raw, p["question"])
            dirty = True
        except Exception as e:
            logger.warning("Falha no rewrite de %s: %s", qid, e)
            cache[qid] = p["question"]  # fallback pergunta bruta
            dirty = True
        # Checkpoint incremental a cada 10 itens (VPN oscila)
        if dirty and (i + 1) % 10 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Checkpoint rewrite: %d/%d", i + 1, len(qa_pairs))
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def detect_norm(text: str) -> Optional[str]:
    """Detecta uma norma explícita (nr-XX) mencionada na fala/rewrite, se houver."""
    m = re.search(r'\bnr[-\s]?(\d{1,2})\b', text.lower())
    if m:
        return f"nr-{int(m.group(1)):02d}"
    return None


def search(
    con: sqlite3.Connection,
    fts_q: str,
    limit: int = 5,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    doc_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Executa a busca BM25 no FTS5 com pesos de campo opcionais."""
    if fts_q == '""':
        return []
    cur = con.cursor()
    w = weights
    if doc_filter:
        sql = f"""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts, {w[0]}, {w[1]}, {w[2]}, {w[3]}) AS score
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE c.doc = ? AND chunks_fts MATCH ?
            ORDER BY score ASC LIMIT ?;
        """
        try:
            cur.execute(sql, (doc_filter, fts_q, limit))
        except sqlite3.OperationalError as e:
            logger.warning("FTS erro '%s': %s", fts_q, e)
            return []
    else:
        sql = f"""
            SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
                   bm25(chunks_fts, {w[0]}, {w[1]}, {w[2]}, {w[3]}) AS score
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC LIMIT ?;
        """
        try:
            cur.execute(sql, (fts_q, limit))
        except sqlite3.OperationalError as e:
            logger.warning("FTS erro '%s': %s", fts_q, e)
            return []
    return [dict(r) for r in cur.fetchall()]


def search_2layer(
    con: sqlite3.Connection,
    fts_q: str,
    limit: int = 5,
    core_bonus: float = 2.0,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> List[Dict[str, Any]]:
    """Índice 2 camadas: busca ampla e re-rankeia priorizando as 5 NRs críticas.

    bm25() retorna score negativo (menor = melhor). Aplicamos um bônus multiplicativo
    às NRs críticas e uma penalização às demais, re-ordenando um pool amplo (limit*6).
    """
    pool = search(con, fts_q, limit=limit * 6, weights=weights)
    for r in pool:
        base = r["score"]
        if r["doc"] in CORE_5_NRS:
            # bm25 negativo: multiplicar por >1 torna mais negativo (melhor)
            r["adj"] = base * core_bonus
        else:
            r["adj"] = base * (1.0 / 1.5)  # penaliza (menos negativo = pior)
    pool.sort(key=lambda x: x["adj"])
    return pool[:limit]


def eval_strategy(
    con: sqlite3.Connection,
    qa_pairs: List[Dict[str, Any]],
    rewrites: Dict[str, str],
    strategy: str,
) -> Tuple[EvalMetrics, Dict[str, EvalMetrics]]:
    """Avalia uma estratégia; retorna métricas totais e por-norma (recall@1/2/3/5)."""
    total = EvalMetrics(total_queries=len(qa_pairs), hits_at_1=0, hits_at_3=0, hits_at_5=0, mrr_sum=0.0)
    hits_at_2 = 0
    by_doc: Dict[str, EvalMetrics] = defaultdict(
        lambda: EvalMetrics(total_queries=0, hits_at_1=0, hits_at_3=0, hits_at_5=0, mrr_sum=0.0)
    )
    by_doc_h2: Dict[str, int] = defaultdict(int)

    for p in qa_pairs:
        doc = p.get("doc", "outros")
        dm = by_doc[doc]
        dm.total_queries += 1

        # Pesos BM25 de produção (Fts5Retriever.kt): doc=1.5, section=3.0, title=2.0, text=1.0
        APP_W = (1.5, 3.0, 2.0, 1.0)
        rw = rewrites.get(p["id"], p["question"])

        if strategy == "app_raw":
            # Reflete EXATAMENTE o que o dispositivo faz hoje com a pergunta bruta
            q = app_fts_query(p["question"])
            res = search(con, q, limit=5, weights=APP_W)
        elif strategy == "app_rw":
            # Rewrite zero-shot -> sanitize do app -> busca ampla (sem filtro de norma)
            q = app_fts_query(rw)
            res = search(con, q, limit=5, weights=APP_W)
        elif strategy == "app_rw_normfilter":
            # Rewrite + filtro de norma se detectada na pergunta/rewrite (fallback amplo)
            q = app_fts_query(rw)
            norm = detect_norm(p["question"]) or detect_norm(rw)
            if norm:
                res = search(con, q, limit=5, weights=APP_W, doc_filter=norm)
                if len(res) < 5:
                    extra = search(con, q, limit=5, weights=APP_W)
                    seen = {r["id"] for r in res}
                    res += [r for r in extra if r["id"] not in seen]
                    res = res[:5]
            else:
                res = search(con, q, limit=5, weights=APP_W)
        elif strategy == "app_rw_2layer":
            q = app_fts_query(rw)
            res = search_2layer(con, q, limit=5, weights=APP_W)
        elif strategy == "ceil_terms":
            # Teto do caminho do app: termos curados + sanitize do app (sem filtro)
            q = app_fts_query(p["query_terms"])
            res = search(con, q, limit=5, weights=APP_W)
        elif strategy == "ceil_terms_normfilter":
            # Teto absoluto: termos curados + filtro pela norma-ouro
            q = app_fts_query(p["query_terms"])
            res = search(con, q, limit=5, weights=APP_W, doc_filter=p["doc"])
        else:
            raise ValueError(f"estratégia desconhecida: {strategy}")

        rank: Optional[int] = None
        for idx, r in enumerate(res[:5]):
            if check_hit(r, p):
                rank = idx + 1
                break

        if rank is not None:
            if rank <= 1:
                total.hits_at_1 += 1; dm.hits_at_1 += 1
            if rank <= 2:
                hits_at_2 += 1; by_doc_h2[doc] += 1
            if rank <= 3:
                total.hits_at_3 += 1; dm.hits_at_3 += 1
            if rank <= 5:
                total.hits_at_5 += 1; dm.hits_at_5 += 1
            total.mrr_sum += 1.0 / rank
            dm.mrr_sum += 1.0 / rank

    # Anexa recall@2 como atributo dinâmico
    total.hits_at_2 = hits_at_2  # type: ignore[attr-defined]
    for d, m in by_doc.items():
        m.hits_at_2 = by_doc_h2[d]  # type: ignore[attr-defined]
    return total, dict(by_doc)


def r2(m: EvalMetrics) -> float:
    return (getattr(m, "hits_at_2", 0) / m.total_queries) * 100 if m.total_queries else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação Fase 1 M3 (índice fino + rewrite zero-shot).")
    ap.add_argument("--db", type=str, default="corpus/index_hf_36nr_fino.db")
    ap.add_argument("--qa", type=str, default="corpus/qa_pairs_v2.jsonl")
    ap.add_argument("--cache", type=str, default="corpus/data/rewrites_zeroshot.json")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", type=str, default="local")
    ap.add_argument("--no-rewrite", action="store_true", help="Não gera rewrites (usa cache existente ou pergunta bruta).")
    ap.add_argument("--fewshot", action="store_true", help="Usa o prompt few-shot melhorado (nomeia NR + termos).")
    ap.add_argument("--reasoning", action="store_true", help="Modelo de raciocínio (LFM2.5-2.6B): orçamento de tokens maior.")
    ap.add_argument("--json-out", type=str, default=None, help="Salva métricas em JSON.")
    args = ap.parse_args()

    qa = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
    logger.info("Perguntas carregadas: %d", len(qa))

    cache_path = Path(args.cache)
    if args.no_rewrite:
        rewrites = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    else:
        t0 = time.time()
        rewrites = build_rewrite_cache(qa, args.base_url, cache_path, model=args.model, fewshot=args.fewshot, reasoning=args.reasoning)
        logger.info("Rewrites prontos em %.1fs", time.time() - t0)

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    strategies = [
        ("app_raw", "App HOJE: pergunta bruta (1.5,3,2,1)"),
        ("app_rw", "App + rewrite ZS (sem filtro)"),
        ("app_rw_normfilter", "App + rewrite ZS + filtro norma"),
        ("app_rw_2layer", "App + rewrite ZS + 2 camadas"),
        ("ceil_terms", "TETO: termos curados (sem filtro)"),
        ("ceil_terms_normfilter", "TETO: termos curados + filtro norma"),
    ]

    print("\n" + "=" * 96)
    print(f"  FASE 1 M3 — RETRIEVAL ÍNDICE FINO ({Path(args.db).name}) — {len(qa)} PERGUNTAS REAIS (NÃO-CONTAMINADAS)")
    print("=" * 96)
    print(f"| {'Estratégia':<38} | {'R@1':>6} | {'R@2':>6} | {'R@3':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 40 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")

    results: Dict[str, Any] = {}
    for strat, label in strategies:
        m, by_doc = eval_strategy(con, qa, rewrites, strat)
        print(f"| {label:<38} | {m.recall_at_1:>5.1f}% | {r2(m):>5.1f}% | {m.recall_at_3:>5.1f}% | {m.recall_at_5:>5.1f}% | {m.mrr:>7.4f} |")
        results[strat] = {
            "recall_at_1": m.recall_at_1, "recall_at_2": r2(m),
            "recall_at_3": m.recall_at_3, "recall_at_5": m.recall_at_5, "mrr": m.mrr,
            "by_doc": {d: {"n": dm.total_queries, "r2": r2(dm), "r5": dm.recall_at_5} for d, dm in by_doc.items()},
        }
    print("=" * 96)

    # Destaque das 5 NRs críticas na melhor estratégia por R@2
    best = max(results, key=lambda k: results[k]["recall_at_2"])
    print(f"\nMelhor estratégia por Recall@2: {best} ({results[best]['recall_at_2']:.1f}%)")
    print("\n--- Detalhamento por norma (5 críticas) na melhor estratégia ---")
    for d in sorted(CORE_5_NRS):
        bd = results[best]["by_doc"].get(d)
        if bd:
            print(f"  {d.upper()}: n={bd['n']:>3}  R@2={bd['r2']:>5.1f}%  R@5={bd['r5']:>5.1f}%")

    con.close()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Métricas salvas em %s", args.json_out)


if __name__ == "__main__":
    main()
