#!/usr/bin/env python3
"""
hybrid.py — FASE C: pipeline híbrido de 2 estágios (classificador -> BM25 com boost suave).

Estágio 1: o classificador vencedor (LogisticRegression TF-IDF) prevê top-1/top-2 NRs
a partir da FALA BRUTA (entrada honesta de campo, sem depender do rewriter).
Estágio 2: BM25 no índice de produção (index_hf_36nr.db) com BOOST SUAVE (não filtro
duro) nas NRs indicadas. Se o classificador disser 'nenhuma', busca sem boost.

Calibração do peso do boost (brief): 2x / 5x / 10x e filtro-duro como referência,
escolhendo pelo recall@2 nas 151 (aqui reportamos também o holdout completo 171).

O caminho de busca replica FIELMENTE o app (app_fts_query + pesos 1.5/3/2/1 +
check_hit), reusando corpus/eval_fino.py para paridade honesta com o dispositivo.

Uso:
    python3 classifier/hybrid.py --db corpus/index_hf_36nr.db --json-out classifier/data/hybrid_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))  # p/ importar corpus.*

from corpus.eval_fino import APP_STOPWORDS, app_fts_query  # noqa: E402
from corpus.eval_retrieval import check_hit  # noqa: E402
from data_utils import load_holdout  # noqa: E402
from nr_taxonomy import NONE_LABEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("hybrid")

APP_W = (1.5, 3.0, 2.0, 1.0)  # pesos BM25 de produção (Fts5Retriever.kt)


def load_classifier(path: Path):
    """Carrega o pipeline vencedor (vetorizadores char/word + modelo)."""
    with path.open("rb") as f:
        blob = pickle.load(f)
    return blob["char"], blob["word"], blob["clf"], blob.get("name", "?")


def predict_topk(char_vec, word_vec, clf, texts: List[str], k: int = 2) -> Tuple[List[List[str]], np.ndarray]:
    """Prevê as k NRs de maior score por fala. Retorna (top-k códigos, matriz de probs)."""
    Xc = char_vec.transform(texts)
    Xw = word_vec.transform(texts)
    X = hstack([Xc, Xw]).tocsr()
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)
    else:
        scores = clf.decision_function(X)
    classes = clf.classes_
    order = np.argsort(-scores, axis=1)[:, :k]
    topk = [[classes[j] for j in row] for row in order]
    return topk, scores


def search_pool(con: sqlite3.Connection, fts_q: str, limit: int) -> List[Dict[str, Any]]:
    """Busca BM25 ampla (pool) com pesos de produção; score bm25 negativo (menor=melhor)."""
    if fts_q == '""':
        return []
    w = APP_W
    sql = f"""
        SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
               bm25(chunks_fts, {w[0]}, {w[1]}, {w[2]}, {w[3]}) AS score
        FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC LIMIT ?;
    """
    cur = con.cursor()
    try:
        cur.execute(sql, (fts_q, limit))
    except sqlite3.OperationalError as e:
        logger.warning("FTS erro '%s': %s", fts_q, e)
        return []
    return [dict(r) for r in cur.fetchall()]


def apply_soft_boost(pool: List[Dict[str, Any]], boost_nrs: List[str], factor: float) -> List[Dict[str, Any]]:
    """Reordena o pool aplicando boost multiplicativo às NRs indicadas.

    bm25 devolve score NEGATIVO (menor=melhor). Multiplicar por factor>1 torna o score
    mais negativo (melhor) para chunks das NRs previstas — boost SUAVE, não filtro duro:
    chunks de outras NRs continuam elegíveis, apenas perdem prioridade relativa.
    """
    boost_set = set(boost_nrs)
    for r in pool:
        r["adj"] = r["score"] * factor if r["doc"] in boost_set else r["score"]
    pool.sort(key=lambda x: x["adj"])
    return pool


def hard_filter(con: sqlite3.Connection, fts_q: str, boost_nrs: List[str], limit: int) -> List[Dict[str, Any]]:
    """Referência: filtro DURO restringindo a busca às NRs previstas (com fallback amplo)."""
    if fts_q == '""' or not boost_nrs:
        return search_pool(con, fts_q, limit)
    w = APP_W
    placeholders = ",".join("?" for _ in boost_nrs)
    sql = f"""
        SELECT c.id, c.doc, c.section, c.title, c.page, c.text,
               bm25(chunks_fts, {w[0]}, {w[1]}, {w[2]}, {w[3]}) AS score
        FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE c.doc IN ({placeholders}) AND chunks_fts MATCH ?
        ORDER BY score ASC LIMIT ?;
    """
    cur = con.cursor()
    try:
        cur.execute(sql, (*boost_nrs, fts_q, limit))
    except sqlite3.OperationalError as e:
        logger.warning("FTS erro '%s': %s", fts_q, e)
        return []
    res = [dict(r) for r in cur.fetchall()]
    if len(res) < limit:  # completa com busca ampla
        extra = search_pool(con, fts_q, limit)
        seen = {r["id"] for r in res}
        res += [r for r in extra if r["id"] not in seen]
    return res[:limit]


def eval_strategy(
    con: sqlite3.Connection,
    holdout: List[Dict[str, Any]],
    topk_preds: List[List[str]],
    strategy: str,
    factor: float = 1.0,
    k_nr: int = 2,
    pool_limit: int = 60,
    use_oracle: bool = False,
    query_source: str = "raw",
    rewrites: Optional[Dict[str, str]] = None,
    probs: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    conf_thr: float = 0.5,
) -> Dict[str, float]:
    """Avalia uma estratégia de boost. Retorna recall@1/2/5 e MRR (sobre itens com gold NR)."""
    # Itens 'nenhuma' no gold não têm NR-alvo recuperável -> excluídos do denominador de recall
    scored = [(i, h, pr) for i, (h, pr) in enumerate(zip(holdout, topk_preds)) if h["gold"] != NONE_LABEL]
    n = len(scored)
    h1 = h2 = h5 = 0
    mrr = 0.0
    for orig_i, h, preds in scored:
        if use_oracle:
            # Teto do desenho: usa a norma-OURO no lugar da predição do classificador
            boost_nrs = [h["gold"]]
        else:
            boost_nrs = [p for p in preds[:k_nr] if p != NONE_LABEL]
        # Fonte da query BM25: fala bruta (voz) ou rewrite do Turno 1 (produção)
        if query_source == "rewrite" and rewrites:
            qtext = rewrites.get(h["id"], h["text"])
        else:
            qtext = h["text"]
        q = app_fts_query(qtext)
        if strategy == "baseline":
            res = search_pool(con, q, 5)
        elif strategy == "soft":
            pool = search_pool(con, q, pool_limit)
            res = apply_soft_boost(pool, boost_nrs, factor)[:5]
        elif strategy == "hard":
            res = hard_filter(con, q, boost_nrs, 5)
        elif strategy == "gated":
            # Confiança-gated: se o classificador está confiante (top-1 prob >= thr),
            # aplica filtro DURO na top-1; senão, boost SUAVE nas top-2 (mais seguro).
            p1 = float(probs[orig_i].max()) if probs is not None else 0.0
            if p1 >= conf_thr and boost_nrs:
                res = hard_filter(con, q, boost_nrs[:1], 5)
            else:
                pool = search_pool(con, q, pool_limit)
                res = apply_soft_boost(pool, boost_nrs, factor)[:5]
        else:
            raise ValueError(strategy)
        gold = _gold_pair(h)
        rank = None
        for idx, r in enumerate(res[:5]):
            if check_hit(r, gold):
                rank = idx + 1
                break
        if rank is not None:
            if rank <= 1:
                h1 += 1
            if rank <= 2:
                h2 += 1
            if rank <= 5:
                h5 += 1
            mrr += 1.0 / rank
    return {
        "recall_at_1": round(100 * h1 / n, 2),
        "recall_at_2": round(100 * h2 / n, 2),
        "recall_at_5": round(100 * h5 / n, 2),
        "mrr": round(mrr / n, 4),
        "n": n,
    }


# check_hit precisa do par-ouro completo (chunk_id, relevant_chunk_ids, section) para
# reproduzir fielmente o eval do app; guardamos o par original por id.
_GOLD_PAIR: Dict[str, Dict[str, Any]] = {}


def _gold_pair(h: Dict[str, Any]) -> Dict[str, Any]:
    """Par-ouro completo p/ check_hit; fallback p/ smoke (sem relevant_chunk_ids)."""
    return _GOLD_PAIR.get(
        h["id"],
        {"doc": h["gold"], "section": "", "chunk_id": -1, "relevant_chunk_ids": []},
    )


def _load_gold_sections() -> None:
    """Carrega o par-ouro de qa_pairs_v2 e smoke para casar via check_hit."""
    for path in [_ROOT / "corpus" / "qa_pairs_v2.jsonl", _ROOT / "bench" / "data" / "smoke_qa_20.jsonl"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            _GOLD_PAIR[r["id"]] = {
                "doc": str(r.get("doc", "") or "").lower(),
                "section": str(r.get("section", "") or ""),
                "chunk_id": r.get("chunk_id", -1),
                "relevant_chunk_ids": r.get("relevant_chunk_ids", []),
            }


def main() -> None:
    ap = argparse.ArgumentParser(description="FASE C: híbrido 2 estágios + calibração de boost.")
    ap.add_argument("--db", type=str, default=str(_ROOT / "corpus" / "index_hf_36nr.db"))
    ap.add_argument("--model", type=str, default=str(_HERE / "models" / "classic_winner.pkl"))
    ap.add_argument("--json-out", type=str, default=str(_HERE / "data" / "hybrid_results.json"))
    ap.add_argument("--subset", type=str, default="qa_v2", choices=["qa_v2", "all"],
                    help="qa_v2=151 reais (métrica oficial do brief); all=171.")
    ap.add_argument("--rewrites", type=str, default=str(_ROOT / "corpus" / "data" / "rewrites_zeroshot.json"),
                    help="Cache de rewrites do Turno 1 (produção) p/ variante rewrite+boost.")
    args = ap.parse_args()

    _load_gold_sections()
    char_vec, word_vec, clf, name = load_classifier(Path(args.model))
    logger.info("Classificador: %s", name)

    holdout = load_holdout()
    if args.subset == "qa_v2":
        holdout = [h for h in holdout if h["source"] == "qa_v2"]
    texts = [h["text"] for h in holdout]

    t0 = time.perf_counter()
    topk, probs = predict_topk(char_vec, word_vec, clf, texts, k=2)
    clf_ms = (time.perf_counter() - t0) / len(texts) * 1000
    # Acurácia do classificador no subset
    golds = [h["gold"] for h in holdout]
    c1 = sum(1 for p, g in zip(topk, golds) if p[0] == g)
    c2 = sum(1 for p, g in zip(topk, golds) if g in p[:2])
    logger.info("Classificador no subset (%d): top1=%.1f%% top2=%.1f%% | %.2f ms/consulta",
                len(holdout), 100 * c1 / len(holdout), 100 * c2 / len(holdout), clf_ms)

    rewrites: Dict[str, str] = {}
    rw_path = Path(args.rewrites)
    if rw_path.exists():
        rewrites = json.loads(rw_path.read_text(encoding="utf-8"))
        logger.info("Rewrites carregados: %d (variante rewrite+boost)", len(rewrites))

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    configs = [
        ("baseline", "Baseline (fala bruta, sem boost)", {"strategy": "baseline"}),
        ("soft_2x", "Híbrido boost suave 2x (top-2 NR)", {"strategy": "soft", "factor": 2.0}),
        ("soft_5x", "Híbrido boost suave 5x (top-2 NR)", {"strategy": "soft", "factor": 5.0}),
        ("soft_10x", "Híbrido boost suave 10x (top-2 NR)", {"strategy": "soft", "factor": 10.0}),
        ("soft_5x_top1", "Híbrido boost suave 5x (top-1 NR)", {"strategy": "soft", "factor": 5.0, "k_nr": 1}),
        ("gated_05", "Híbrido CONFIANÇA-GATED (thr 0.5) *REC*", {"strategy": "gated", "factor": 5.0, "conf_thr": 0.5, "probs": None}),
        ("hard", "Referência: filtro DURO (top-2 NR)", {"strategy": "hard"}),
        ("hard_top1", "Referência: filtro DURO (top-1 NR)", {"strategy": "hard", "k_nr": 1}),
        ("rw_soft5x", "Híbrido rewrite+boost suave 5x (top-2)", {"strategy": "soft", "factor": 5.0, "query_source": "rewrite", "rewrites": rewrites}),
        ("rw_hard", "Híbrido rewrite+filtro DURO (top-2)", {"strategy": "hard", "query_source": "rewrite", "rewrites": rewrites}),
        ("oracle_soft5x", "TETO: boost suave 5x com norma-OURO", {"strategy": "soft", "factor": 5.0, "use_oracle": True}),
        ("oracle_hard", "TETO: filtro DURO com norma-OURO", {"strategy": "hard", "use_oracle": True}),
        ("oracle_rw_hard", "TETO: rewrite+filtro DURO norma-OURO", {"strategy": "hard", "use_oracle": True, "query_source": "rewrite", "rewrites": rewrites}),
    ]

    print("\n" + "=" * 92)
    print(f"  FASE C — HÍBRIDO 2 ESTÁGIOS ({args.subset}, {len(holdout)} perguntas) — índice {Path(args.db).name}")
    print("=" * 92)
    print(f"| {'Configuração':<38} | {'R@1':>6} | {'R@2':>6} | {'R@5':>6} | {'MRR':>7} |")
    print("|" + "-" * 40 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|")

    results = {}
    for key, label, kw in configs:
        if kw.get("strategy") == "gated":
            kw = {**kw, "probs": probs, "classes": clf.classes_}
        m = eval_strategy(con, holdout, topk, **kw)
        results[key] = {"label": label, **m}
        print(f"| {label:<38} | {m['recall_at_1']:>5.1f}% | {m['recall_at_2']:>5.1f}% | "
              f"{m['recall_at_5']:>5.1f}% | {m['mrr']:>7.4f} |")
    print("=" * 92)

    # Escolha do desenho de produção: melhor R@2 entre configs REAIS (não-oracle, não-baseline)
    real_keys = [k for k in results if not k.startswith("oracle") and k != "baseline"]
    best_real = max(real_keys, key=lambda k: (results[k]["recall_at_2"], results[k]["recall_at_5"]))
    soft_keys = [k for k in results if k.startswith("soft")]
    best_soft = max(soft_keys, key=lambda k: results[k]["recall_at_2"])
    base_r2 = results["baseline"]["recall_at_2"]
    print(f"\nBaseline R@2: {base_r2:.1f}%")
    print(f"Melhor config REAL por R@2: {best_real} ({results[best_real]['label']}) "
          f"R@2={results[best_real]['recall_at_2']:.1f}% R@5={results[best_real]['recall_at_5']:.1f}% "
          f"-> +{results[best_real]['recall_at_2'] - base_r2:.1f} p.p.")
    print(f"Melhor boost suave puro por R@2: {best_soft} ({results[best_soft]['recall_at_2']:.1f}%)")
    print(f"Filtro duro (referência) R@2: {results['hard']['recall_at_2']:.1f}%")
    print(f"TETO do desenho (norma-ouro, boost suave): {results['oracle_soft5x']['recall_at_2']:.1f}% R@2")
    meta = 35.0
    ok = results[best_real]["recall_at_2"] >= meta
    print(f"Meta honesta R@2 >= {meta:.0f}%: {'ATINGIDA' if ok else 'NÃO atingida'} "
          f"({results[best_real]['recall_at_2']:.1f}%)")

    con.close()
    out = {
        "subset": args.subset,
        "n": len(holdout),
        "classifier": name,
        "classifier_top1": round(100 * c1 / len(holdout), 2),
        "classifier_top2": round(100 * c2 / len(holdout), 2),
        "classifier_ms_per_query": round(clf_ms, 3),
        "baseline_r2": base_r2,
        "best_soft": best_soft,
        "best_real": best_real,
        "meta_r2": meta,
        "meta_met": ok,
        "results": results,
    }
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Métricas salvas em %s", args.json_out)


if __name__ == "__main__":
    main()
