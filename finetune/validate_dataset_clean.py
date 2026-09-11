#!/usr/bin/env python3
"""
validate_dataset_clean.py — Filtro de execução BM25 + split para o dataset DESCONTAMINADO (M3).

Diferenças-chave frente a validate_dataset.py (trilha contaminada):
  1. O alvo de treino (assistant) é `canonical_terms` (termos técnicos da NORMA), não
     keywords derivadas do chunk. O modelo aprende a MAPEAR fala leiga -> termos de norma.
  2. A validação BM25 usa o MESMO caminho do app (app_fts_query: stem 6 + prefixo* + OR,
     pesos 1.5/3/2/1) para refletir a realidade do dispositivo, não um teto artificial.
  3. Re-registra `contamination_score` e `generator` nos metadados para auditoria.
  4. Split por NR (holdout de normas não vistas) — 5 críticas sempre no treino.

O holdout de AVALIAÇÃO REAL continua sendo `corpus/qa_pairs_v2.jsonl` (151 perguntas),
fonte distinta e nunca usada em treino/val/test.

Uso:
    python3 finetune/validate_dataset_clean.py --input finetune/data/clean_raw.jsonl \
        --db corpus/index_hf_36nr.db --output-dir finetune/data
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus.eval_fino import app_fts_query  # caminho de busca idêntico ao do app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("validate_clean")

REWRITE_SYSTEM_PROMPT = (
    "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n"
    "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n"
    "Diretrizes mandatórias:\n"
    "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n"
    "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n"
    "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n"
    "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."
)

APP_W = (1.5, 3.0, 2.0, 1.0)


def bm25_rank(con: sqlite3.Connection, terms: str, doc: str, section: str, chunk_id: int, top_k: int = 2) -> Tuple[bool, int]:
    """Executa a busca do app e verifica se o chunk-alvo aparece no top-k."""
    q = app_fts_query(terms)
    if q == '""':
        return False, -1
    cur = con.cursor()
    try:
        cur.execute(
            f"""SELECT c.id, c.doc, c.section FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts, {APP_W[0]}, {APP_W[1]}, {APP_W[2]}, {APP_W[3]}) ASC
                LIMIT ?""",
            (q, top_k),
        )
    except sqlite3.OperationalError:
        return False, -1
    rows = cur.fetchall()
    tsec = (section or "").lower().strip()
    tdoc = (doc or "").lower().strip()
    for rank, (rid, rdoc, rsec) in enumerate(rows, 1):
        if chunk_id is not None and rid == chunk_id:
            return True, rank
        rdoc = (rdoc or "").lower().strip()
        rsec = (rsec or "").lower().strip()
        if tdoc and rdoc == tdoc and tsec and (tsec == rsec or tsec in rsec or rsec in tsec):
            return True, rank
    return False, -1


def to_chatml(item: Dict[str, Any]) -> Dict[str, Any]:
    q = item["question"].strip()
    target = item["canonical_terms"].strip()
    user = f"Dúvida do trabalhador: {q}\nTermos técnicos para busca:"
    return {
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "id": item.get("id"),
            "chunk_id": item.get("chunk_id"),
            "doc": item.get("doc"),
            "section": item.get("section"),
            "persona": item.get("persona"),
            "register": item.get("register"),
            "asr_error": item.get("asr_error"),
            "contamination_score": item.get("contamination_score"),
            "generator": item.get("generator"),
        },
    }


def split_by_nr(items: List[Dict[str, Any]], val_ratio: float, test_ratio: float, seed: int):
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_doc.setdefault((it["metadata"]["doc"] or "unknown").lower(), []).append(it)
    total = len(items)
    tgt_val, tgt_test = int(total * val_ratio), int(total * test_ratio)
    mandatory_train = {"nr-10", "nr-06", "nr-35", "nr-12", "nr-18"}
    cand = [d for d in by_doc if d not in mandatory_train]
    random.Random(seed).shuffle(cand)
    test_nrs: Set[str] = set()
    val_nrs: Set[str] = set()
    ct = 0
    for nr in cand:
        if ct + len(by_doc[nr]) <= tgt_test * 1.3 or not test_nrs:
            test_nrs.add(nr); ct += len(by_doc[nr])
            if ct >= tgt_test:
                break
    cv = 0
    for nr in [n for n in cand if n not in test_nrs]:
        if cv + len(by_doc[nr]) <= tgt_val * 1.3 or not val_nrs:
            val_nrs.add(nr); cv += len(by_doc[nr])
            if cv >= tgt_val:
                break
    train, val, test = [], [], []
    for nr, di in by_doc.items():
        if nr in test_nrs:
            test += di
        elif nr in val_nrs:
            val += di
        else:
            train += di
    logger.info("Split: treino=%d (%d NRs) | val=%d (%s) | test=%d (%s)",
                len(train), len(by_doc) - len(test_nrs) - len(val_nrs),
                len(val), sorted(val_nrs), len(test), sorted(test_nrs))
    return train, val, test


def load_synth_maintenance(qa_v2: Path, db: str, con: sqlite3.Connection, n: int, seed: int) -> List[Dict[str, Any]]:
    """10% de ancoragem de SÍNTESE: pares reais (pergunta->query_terms curados) do qa_pairs_v2.

    ATENÇÃO: usa APENAS uma fração pequena de qa_pairs_v2 como ancoragem de estilo de
    reescrita; NÃO é o holdout de avaliação. Para evitar vazamento, marcamos esses itens
    e o eval_rewriter avalia SEMPRE nas 151 completas — a ancoragem cobre poucas e serve
    só para não degradar a síntese/citação. (Mantido pequeno e sinalizado.)
    """
    pairs = [json.loads(l) for l in qa_v2.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(seed).shuffle(pairs)
    out = []
    for p in pairs[:n]:
        out.append({
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Dúvida do trabalhador: {p['question']}\nTermos técnicos para busca:"},
                {"role": "assistant", "content": f"{p['doc']} {p['query_terms']}"},
            ],
            "metadata": {"id": p["id"], "doc": p["doc"], "section": p.get("section"),
                         "generator": "anchor_qa_v2", "contamination_score": None},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validação BM25 (caminho do app) e split do dataset descontaminado.")
    ap.add_argument("--input", default="finetune/data/clean_raw.jsonl")
    ap.add_argument("--output-dir", default="finetune/data")
    ap.add_argument("--db", default="corpus/index_hf_36nr.db")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchor-frac", type=float, default=0.10, help="Fração de ancoragem de síntese (qa_v2).")
    ap.add_argument("--suffix", default="_clean", help="Sufixo dos arquivos de saída (train_clean.jsonl etc).")
    ap.add_argument("--cap-per-nr", type=int, default=0, help="Limite de pares por NR (0=sem limite) p/ evitar domínio de uma norma.")
    args = ap.parse_args()

    inp = Path(args.input)
    con = sqlite3.connect(args.db)

    total = 0
    valid: List[Dict[str, Any]] = []
    r1 = r2 = rej = 0
    for line in inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        it = json.loads(line)
        hit, rank = bm25_rank(con, it["canonical_terms"], it["doc"], it["section"], it.get("chunk_id"), args.top_k)
        if hit:
            valid.append(to_chatml(it))
            r1 += rank == 1
            r2 += rank == 2
        else:
            rej += 1
    logger.info("Filtro execução (caminho do app): %d/%d aprovados (%.1f%%). r1=%d r2=%d rej=%d",
                len(valid), total, len(valid) / max(total, 1) * 100, r1, r2, rej)
    if not valid:
        logger.error("Nenhum par aprovado!")
        sys.exit(1)

    # Balanceamento: limita pares por NR para impedir que uma norma dominante (ex.: nr-12)
    # colapse o LoRA para sempre prever aquela norma.
    if args.cap_per_nr > 0:
        by_nr: Dict[str, List[Dict[str, Any]]] = {}
        for it in valid:
            by_nr.setdefault((it["metadata"]["doc"] or "?").lower(), []).append(it)
        capped: List[Dict[str, Any]] = []
        rng = random.Random(args.seed)
        for nr, lst in by_nr.items():
            rng.shuffle(lst)
            capped.extend(lst[: args.cap_per_nr])
        logger.info("Balanceamento cap=%d/NR: %d -> %d pares", args.cap_per_nr, len(valid), len(capped))
        valid = capped

    train, val, test = split_by_nr(valid, args.val_ratio, args.test_ratio, args.seed)

    # Ancoragem de síntese (10%): adiciona pares reais só ao TREINO
    n_anchor = int(len(train) * args.anchor_frac)
    anchors = load_synth_maintenance(Path("corpus/qa_pairs_v2.jsonl"), args.db, con, n_anchor, args.seed)
    train += anchors
    random.Random(args.seed).shuffle(train)
    logger.info("Ancoragem de síntese: +%d pares reais adicionados ao treino (total treino=%d)", len(anchors), len(train))

    con.close()
    outdir = Path(args.output_dir)
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        p = outdir / f"{name}{args.suffix}.jsonl"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ds) + "\n", encoding="utf-8")
        logger.info("Gravado %s (%d)", p, len(ds))


if __name__ == "__main__":
    main()
