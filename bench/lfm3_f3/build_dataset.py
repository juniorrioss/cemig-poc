#!/usr/bin/env python3
"""
build_dataset.py — item 1c/1d do brief: monta o dataset de robustez da F3 a
partir do top-8 do retriever REAL (data/retrieval/topk_*.jsonl).

Duas fases:

  --assign : atribui a cada QA um caso + um k + posições (determinístico, seed
             fixa) e grava data/train/assignments.jsonl. Casos (quotas brief):
      gold    (~60%): ouro presente no contexto -> resposta ancorada+citação
      absent  (~20%): ouro ausente -> recusa curta padrão + o que consultaria
                      (prioriza perguntas que o retriever GENUINAMENTE erra;
                       completa removendo o ouro de casos com ouro, registrado)
      partial (~10%): sem ouro, melhor passagem 'partial' (coverage<1) ->
                      responde o sustentado e sinaliza o limite
      closed  (~10%): âncora closed-book da F2 (sem contexto, anti-esquecimento)
    k VARIÁVEL (ordem do capitão): ~50% k=5, ~25% k=3, ~25% k=8, posição do
    ouro ALEATÓRIA e registrada (gold_position).

  --render : junta assignments + targets.jsonl (27B, gen_targets.py) e grava
             data/train/{train,val}.jsonl no formato chat (messages) p/ TRL
             SFT com assistant_only_loss (system prompt IDÊNTICO ao do
             app/avaliação = common_f0.SYSTEM_PROMPT; contexto renderizado
             IDÊNTICO ao render_user do F0).

Dedup anticontaminação: Jaccard >= 0.5 da pergunta vs as 1536 do
benchmark_mobile (mesmo critério do s6 da F2) -> descartada e registrada.

Uso:
    classifier/.venv/bin/python bench/lfm3_f3/build_dataset.py --assign
    classifier/.venv/bin/python bench/lfm3_f3/build_dataset.py --render
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))

from common_synth import norm_tokens, read_jsonl, write_jsonl  # noqa: E402
from common_f0 import SYSTEM_PROMPT  # noqa: E402

TRAIN = HERE / "data" / "train"
RETR = HERE / "data" / "retrieval"
V4_DB = REPO / "android" / "app" / "src" / "main" / "assets" / "index.db"
BENCH_PARQUET = Path.home() / (
    ".cache/huggingface/hub/datasets--UFGCEMIGONA--benchmark_mobile/snapshots/"
    "5a5aa32ae4087481be828e071aff169827ec2993/normas_regulamentadoras/"
    "train-00000-of-00001.parquet")

SEED = 20260206
K_CHOICES = [5, 3, 8]
K_WEIGHTS = [0.50, 0.25, 0.25]
QUOTAS = {"gold": 0.60, "absent": 0.20, "partial": 0.10, "closed": 0.10}
VAL_FRAC = 0.02

REFUSAL_TEMPLATES = [
    "Não encontrei essa informação nas normas recuperadas. Para confirmar, "
    "eu consultaria a {nr_hint} diretamente ou o responsável de segurança.",
    "Os trechos de norma recuperados não cobrem esse ponto. O caminho seguro "
    "é verificar a {nr_hint} na íntegra ou falar com o técnico de segurança.",
    "Isso não está nos trechos das normas que recuperei. Vale consultar a "
    "{nr_hint} ou o engenheiro de segurança antes de prosseguir.",
    "Essa resposta não está nas normas recuperadas. Recomendo conferir a "
    "{nr_hint} completa ou acionar o SESMT da empresa.",
]


def nr_hint(ref: str, nr: int) -> str:
    """Âncora do que consultar: o item exato se a ref tiver, senão a NR."""
    return ref if ref else f"NR-{nr}"


def load_bench_questions() -> list[str]:
    import pyarrow.parquet as pq
    rows = pq.read_table(str(BENCH_PARQUET)).to_pylist()
    return [r["question"] for r in rows]


def load_chunk_texts() -> dict[int, dict]:
    con = sqlite3.connect(str(V4_DB))
    con.row_factory = sqlite3.Row
    out = {r["id"]: dict(r) for r in con.execute(
        "SELECT id, doc, section, title, text FROM chunks")}
    con.close()
    return out


def pick_k(rng: random.Random) -> int:
    return rng.choices(K_CHOICES, weights=K_WEIGHTS, k=1)[0]


# ---------------------------------------------------------------------------
# --assign
# ---------------------------------------------------------------------------
def assign() -> None:
    topk_rows, _seen = [], set()
    for p in sorted(RETR.glob("topk_*.jsonl")):
        for r in read_jsonl(p):
            if r["qa_id"] not in _seen:      # dedup por qa_id (colisão rara no pool)
                _seen.add(r["qa_id"])
                topk_rows.append(r)
    print(f"[assign] {len(topk_rows)} QAs com retrieval (dedup qa_id)")

    # dedup anticontaminação vs benchmark (mesmo critério do s6 da F2:
    # Jaccard de norm_tokens >= 0.5) — pré-tokenizado p/ não ser O(n*m) caro
    bench_tok = [set(norm_tokens(q)) for q in load_bench_questions()]

    def contaminated(q: str) -> bool:
        qt = set(norm_tokens(q))
        if not qt:
            return False
        for bt in bench_tok:
            inter = len(qt & bt)
            if inter and inter / len(qt | bt) >= 0.5:
                return True
        return False

    rng = random.Random(SEED)
    chunks = load_chunk_texts()

    # particiona por disponibilidade de ouro/parcial no top-8
    with_gold, no_gold_partial, no_gold_plain = [], [], []
    removed_contam = []
    for r in topk_rows:
        if contaminated(r["question"]):
            removed_contam.append(r["qa_id"])
            continue
        if r["gold_in_top8"]:
            with_gold.append(r)
        elif any(p["label"] == "partial" for p in r["topk"]):
            no_gold_partial.append(r)
        else:
            no_gold_plain.append(r)
    n_ctx_pool = len(with_gold) + len(no_gold_partial) + len(no_gold_plain)
    print(f"[assign] contaminadas descartadas: {len(removed_contam)}")
    print(f"[assign] com ouro@8: {len(with_gold)} | sem ouro c/ parcial: "
          f"{len(no_gold_partial)} | sem ouro puro: {len(no_gold_plain)}")

    # tamanho total do dataset com contexto = limitado pelo caso gold
    # quota: gold 60 / absent 20 / partial 10 / closed 10 (closed vem da F2)
    rng.shuffle(with_gold)
    rng.shuffle(no_gold_partial)
    rng.shuffle(no_gold_plain)

    n_gold = min(len(with_gold), int(n_ctx_pool * QUOTAS["gold"]))
    n_partial = min(len(no_gold_partial), int(n_gold * QUOTAS["partial"] / QUOTAS["gold"]))
    n_absent_target = int(n_gold * QUOTAS["absent"] / QUOTAS["gold"])
    n_closed = int(n_gold * QUOTAS["closed"] / QUOTAS["gold"])

    assignments = []

    # --- caso GOLD -----------------------------------------------------------
    gold_used = with_gold[:n_gold]
    for r in gold_used:
        k = pick_k(rng)
        golds = [p for p in r["topk"] if p["label"] == "gold"]
        others = [p for p in r["topk"] if p["label"] != "gold"]
        g = golds[0]  # melhor ouro (ordem do retriever)
        ctx = others[: k - 1]
        pos = rng.randrange(k) if len(ctx) == k - 1 else rng.randrange(len(ctx) + 1)
        ctx_ids = [p["id"] for p in ctx]
        ctx_ids.insert(pos, g["id"])
        assignments.append({
            "qa_id": r["qa_id"], "nr": r["nr"], "unit_id": r["unit_id"],
            "ref": r["ref"], "question": r["question"], "case": "gold",
            "k": len(ctx_ids), "gold_position": pos, "context_ids": ctx_ids,
            "anchor_passage": {**{kk: chunks[g["id"]][kk] for kk in
                                  ("id", "doc", "section", "title", "text")}},
            "labels": [("gold" if i == pos else
                        next(p["label"] for p in r["topk"] if p["id"] == cid))
                       for i, cid in enumerate(ctx_ids)],
            "origin_absent": None,
        })

    # --- caso ABSENT ---------------------------------------------------------
    # preferência: misses genuínos sem parcial; completa com remoção do ouro
    absent_rows = no_gold_plain[:n_absent_target]
    n_genuine = len(absent_rows)
    synth_pool = with_gold[n_gold:]
    i = 0
    while len(absent_rows) < n_absent_target and i < len(synth_pool):
        absent_rows.append(synth_pool[i])
        i += 1
    for r in absent_rows:
        genuine = not r["gold_in_top8"]
        k = pick_k(rng)
        # só distratores puros: uma 'partial' sobrando tornaria a recusa-alvo
        # desonesta (o caso coberto-em-parte é o 'partial', não o 'absent')
        pool = [p for p in r["topk"] if p["label"] == "distractor"][:k]
        if not pool:
            continue
        assignments.append({
            "qa_id": r["qa_id"], "nr": r["nr"], "unit_id": r["unit_id"],
            "ref": r["ref"], "question": r["question"], "case": "absent",
            "k": len(pool), "gold_position": None,
            "context_ids": [p["id"] for p in pool],
            "anchor_passage": None,
            "labels": [p["label"] for p in pool],
            "origin_absent": "genuine" if genuine else "gold_removed",
        })

    # --- caso PARTIAL --------------------------------------------------------
    for r in no_gold_partial[:n_partial]:
        k = pick_k(rng)
        pool = r["topk"][:k]
        # garante que a melhor parcial esteja no contexto
        best_partial = max((p for p in r["topk"] if p["label"] == "partial"),
                           key=lambda p: p["containment"])
        if best_partial["id"] not in {p["id"] for p in pool}:
            pool = pool[: k - 1] + [best_partial]
            rng.shuffle(pool)
        assignments.append({
            "qa_id": r["qa_id"], "nr": r["nr"], "unit_id": r["unit_id"],
            "ref": r["ref"], "question": r["question"], "case": "partial",
            "k": len(pool), "gold_position": None,
            "context_ids": [p["id"] for p in pool],
            "anchor_passage": {kk: chunks[best_partial["id"]][kk] for kk in
                               ("id", "doc", "section", "title", "text")},
            "labels": [p["label"] for p in pool],
            "origin_absent": None,
        })

    # --- caso CLOSED (âncora F2, anti-esquecimento) --------------------------
    f2_synth = REPO / "bench" / "lfm3_f2" / "data" / "synth"
    used_ids = {a["qa_id"] for a in assignments}
    closed_pool = []
    for nr in (10, 18, 34):
        closed_pool.extend(read_jsonl(f2_synth / f"qa_passed_nr{nr}.jsonl"))
    closed_pool = [q for q in closed_pool if q["qa_id"] not in used_ids]
    rng.shuffle(closed_pool)
    for q in closed_pool[:n_closed]:
        assignments.append({
            "qa_id": q["qa_id"], "nr": q["nr"], "unit_id": q["unit_id"],
            "ref": q["ref"], "question": q["question"], "case": "closed",
            "k": 0, "gold_position": None, "context_ids": [],
            "anchor_passage": None, "labels": [],
            "origin_absent": None, "closed_answer": q["answer"],
        })

    rng.shuffle(assignments)
    TRAIN.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN / "assignments.jsonl", assignments)

    stats = {
        "n_total": len(assignments),
        "by_case": dict(collections.Counter(a["case"] for a in assignments)),
        "by_k": dict(collections.Counter(a["k"] for a in assignments if a["k"])),
        "by_nr": dict(collections.Counter(a["nr"] for a in assignments)),
        "absent_origin": dict(collections.Counter(
            a["origin_absent"] for a in assignments if a["case"] == "absent")),
        "gold_pos_hist": dict(collections.Counter(
            a["gold_position"] for a in assignments if a["case"] == "gold")),
        "contaminated_removed": len(removed_contam),
        "absent_genuine": n_genuine,
        "seed": SEED,
    }
    (TRAIN / "assign_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# --render
# ---------------------------------------------------------------------------
def render_context(chunks: dict[int, dict], ids: list[int]) -> str:
    parts = []
    for i, cid in enumerate(ids, 1):
        c = chunks[cid]
        src = f"{c['doc']} - {c['section']} - {c['title']}"
        parts.append(f"[{i}] ({src}):\n{c['text']}")
    return "\n\n".join(parts)


def render() -> None:
    assignments = read_jsonl(TRAIN / "assignments.jsonl")
    targets = {t["qa_id"]: t["target"] for t in read_jsonl(TRAIN / "targets.jsonl")}
    chunks = load_chunk_texts()
    rng = random.Random(SEED + 1)

    rows, missing = [], 0
    seen_q: set[str] = set()   # dedup por texto de pergunta (colisões raras entre unidades)
    for a in assignments:
        case = a["case"]
        qnorm = " ".join(a["question"].lower().split())
        if qnorm in seen_q:
            continue
        seen_q.add(qnorm)
        if case in ("gold", "partial"):
            tgt = targets.get(a["qa_id"])
            if not tgt:
                missing += 1
                continue
        elif case == "absent":
            tpl = rng.choice(REFUSAL_TEMPLATES)
            tgt = tpl.format(nr_hint=nr_hint(a["ref"], a["nr"]))
        else:  # closed
            tgt = a["closed_answer"]

        if case == "closed":
            user = f"Pergunta:\n{a['question']}"
        else:
            ctx = render_context(chunks, a["context_ids"])
            user = (f"Contexto normativo consultado:\n{ctx}\n\n"
                    f"Pergunta:\n{a['question']}")
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": tgt},
            ],
            "qa_id": a["qa_id"], "case": case, "k": a["k"], "nr": a["nr"],
        })

    rng.shuffle(rows)
    # split estratificado por caso
    by_case: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    train, val = [], []
    for case, items in sorted(by_case.items()):
        n_val = max(1, int(len(items) * VAL_FRAC))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    write_jsonl(TRAIN / "train.jsonl", train)
    write_jsonl(TRAIN / "val.jsonl", val)
    print(f"[ok] train={len(train)} val={len(val)} sem_target={missing}")
    print("train by case:", dict(collections.Counter(r["case"] for r in train)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.assign:
        assign()
    if args.render:
        render()
    if not (args.assign or args.render):
        print("use --assign e/ou --render")


if __name__ == "__main__":
    main()
