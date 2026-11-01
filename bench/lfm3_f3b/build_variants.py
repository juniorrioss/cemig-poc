#!/usr/bin/env python3
"""
build_variants.py — datasets das células do sweep F3b (one-factor-at-a-time
sobre a receita F3). Cada variante nasce do dataset F3 CONGELADO
(bench/lfm3_f3/data/train/{assignments,targets}.jsonl) mudando UM fator:

  recusa35 : dose de recusa 20% -> 35%. Converte um sorteio seedado de casos
             gold em casos absent (ouro removido do contexto), com distratores
             DUROS (mesma NR, seção errada — label 'distractor' do próprio
             top-8 real, same-doc primeiro). Partial/closed intactos; n_total
             preservado. Alvo = recusa-padrão (mesmos templates, via render
             da F3).
  anchor25 : âncora anti-esquecimento 10% -> 25%. Mantém o F3 train.jsonl
             INTEIRO e adiciona pares de instrução GERAL PT-BR (não-domínio,
             gerados pelo 27B em gen_general.py) até âncora (closed+geral)
             = 25% do total. Pares gerais SEM system prompt (como o probe).
  data2x   : dataset ~2x. Usa topk_f3b.jsonl (funil das NRs restantes) somado
             aos topk da F3 e re-roda o assign/render da F3 com os MESMOS
             quotas/seed (os targets novos vêm do gen_targets com TRAIN
             patchado).

Saída: data/train/<variant>/{assignments,targets,train,val}.jsonl
Uso:   classifier/.venv/bin/python bench/lfm3_f3b/build_variants.py --variant recusa35
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # bench/lfm3_f3b
REPO = HERE.parents[1]
F3 = REPO / "bench" / "lfm3_f3"
F3_TRAIN = F3 / "data" / "train"
F3_RETR = F3 / "data" / "retrieval"
OUT = HERE / "data" / "train"

sys.path.insert(0, str(REPO / "bench" / "lfm3_f2" / "synth"))
from common_synth import read_jsonl, write_jsonl  # noqa: E402

SEED = 20260213
ABSENT_FRAC_TARGET = 0.35          # recusa35: fração absent sobre o total
ANCHOR_FRAC_TARGET = 0.25          # anchor25: (closed + geral) / total


def load_f3_build_dataset():
    """Importa bench/lfm3_f3/build_dataset.py como módulo (p/ reusar render)."""
    spec = importlib.util.spec_from_file_location(
        "f3_build_dataset", F3 / "build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_topk_index() -> dict[str, dict]:
    idx = {}
    for p in sorted(F3_RETR.glob("topk_*.jsonl")):
        for r in read_jsonl(p):
            idx.setdefault(r["qa_id"], r)
    return idx


# ---------------------------------------------------------------------------
# recusa35
# ---------------------------------------------------------------------------
def build_recusa35() -> None:
    out_dir = OUT / "recusa35"
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = read_jsonl(F3_TRAIN / "assignments.jsonl")
    topk = load_topk_index()
    rng = random.Random(f"{SEED}:recusa35")

    # unidades (fato literal) p/ garantir que o distrator duro é honesto
    # (containment < 0.30 vs o fato => não cobre a resposta)
    sys.path.insert(0, str(F3 / "synth"))
    from common_synth import norm_tokens  # noqa: PLC0415
    units: dict[str, dict] = {}
    for p in sorted((F3 / "data" / "synth").glob("units_nr*.jsonl")):
        for u in read_jsonl(p):
            units[u["unit_id"]] = u
    for p in sorted((REPO / "bench" / "lfm3_f2" / "data" / "synth").glob("units_nr*.jsonl")):
        for u in read_jsonl(p):
            units.setdefault(u["unit_id"], u)

    bd = load_f3_build_dataset()
    chunks = bd.load_chunk_texts()
    by_doc: dict[str, list[int]] = collections.defaultdict(list)
    for cid, c in chunks.items():
        by_doc[c["doc"].lower()].append(cid)

    n_total = len(assignments)
    n_absent_now = sum(1 for a in assignments if a["case"] == "absent")
    n_absent_target = int(n_total * ABSENT_FRAC_TARGET)
    n_convert = n_absent_target - n_absent_now
    assert n_convert > 0, "nada a converter"

    gold_rows = [a for a in assignments if a["case"] == "gold"]
    rng.shuffle(gold_rows)
    convert_ids = set()
    converted, skipped = 0, 0
    new_by_id = {}
    for a in gold_rows:
        if converted >= n_convert:
            break
        tk = topk.get(a["qa_id"])
        if not tk:
            skipped += 1
            continue
        # distratores do próprio top-8 real; DUROS (same-doc) primeiro.
        # partial é EXCLUÍDO (recusa-alvo seria desonesta).
        same_docs = {f"nr-{a['nr']:02d}", f"nr-{a['nr']}"}
        dis = [p for p in tk["topk"] if p["label"] == "distractor"]
        hard = [p for p in dis if p["doc"].lower() in same_docs]
        soft = [p for p in dis if p not in hard]
        # completa os duros com chunks same-doc de OUTRA seção do índice v4
        # (fora do top-8), honestos: containment < 0.30 vs o fato da unidade
        u = units.get(a["unit_id"])
        ft = set(norm_tokens(u["fato"])) if u else set()
        want_hard = max(1, a["k"] // 2)
        if len(hard) < want_hard and ft:
            used = {p["id"] for p in tk["topk"]}
            cand = [cid for d in same_docs for cid in by_doc.get(d, [])
                    if cid not in used]
            rng.shuffle(cand)
            for cid in cand:
                if len(hard) >= want_hard:
                    break
                ct = set(norm_tokens(chunks[cid]["text"]))
                cont = len(ft & ct) / len(ft)
                if cont < 0.30:
                    hard.append({"id": cid, "doc": chunks[cid]["doc"],
                                 "label": "distractor", "containment": round(cont, 4),
                                 "source": "index_hard_neg"})
        pool = (hard + soft)[: a["k"]]
        rng.shuffle(pool)
        if not pool:
            skipped += 1
            continue
        new_by_id[a["qa_id"]] = {
            **a, "case": "absent", "k": len(pool), "gold_position": None,
            "context_ids": [p["id"] for p in pool],
            "anchor_passage": None, "labels": [p["label"] for p in pool],
            "origin_absent": "gold_removed_hard",
            "n_hard_distractors": sum(1 for p in pool if p in hard),
        }
        convert_ids.add(a["qa_id"])
        converted += 1

    out_rows = [new_by_id.get(a["qa_id"], a) if a["case"] == "gold" else a
                for a in assignments]
    write_jsonl(out_dir / "assignments.jsonl", out_rows)
    # targets: só gold/partial precisam; copia o da F3 (subset continua válido)
    shutil.copy(F3_TRAIN / "targets.jsonl", out_dir / "targets.jsonl")

    stats = {
        "variant": "recusa35", "n_total": len(out_rows),
        "by_case": dict(collections.Counter(a["case"] for a in out_rows)),
        "converted_gold_to_absent": converted, "skipped": skipped,
        "hard_distractor_share": round(
            sum(r.get("n_hard_distractors", 0) for r in new_by_id.values())
            / max(1, sum(r["k"] for r in new_by_id.values())), 4),
        "absent_frac": round(sum(1 for a in out_rows if a["case"] == "absent")
                             / len(out_rows), 4),
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    _render_variant(out_dir)


def _render_variant(out_dir: Path) -> None:
    """Roda o render da F3 com TRAIN apontado p/ a variante."""
    bd = load_f3_build_dataset()
    bd.TRAIN = out_dir
    bd.render()


# ---------------------------------------------------------------------------
# anchor25
# ---------------------------------------------------------------------------
def build_anchor25() -> None:
    out_dir = OUT / "anchor25"
    out_dir.mkdir(parents=True, exist_ok=True)
    f3_train = read_jsonl(F3_TRAIN / "train.jsonl")
    f3_val = read_jsonl(F3_TRAIN / "val.jsonl")
    general = read_jsonl(HERE / "data" / "general" / "general_pairs.jsonl")
    assert len(general) >= 100, "rode gen_general.py antes"

    n_closed = sum(1 for r in f3_train if r["case"] == "closed")
    n0 = len(f3_train)
    # (closed + geral) / (n0 + geral) = ANCHOR_FRAC_TARGET
    need = int((ANCHOR_FRAC_TARGET * n0 - n_closed) / (1 - ANCHOR_FRAC_TARGET))
    rng = random.Random(f"{SEED}:anchor25")
    rng.shuffle(general)
    assert len(general) >= need + 20, f"faltam pares gerais: need={need}+20, tem {len(general)}"
    g_train = general[:need]
    g_val = general[need:need + 20]

    def as_row(g: dict) -> dict:
        # SEM system prompt (como o probe de capacidade usa o modelo)
        return {"messages": [{"role": "user", "content": g["prompt"]},
                             {"role": "assistant", "content": g["answer"]}],
                "qa_id": g["id"], "case": "general", "k": 0, "nr": 0}

    train = f3_train + [as_row(g) for g in g_train]
    val = f3_val + [as_row(g) for g in g_val]
    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats = {
        "variant": "anchor25", "n_train": len(train), "n_val": len(val),
        "n_general_train": len(g_train), "n_closed": n_closed,
        "anchor_frac": round((n_closed + len(g_train)) / len(train), 4),
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# data2x
# ---------------------------------------------------------------------------
def build_data2x() -> None:
    """Assign+render da F3 sobre o pool ampliado (topk da F3 + topk_f3b)."""
    out_dir = OUT / "data2x"
    out_dir.mkdir(parents=True, exist_ok=True)
    retr_dir = HERE / "data" / "retrieval"
    assert (retr_dir / "topk_f3b.jsonl").exists(), "rode o funil/retrieve f3b antes"
    # junta topk da F3 + novos num dir só da variante
    for p in sorted(F3_RETR.glob("topk_*.jsonl")):
        shutil.copy(p, retr_dir / p.name)
    bd = load_f3_build_dataset()
    bd.TRAIN = out_dir
    bd.RETR = retr_dir
    bd.assign()
    print("[data2x] assign pronto; gere os targets novos (gen_targets com "
          "TRAIN patchado via make targets-2x) e rode --variant data2x-render")


def render_data2x() -> None:
    _render_variant(OUT / "data2x")


# ---------------------------------------------------------------------------
# combo: recusa35 + anchor25 (os 2 eixos dominantes do sweep)
# ---------------------------------------------------------------------------
def build_combo() -> None:
    """train do recusa35 + pares gerais até âncora (closed+geral) = 25%."""
    out_dir = OUT / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = OUT / "recusa35"
    f3_train = read_jsonl(base_dir / "train.jsonl")
    f3_val = read_jsonl(base_dir / "val.jsonl")
    general = read_jsonl(HERE / "data" / "general" / "general_pairs.jsonl")

    n_closed = sum(1 for r in f3_train if r["case"] == "closed")
    n0 = len(f3_train)
    need = int((ANCHOR_FRAC_TARGET * n0 - n_closed) / (1 - ANCHOR_FRAC_TARGET))
    rng = random.Random(f"{SEED}:combo")
    rng.shuffle(general)
    assert len(general) >= need + 20, f"faltam pares gerais: {need}+20 > {len(general)}"
    g_train = general[:need]
    g_val = general[need:need + 20]

    def as_row(g: dict) -> dict:
        return {"messages": [{"role": "user", "content": g["prompt"]},
                             {"role": "assistant", "content": g["answer"]}],
                "qa_id": g["id"], "case": "general", "k": 0, "nr": 0}

    train = f3_train + [as_row(g) for g in g_train]
    val = f3_val + [as_row(g) for g in g_val]
    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats = {
        "variant": "combo (recusa35 + anchor25)",
        "n_train": len(train), "n_val": len(val),
        "n_general_train": len(g_train), "n_closed": n_closed,
        "anchor_frac": round((n_closed + len(g_train)) / len(train), 4),
        "absent_frac": round(sum(1 for r in train if r["case"] == "absent")
                             / max(1, sum(1 for r in train if r["case"] != "general")), 4),
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# combo3: data2x + recusa35 + anchor25 (as 3 dominâncias ortogonais)
# ---------------------------------------------------------------------------
def build_combo3() -> None:
    """Converte gold->absent (35%) sobre o ASSIGN do data2x (distratores
    duros, mesma lógica do recusa35), re-renderiza e soma pares gerais até
    âncora 25%."""
    out_dir = OUT / "combo3"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = OUT / "data2x"
    assignments = read_jsonl(src_dir / "assignments.jsonl")
    # topk ampliado: F3 + f3b
    topk = {}
    for p in sorted((HERE / "data" / "retrieval").glob("topk_*.jsonl")):
        for r in read_jsonl(p):
            topk.setdefault(r["qa_id"], r)
    for p in sorted(F3_RETR.glob("topk_*.jsonl")):
        for r in read_jsonl(p):
            topk.setdefault(r["qa_id"], r)
    rng = random.Random(f"{SEED}:combo3")

    sys.path.insert(0, str(F3 / "synth"))
    from common_synth import norm_tokens  # noqa: PLC0415
    units: dict[str, dict] = {}
    for base in ((F3 / "data" / "synth"), (HERE / "data" / "synth"),
                 (REPO / "bench" / "lfm3_f2" / "data" / "synth")):
        for p in sorted(base.glob("units_nr*.jsonl")):
            for u in read_jsonl(p):
                units.setdefault(u["unit_id"], u)

    bd = load_f3_build_dataset()
    chunks = bd.load_chunk_texts()
    by_doc: dict[str, list[int]] = collections.defaultdict(list)
    for cid, c in chunks.items():
        by_doc[c["doc"].lower()].append(cid)

    n_total = len(assignments)
    n_absent_now = sum(1 for a in assignments if a["case"] == "absent")
    n_convert = int(n_total * ABSENT_FRAC_TARGET) - n_absent_now
    gold_rows = [a for a in assignments if a["case"] == "gold"]
    rng.shuffle(gold_rows)
    new_by_id, converted, skipped = {}, 0, 0
    for a in gold_rows:
        if converted >= n_convert:
            break
        tk = topk.get(a["qa_id"])
        if not tk:
            skipped += 1
            continue
        same_docs = {f"nr-{a['nr']:02d}", f"nr-{a['nr']}"}
        dis = [p for p in tk["topk"] if p["label"] == "distractor"]
        hard = [p for p in dis if p["doc"].lower() in same_docs]
        soft = [p for p in dis if p not in hard]
        u = units.get(a["unit_id"])
        ft = set(norm_tokens(u["fato"])) if u else set()
        want_hard = max(1, a["k"] // 2)
        if len(hard) < want_hard and ft:
            used = {p["id"] for p in tk["topk"]}
            cand = [cid for d in same_docs for cid in by_doc.get(d, [])
                    if cid not in used]
            rng.shuffle(cand)
            for cid in cand:
                if len(hard) >= want_hard:
                    break
                ct = set(norm_tokens(chunks[cid]["text"]))
                if len(ft & ct) / len(ft) < 0.30:
                    hard.append({"id": cid, "doc": chunks[cid]["doc"],
                                 "label": "distractor",
                                 "source": "index_hard_neg"})
        pool = (hard + soft)[: a["k"]]
        rng.shuffle(pool)
        if not pool:
            skipped += 1
            continue
        new_by_id[a["qa_id"]] = {
            **a, "case": "absent", "k": len(pool), "gold_position": None,
            "context_ids": [p["id"] for p in pool],
            "anchor_passage": None, "labels": [p["label"] for p in pool],
            "origin_absent": "gold_removed_hard",
        }
        converted += 1
    out_rows = [new_by_id.get(a["qa_id"], a) if a["case"] == "gold" else a
                for a in assignments]
    write_jsonl(out_dir / "assignments.jsonl", out_rows)
    shutil.copy(src_dir / "targets.jsonl", out_dir / "targets.jsonl")
    _render_variant(out_dir)

    # + âncora 25% com pares gerais
    f3_train = read_jsonl(out_dir / "train.jsonl")
    f3_val = read_jsonl(out_dir / "val.jsonl")
    general = read_jsonl(HERE / "data" / "general" / "general_pairs.jsonl")
    n_closed = sum(1 for r in f3_train if r["case"] == "closed")
    need = int((ANCHOR_FRAC_TARGET * len(f3_train) - n_closed)
               / (1 - ANCHOR_FRAC_TARGET))
    rng.shuffle(general)
    assert len(general) >= need + 20, f"faltam pares gerais: {need}+20 > {len(general)}"

    def as_row(g: dict) -> dict:
        return {"messages": [{"role": "user", "content": g["prompt"]},
                             {"role": "assistant", "content": g["answer"]}],
                "qa_id": g["id"], "case": "general", "k": 0, "nr": 0}

    train = f3_train + [as_row(g) for g in general[:need]]
    val = f3_val + [as_row(g) for g in general[need:need + 20]]
    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats = {
        "variant": "combo3 (data2x + recusa35 + anchor25)",
        "n_train": len(train), "n_val": len(val),
        "converted_gold_to_absent": converted, "skipped": skipped,
        "n_general_train": need, "n_closed": n_closed,
        "anchor_frac": round((n_closed + need) / len(train), 4),
        "absent_frac": round(sum(1 for r in train if r["case"] == "absent")
                             / max(1, sum(1 for r in train if r["case"] != "general")), 4),
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["recusa35", "anchor25", "data2x", "data2x-render",
                             "combo", "combo3"])
    args = ap.parse_args()
    {"recusa35": build_recusa35, "anchor25": build_anchor25,
     "data2x": build_data2x, "data2x-render": render_data2x,
     "combo": build_combo, "combo3": build_combo3}[args.variant]()


if __name__ == "__main__":
    main()
