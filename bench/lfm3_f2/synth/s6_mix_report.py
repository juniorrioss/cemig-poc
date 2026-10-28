"""
s6_mix_report.py — estágio (e)+(f) do funil: dedup (anticontaminação + intra),
métricas de diversidade e mistura final estilo pré-treino.

Dedup (thresholds REGISTRADOS aqui, aplicados antes do corte final):
  - contra as 1536 queries do benchmark_mobile: Jaccard >= 0.50 em tokens
    normalizados => descarte (anticontaminação, mesmo limiar da POC ~0.4-0.5;
    usamos 0.50 e publicamos o histograma para auditoria);
  - intra-conjunto (near-duplicates): assinatura dos 8 primeiros tokens
    normalizados + Jaccard >= 0.80 entre perguntas da MESMA unidade => mantém
    a primeira. A taxa de near-dup é métrica publicada (medida ANTES do filtro).

Diversidade publicada em data/synth/diversity_report.json (+ .md legível):
  - entropia (bits) dos n primeiros tokens das perguntas, n=3,4,5 — comparada
    com a entropia máxima possível (log2 do nº de prefixos distintos) e com a
    das 1536 do benchmark como referência;
  - distribuição personas / cenários / estilos / origem;
  - taxa de near-duplicates pré-filtro;
  - cobertura: % de unidades de conhecimento com >=1 QA aprovado.

Mistura final (data/synth/train_mix.jsonl), proporção documentada no README:
  - raw       : janelas cruas da norma (texto normativo literal)
  - paraphrase: reescritas multi-registro (s5)
  - qa        : pares aprovados pelo crítico (s4), pós-dedup

Uso:
    classifier/.venv/bin/python bench/lfm3_f2/synth/s6_mix_report.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_synth import (SYNTH, jaccard, load_norm_corpus, norm_tokens,
                          read_jsonl, treated_norms, write_jsonl)
from s1_decompose import split_windows

BENCH_DEDUP_JACCARD = 0.50   # anticontaminação vs benchmark
INTRA_DEDUP_JACCARD = 0.80   # near-dup intra-conjunto (mesma unidade)
PREFIX_SIG_N = 8             # tokens da assinatura de prefixo


def find_benchmark_parquet() -> str:
    pats = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "datasets--UFGCEMIGONA--benchmark_mobile/snapshots/*/"
                         "normas_regulamentadoras/train-00000-of-00001.parquet"))
    assert pats, "benchmark parquet não encontrado"
    return pats[0]


def prefix_entropy(questions: list[str], n: int) -> dict:
    """Entropia de Shannon (bits) do prefixo de n tokens normalizados."""
    prefixes = [" ".join(norm_tokens(q)[:n]) for q in questions]
    counts = collections.Counter(prefixes)
    total = sum(counts.values())
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return {"n": n, "entropy_bits": round(h, 3),
            "distinct_prefixes": len(counts), "total": total,
            "max_entropy_bits": round(math.log2(len(counts)), 3) if counts else 0.0,
            "top10": counts.most_common(10)}


def main() -> None:
    import pyarrow.parquet as pq
    bench = pq.read_table(find_benchmark_parquet()).to_pylist()
    bench_qs = [r["question"] for r in bench]
    bench_tok = [set(norm_tokens(q)) for q in bench_qs]

    all_qa, all_units, all_para = [], [], []
    for nr in treated_norms():
        all_qa.extend(read_jsonl(SYNTH / f"qa_passed_nr{nr}.jsonl"))
        all_units.extend(read_jsonl(SYNTH / f"units_nr{nr}.jsonl"))
        all_para.extend(read_jsonl(SYNTH / f"para_nr{nr}.jsonl"))

    print(f"entrada: {len(all_qa)} QAs aprovados pelo crítico, "
          f"{len(all_units)} unidades, {len(all_para)} parafraseios")

    # --- dedup anticontaminação (vs benchmark) --------------------------------
    kept, contaminated = [], []
    for qa in all_qa:
        qt = set(norm_tokens(qa["question"]))
        worst = max((len(qt & bt) / len(qt | bt) if qt | bt else 0.0)
                    for bt in bench_tok)
        if worst >= BENCH_DEDUP_JACCARD:
            contaminated.append(dict(qa, bench_jaccard=round(worst, 3)))
        else:
            kept.append(qa)
    print(f"anticontaminação: {len(contaminated)} descartados "
          f"(jaccard>={BENCH_DEDUP_JACCARD} vs benchmark)")

    # --- near-dup intra (medida ANTES do filtro, depois filtra) ---------------
    # agrupa por unidade; dentro da unidade compara pares (barato: unidades têm <40 QAs)
    n_near = 0
    by_unit = collections.defaultdict(list)
    for qa in kept:
        by_unit[qa["unit_id"]].append(qa)
    final = []
    for unit_qas in by_unit.values():
        chosen: list[dict] = []
        for qa in unit_qas:
            dup = any(jaccard(qa["question"], c["question"]) >= INTRA_DEDUP_JACCARD
                      for c in chosen)
            if dup:
                n_near += 1
            else:
                chosen.append(qa)
        final.extend(chosen)
    near_rate = n_near / max(1, len(kept))
    print(f"near-dup intra: {n_near} removidos ({near_rate:.1%})")

    # --- diversidade -----------------------------------------------------------
    questions = [qa["question"] for qa in final]
    report = {
        "thresholds": {"bench_jaccard": BENCH_DEDUP_JACCARD,
                       "intra_jaccard": INTRA_DEDUP_JACCARD,
                       "prefix_sig_n": PREFIX_SIG_N},
        "counts": {
            "qa_critic_passed": len(all_qa),
            "qa_contaminated_removed": len(contaminated),
            "qa_neardup_removed": n_near,
            "qa_final": len(final),
            "paraphrases": len(all_para),
            "units_total": len(all_units),
        },
        "near_dup_rate_preff": round(near_rate, 4),
        "prefix_entropy": {f"n{n}": prefix_entropy(questions, n) for n in (3, 4, 5)},
        "prefix_entropy_benchmark_ref": {
            f"n{n}": prefix_entropy(bench_qs, n) for n in (3, 4, 5)},
        "persona_dist": dict(collections.Counter(q["persona"] for q in final)),
        "cenario_dist": dict(collections.Counter(q["cenario"] for q in final)),
        "style_dist": dict(collections.Counter(q["style"] for q in final)),
        "origin_dist": dict(collections.Counter(q["origin"] for q in final)),
        "per_nr": dict(collections.Counter(q["nr"] for q in final)),
    }
    # cobertura por unidade
    covered = {qa["unit_id"] for qa in final}
    per_nr_cov = {}
    for nr in treated_norms():
        units_nr = [u for u in all_units if u["nr"] == nr]
        cov = sum(1 for u in units_nr if u["unit_id"] in covered)
        per_nr_cov[f"NR-{nr}"] = {"units": len(units_nr), "covered": cov,
                                  "coverage": round(cov / max(1, len(units_nr)), 3)}
    report["unit_coverage"] = per_nr_cov

    (SYNTH / "diversity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(SYNTH / "qa_contaminated.jsonl", contaminated)

    # --- mistura final ----------------------------------------------------------
    mix = []
    for nr in treated_norms():
        corpus = load_norm_corpus(nr)
        for i, w in enumerate(split_windows(corpus["conteudo"])):
            mix.append({"kind": "raw", "nr": nr, "key": f"raw:{nr}:{i}",
                        "text": w})
    for p in all_para:
        mix.append({"kind": "paraphrase", "nr": p["nr"],
                    "key": f"para:{p['para_id']}", "text": p["text"]})
    for qa in final:
        mix.append({"kind": "qa", "nr": qa["nr"], "key": f"qa:{qa['qa_id']}",
                    "question": qa["question"], "answer": qa["answer"],
                    "ref": qa["ref"], "persona": qa["persona"],
                    "style": qa["style"], "origin": qa["origin"],
                    "unit_id": qa["unit_id"]})
    write_jsonl(SYNTH / "train_mix.jsonl", mix)
    kinds = collections.Counter(m["kind"] for m in mix)
    report["mix"] = dict(kinds)
    (SYNTH / "diversity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # resumo legível
    lines = ["# Relatório de diversidade — F2 geração sintética", ""]
    lines.append(f"- QAs aprovados pelo crítico: {len(all_qa)}")
    lines.append(f"- Removidos por contaminação (J>={BENCH_DEDUP_JACCARD} vs benchmark): {len(contaminated)}")
    lines.append(f"- Near-dups intra removidos (J>={INTRA_DEDUP_JACCARD}): {n_near} ({near_rate:.1%})")
    lines.append(f"- QAs finais: {len(final)}; parafraseios: {len(all_para)}; mistura: {dict(kinds)}")
    lines.append("")
    lines.append("| prefixo | entropia (bits) | máx possível | distintos | benchmark ref |")
    lines.append("|---|---|---|---|---|")
    for n in (3, 4, 5):
        e = report["prefix_entropy"][f"n{n}"]
        b = report["prefix_entropy_benchmark_ref"][f"n{n}"]
        lines.append(f"| n={n} | {e['entropy_bits']} | {e['max_entropy_bits']} "
                     f"| {e['distinct_prefixes']} | {b['entropy_bits']} |")
    lines.append("")
    for k in ("persona_dist", "cenario_dist", "style_dist", "origin_dist", "per_nr"):
        lines.append(f"- {k}: {report[k]}")
    lines.append(f"- cobertura por unidade: {per_nr_cov}")
    (SYNTH / "diversity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] diversity_report.json/.md + train_mix.jsonl ({len(mix)} docs)")


if __name__ == "__main__":
    main()
