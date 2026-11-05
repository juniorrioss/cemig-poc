#!/usr/bin/env python3
"""
make_report.py — histogramas em grade das rubricas do decision model por pipeline avaliado.

Grade: LINHAS = pipelines avaliados (5 retrievers do benchmark B: naive, bm25, mobilerag,
mobilerag_plus, ours=v4/F6-op; + as 2 células reais do NOSSO pipeline F6 (f4-r32-ep2 op/ctrl,
conjunto R)); COLUNAS = eixos da rubrica (rubric_j.py); EIXO X = nota [0,1] (1 = bom).
Uma figura por decision model (vela3-flash = principal; pplx-decider = comparação).

Saídas (nesta pasta):
  rubricas_<model>.png          figura-grade
  scores_long.parquet           1 linha por (julgador, pipeline, item, eixo) com a nota
  summary.csv                   n / média / mediana / p10 / p90 / %≥0,5 por célula
  hist_counts.csv               contagens dos 10 bins [0,0.1)…[0.9,1.0] por célula
  verdicts_by_pipeline.csv      vereditos binários de referência (Qwen-régua, gpt-oss, gate calibrado)

Uso: python make_report.py   (precisa de matplotlib/pandas/pyarrow; ver RUBRICAS.md)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402
import common_j as C  # noqa: E402
from rubric_j import read_span_unsupported  # noqa: E402

AXES = [  # (coluna, rótulo no gráfico)
    ("context_sufficient", "contexto suficiente\n(retriever)"),
    ("verdict", "veredito\n(correta=1, parcial=0,5)"),
    ("hallucination", "sem alucinação\n(1 − p_aluc)"),
    ("refusal", "respondeu\n(1 − p_recusa)"),
    ("fact_coverage", "cobertura de fatos\n(média p(fato))"),
    ("answers_question", "responde à\npergunta"),
    ("grounded", "apoiada no\ncontexto"),
    ("citation", "cita NR/item\n(informativo)"),
    ("span_unsupported", "span: fração apoiada\n(só Vela)"),
    ("gate", "GATE\nver×(1−aluc)×(1−rec)"),
]
ROWS = [  # (rótulo, conjunto, filtro)
    ("naive", "B", "naive"), ("bm25", "B", "bm25"), ("mobilerag", "B", "mobilerag"),
    ("mobilerag_plus", "B", "mobilerag_plus"), ("ours (v4/F6-op)\nsó contexto", "B", "ours"),
    ("F6 op: f4-r32-ep2\n(resposta real)", "R", "op"), ("F6 ctrl: f4-r32-ep2\n(resposta real)", "R", "ctrl"),
]
BINS = [i / 10 for i in range(11)]


def long_scores(model: str, sets: dict) -> pd.DataFrame:
    rows = []
    for s in ("B", "R"):
        D = A.load_decider(model, 1, s)
        for it in sets[s]:
            d = D.get(it["uid"])
            if not d:
                continue
            pipe = it["method"] if s == "B" else it["cell"]
            vals = {ax: d.get(ax) for ax in ("context_sufficient", "verdict", "hallucination", "refusal",
                                              "answers_question", "grounded", "citation")}
            g = A.gate_scores(d)
            vals["gate"] = g.get("gate")
            vals["fact_coverage"] = g.get("cov")
            if "_span_ans" in d:
                vals["span_unsupported"] = read_span_unsupported(d["_span_ans"], it["answer"] or "")
            for ax, v in vals.items():
                if v is not None:
                    rows.append({"judge": model, "set": s, "pipeline": pipe, "uid": it["uid"],
                                 "axis": ax, "score": float(v)})
    return pd.DataFrame(rows)


def ref_verdicts(sets: dict, cons: dict) -> pd.DataFrame:
    """Taxas binárias de referência por pipeline (para ler ao lado dos histogramas)."""
    out = []
    qB, qR = A.load_qwen("qwen", 1, "B"), A.load_qwen("qwen", 1, "R")
    for _, s, f in ROWS:
        its = [it for it in sets[s] if (it["method"] if s == "B" else it["cell"]) == f and it.get("answer")]
        if not its:
            continue
        q = qB if s == "B" else qR
        r = {"pipeline": f, "set": s, "n": len(its),
             "qwen_regua_aprovado": round(sum(q[i["uid"]]["approved"] for i in its if i["uid"] in q) / len(its), 4)}
        if s == "B":
            r["gptoss_correct"] = round(sum(i["orig"]["gptoss"] == "correct" for i in its) / len(its), 4)
        for m in ("vela", "pplx"):
            D = A.load_decider(m, 1, s)
            t = cons["agree_R_vs_qwen"][f"{m}:gate"]["threshold"]
            g = [A.gate_scores(D[i["uid"]]).get("gate") for i in its if i["uid"] in D]
            r[f"{m}_gate>=limiar({t})"] = round(sum(x >= t for x in g if x is not None) / len(g), 4)
        out.append(r)
    return pd.DataFrame(out)


def plot(df: pd.DataFrame, model: str, refs: pd.DataFrame) -> Path:
    nr, nc = len(ROWS), len(AXES)
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc, 1.55 * nr + 1.2), sharex=True)
    for i, (rl, s, f) in enumerate(ROWS):
        sub = df[(df["set"] == s) & (df["pipeline"] == f)]
        for j, (ax_name, al) in enumerate(AXES):
            a = axs[i][j]
            v = sub[sub["axis"] == ax_name]["score"]
            if len(v):
                a.hist(v, bins=BINS)
                a.set_title(f"μ={v.mean():.2f}  n={len(v)}", fontsize=7, pad=2)
            else:
                a.text(0.5, 0.5, "n/d", ha="center", va="center", transform=a.transAxes, fontsize=8)
            a.tick_params(labelsize=6)
            a.set_yticks([])
            if i == 0:
                a.annotate(al, xy=(0.5, 1.32), xycoords="axes fraction", ha="center", va="bottom", fontsize=7.5)
            if j == 0:
                ref = refs[refs["pipeline"] == f]
                extra = ""
                if len(ref):
                    r0 = ref.iloc[0]
                    extra = f"\nQwen aprov {r0['qwen_regua_aprovado']:.0%}"
                    if "gptoss_correct" in r0 and pd.notna(r0.get("gptoss_correct")):
                        extra += f"\ngpt-oss ok {r0['gptoss_correct']:.0%}"
                a.set_ylabel(rl + extra, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
            if i == nr - 1:
                a.set_xlabel("nota (1 = bom)", fontsize=6)
    name = {"vela": "vela3-flash", "pplx": "pplx-decider-v1.1-27b"}[model]
    fig.suptitle(f"Rubrica de decisão ({name}) — distribuição das notas por pipeline × eixo "
                 f"(B: benchmark_mobile 300 perguntas; R: F6 recorte 344)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / f"rubricas_{model}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    sets = {s: C.read_jsonl(C.DATA / "sets" / f"set_{s}.jsonl") for s in ("B", "R")}
    cons = json.loads((C.DATA / "consolidation.json").read_text())
    refs = ref_verdicts(sets, cons)
    dfs = [long_scores(m, sets) for m in ("vela", "pplx")]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(HERE / "scores_long.parquet", index=False)
    g = df.groupby(["judge", "set", "pipeline", "axis"])["score"]
    summ = g.agg(n="count", mean="mean", median="median",
                 p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9),
                 frac_ge_05=lambda x: (x >= 0.5).mean()).round(4).reset_index()
    summ.to_csv(HERE / "summary.csv", index=False)
    hc = df.assign(bin=pd.cut(df["score"], BINS, include_lowest=True, right=False).astype(str))
    hc.groupby(["judge", "set", "pipeline", "axis", "bin"], observed=True).size().rename("count") \
      .reset_index().to_csv(HERE / "hist_counts.csv", index=False)
    refs.to_csv(HERE / "verdicts_by_pipeline.csv", index=False)
    for m in ("vela", "pplx"):
        print(plot(df[df["judge"] == m], m, refs))
    print(summ[summ["judge"] == "vela"].pivot_table(index="pipeline", columns="axis", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
