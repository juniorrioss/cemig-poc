#!/usr/bin/env python3
"""
analyze.py — consolida o comparativo julgador × {AUROC variantes, kappa, determinismo,
velocidade, custo} + retrievers (tarefa extra) → data/consolidation.json.

Julgadores:
  qwen   = régua honesta de produção (fatos determinísticos + desempate Qwen27b, temp 0);
  pplx   = pplx-decider-v1.1-27b com a rubrica rubric_j (choice/noul);
  vela   = vela3-flash, mesma rubrica (+ eixo extra span, só nele);
  gptoss = veredito ORIGINAL do gpt-oss-120b no conjunto B (árbitro independente).

Nota contínua do decider para o gate ("aprovado"):
  gate_score = verdict_EV × (1 − p_aluc) × (1 − p_recusa)  — análogo multiplicativo da
  régua (cobertura ∧ ¬aluc ∧ ¬recusa). Também reportamos o gate com cobertura por fato
  (`gate_facts` = min(1, cov/0,5) × (1−p_aluc) × (1−p_recusa)). O limiar binário é
  calibrado num split de calibração (metade dos base_qa_id do R) e aplicado na outra
  metade — kappa honesto, sem ajustar no próprio teste.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_j as C  # noqa: E402
from rubric_j import AXES, read_answer, read_span_unsupported  # noqa: E402

RAW = C.DATA / "raw"
GOOD_V = {"gold", "paraphrase", "extra_true"}           # verdade: aprovável
BAD_V = {"wrong_number", "wrong_item", "negation", "swap_obligation", "hallucination_added",
         "refusal", "tautology", "off_topic"}           # verdade: reprovável
# `incomplete` fica fora do gate binário (a régua aprova se cobrir ≥50% dos fatos).


# ----------------------------------------------------------------------------- métricas
def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # Mann-Whitney com empates (rank médio)
    ranks, i = {}, 0
    rk = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            rk[k] = (i + j) / 2 + 1
        i = j + 1
    rpos = sum(r for r, (_, y) in zip(rk, allv) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2) / (n1 * n0)


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def acc(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else None


def best_threshold(scores: list[float], labels: list[bool]) -> float:
    """Limiar que maximiza kappa contra o rótulo de referência (só no split de calibração)."""
    cands = sorted(set(round(s, 4) for s in scores))
    best, bt = -2, 0.5
    for t in cands:
        k = kappa([s >= t for s in scores], labels)
        if k is not None and k > best:
            best, bt = k, t
    return bt


def boot_ci(fn, pairs, n=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        smp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = fn(smp)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)) if vals else None


# ----------------------------------------------------------------------------- leitura
def load_decider(model: str, rep: int, s: str) -> dict[str, dict]:
    """{uid: {axis: score_médio_paráfrases, axis+'_spread':..., 'facts':[p...], 'raw':{key:ans}}}"""
    out: dict[str, dict] = defaultdict(lambda: {"_per": defaultdict(list), "facts": [], "raw": {}})
    txt = C.read_text_any(RAW / f"{model}_rep{rep}_{s}.jsonl")
    if txt is None:
        return {}
    for l in txt.splitlines():
        d = json.loads(l)
        uid, axis, sub = d["key"].rsplit("|", 2)
        o = out[uid]
        o["raw"][d["key"]] = d["answer"]
        if axis == "fact":
            v = read_answer("fact", d["answer"])
            if v is not None:
                o["facts"].append(v)
        elif axis == "span_unsupported":
            o["_span_ans"] = d["answer"]
        else:
            v = read_answer(axis, d["answer"])
            if v is not None:
                o["_per"][axis].append(v)
    res = {}
    for uid, o in out.items():
        r = {"facts": o["facts"], "raw": o["raw"]}
        for ax, vs in o["_per"].items():
            r[ax] = mean(vs)
            r[ax + "_spread"] = max(vs) - min(vs) if len(vs) > 1 else 0.0
        if "_span_ans" in o:
            r["_span_ans"] = o["_span_ans"]
        res[uid] = r
    return res


def gate_scores(d: dict) -> dict:
    v = d.get("verdict")
    h = d.get("hallucination")   # já em 'nota boa' (1-p)
    rf = d.get("refusal")
    if v is None or h is None or rf is None:
        return {}
    cov = mean(d["facts"]) if d["facts"] else None
    out = {"gate": v * h * rf}
    if cov is not None:
        out["cov"] = cov
        out["gate_facts"] = min(1.0, cov / 0.5) * h * rf
    return out


def load_qwen(tag: str, rep: int, s: str) -> dict[str, dict]:
    return {r["uid"]: r for r in C.read_jsonl(RAW / f"{tag}_rep{rep}_{s}.jsonl")}


# ----------------------------------------------------------------------------- análises
def variants_block(models, sets_v):
    """AUROC por julgador: gate (good vs bad) e por eixo (good/bad próprios do eixo)."""
    res = {}
    items = {it["uid"]: it for it in sets_v}
    for m, D in models.items():
        r = {}
        if m == "qwen":
            pos = [1.0 if D[u]["approved"] else 0.0 for u, it in items.items() if it["variant"] in GOOD_V and u in D]
            neg = [1.0 if D[u]["approved"] else 0.0 for u, it in items.items() if it["variant"] in BAD_V and u in D]
            r["gate_auroc"] = auroc(pos, neg)
            # Qwen é binário: AUROC = (TPR+TNR)/2. Também balanced accuracy por variante.
            per_v = defaultdict(list)
            for u, it in items.items():
                if u in D:
                    per_v[it["variant"]].append(D[u]["approved"])
            r["approve_rate_by_variant"] = {k: round(mean(v), 3) for k, v in sorted(per_v.items())}
            hpos = [0.0 if D[u]["hallucination"] else 1.0 for u, it in items.items()
                    if it["variant"] in AXES["hallucination"]["good"] and u in D]
            hneg = [0.0 if D[u]["hallucination"] else 1.0 for u, it in items.items()
                    if it["variant"] in AXES["hallucination"]["bad"] and u in D]
            r["axis_auroc"] = {"hallucination": auroc(hpos, hneg)}
        else:
            G = {u: gate_scores(D[u]) for u in D}
            for g in ("gate", "gate_facts"):
                pos = [G[u][g] for u, it in items.items() if it["variant"] in GOOD_V and g in G.get(u, {})]
                neg = [G[u][g] for u, it in items.items() if it["variant"] in BAD_V and g in G.get(u, {})]
                r[f"{g}_auroc"] = auroc(pos, neg)
            ax_res, spread = {}, {}
            for ax, cfg in AXES.items():
                if not cfg["good"]:
                    continue
                pos = [D[u][ax] for u, it in items.items() if it["variant"] in cfg["good"] and ax in D.get(u, {})]
                neg = [D[u][ax] for u, it in items.items() if it["variant"] in cfg["bad"] and ax in D.get(u, {})]
                ax_res[ax] = auroc(pos, neg)
                sp = [D[u][ax + "_spread"] for u in D if ax + "_spread" in D[u]]
                if sp and len(cfg["paraphrases"]) > 1:
                    spread[ax] = round(mean(sp), 4)
            # cobertura por fato: good vs incomplete/refusal/tautology/off_topic
            pos = [G[u]["cov"] for u, it in items.items() if it["variant"] in GOOD_V and "cov" in G.get(u, {})]
            neg = [G[u]["cov"] for u, it in items.items()
                   if it["variant"] in {"refusal", "tautology", "off_topic"} and "cov" in G.get(u, {})]
            ax_res["facts_coverage"] = auroc(pos, neg)
            # eixo extra span (só vela)
            sp_pos, sp_neg = [], []
            for u, it in items.items():
                a = D.get(u, {}).get("_span_ans")
                if a is None:
                    continue
                v = read_span_unsupported(a, it["answer"])
                if v is None:
                    continue
                if it["variant"] in AXES["hallucination"]["good"]:
                    sp_pos.append(v)
                elif it["variant"] in AXES["hallucination"]["bad"]:
                    sp_neg.append(v)
            if sp_pos:
                ax_res["span_unsupported (extra, só Vela)"] = auroc(sp_pos, sp_neg)
            r["axis_auroc"] = {k: (round(v, 4) if v is not None else None) for k, v in ax_res.items()}
            r["paraphrase_spread"] = spread
            # gate médio por variante (diagnóstico)
            per_v = defaultdict(list)
            for u, it in items.items():
                if "gate" in G.get(u, {}):
                    per_v[it["variant"]].append(G[u]["gate"])
            r["gate_mean_by_variant"] = {k: round(mean(v), 3) for k, v in sorted(per_v.items())}
        res[m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    return res


def agreement_block(set_items, qwen, deciders, key_ref="qwen", calib_frac=0.5, seed=7):
    """Kappa de cada decider vs referência, com limiar calibrado em metade dos base_qa_id."""
    groups = sorted({it["base_qa_id"] for it in set_items})
    rng = random.Random(seed)
    calib = set(rng.sample(groups, int(len(groups) * calib_frac)))
    out = {}
    for m, D in deciders.items():
        G = {u: gate_scores(D[u]) for u in D}
        for g in ("gate", "gate_facts"):
            cal = [(G[it["uid"]][g], qwen[it["uid"]]) for it in set_items
                   if it["base_qa_id"] in calib and g in G.get(it["uid"], {}) and it["uid"] in qwen]
            tst = [(G[it["uid"]][g], qwen[it["uid"]], it["uid"]) for it in set_items
                   if it["base_qa_id"] not in calib and g in G.get(it["uid"], {}) and it["uid"] in qwen]
            if not cal or not tst:
                continue
            t = best_threshold([s for s, _ in cal], [y for _, y in cal])
            pred = [s >= t for s, _, _ in tst]
            ref = [y for _, y, _ in tst]
            pairs = list(zip(pred, ref))
            out[f"{m}:{g}"] = {
                "threshold": t, "n_test": len(tst), "kappa": round(kappa(pred, ref), 4),
                "kappa_ci95": boot_ci(lambda p: kappa([a for a, _ in p], [b for _, b in p]), pairs),
                "accuracy": round(acc(pred, ref), 4), "auroc_vs_ref": round(auroc(
                    [s for s, y, _ in tst if y], [s for s, y, _ in tst if not y]) or 0, 4),
                "approve_rate_decider": round(mean(pred), 4), "approve_rate_ref": round(mean(ref), 4),
                "_pred": {u: bool(s >= t) for s, _, u in tst}, "_score": {u: s for s, _, u in tst},
            }
    return out


def determinism_block(pairs_spec):
    """pairs_spec: {nome: (dictA, dictB, campo)} -> frac idêntica."""
    out = {}
    for name, (A, B, kind) in pairs_spec.items():
        common = sorted(set(A) & set(B))
        if not common:
            continue
        if kind == "decider_raw":
            ka = {k: v for u in common for k, v in A[u]["raw"].items()}
            kb = {k: v for u in common for k, v in B[u]["raw"].items()}
            ks = sorted(set(ka) & set(kb))
            same = sum(json.dumps(ka[k], sort_keys=True) == json.dumps(kb[k], sort_keys=True) for k in ks)
            maxdiff = 0.0
            for k in ks:
                a, b = ka[k], kb[k]
                if "probabilities" in a:
                    maxdiff = max(maxdiff, max(abs(a["probabilities"][x] - b["probabilities"][x]) for x in a["probabilities"]))
                elif "noul" in a:
                    maxdiff = max(maxdiff, abs(a["noul"] - b["noul"]))
            out[name] = {"rows": len(ks), "bit_identical_frac": round(same / len(ks), 6),
                         "max_abs_prob_diff": maxdiff}
        else:
            same_v = sum(A[u]["approved"] == B[u]["approved"] for u in common)
            same_raw = sum(A[u]["raw"] == B[u]["raw"] for u in common)
            same_h = sum(A[u]["hallucination"] == B[u]["hallucination"] for u in common)
            out[name] = {"items": len(common), "verdict_identical_frac": round(same_v / len(common), 4),
                         "raw_text_identical_frac": round(same_raw / len(common), 4),
                         "hallucination_identical_frac": round(same_h / len(common), 4),
                         "kappa_rep1_rep2": round(kappa([A[u]["approved"] for u in common],
                                                        [B[u]["approved"] for u in common]) or 0, 4)}
    return out


def speed_block():
    rows = C.read_jsonl(RAW / "timing.jsonl")
    agg = defaultdict(lambda: {"items": 0, "rows": 0, "sec": 0.0})
    for r in rows:
        k = (r["model"], r.get("temperature", 0.0) if r["model"].startswith("qwen") else None)
        a = agg[r["model"]]
        a["items"] += r["items"]
        a["rows"] += r.get("rows", 0)
        a["sec"] += r.get("gpu_seconds", r.get("wall_seconds", 0.0))
    out = {}
    for m, a in agg.items():
        out[m] = {"items": a["items"], "decision_rows": a["rows"], "seconds": round(a["sec"], 1),
                  "items_per_s": round(a["items"] / a["sec"], 3) if a["sec"] else None,
                  "rows_per_s": round(a["rows"] / a["sec"], 2) if a["sec"] and a["rows"] else None}
    return out


def retrievers_block(set_b, deciders, qwen_b):
    """5 retrievers × {context_sufficient (decider), ev_cov lexical, resposta (4 deles)}."""
    by_m = defaultdict(list)
    for it in set_b:
        by_m[it["method"]].append(it)
    out = {}
    for m, its in by_m.items():
        r = {"n": len(its), "ev_cov_lexical": round(mean(x["ev_cov"] for x in its), 4),
             "ev_cov>=0.7": round(mean(x["ev_cov"] >= 0.7 for x in its), 4)}
        for dm, D in deciders.items():
            cs = [D[x["uid"]]["context_sufficient"] for x in its if "context_sufficient" in D.get(x["uid"], {})]
            if cs:
                r[f"{dm}:context_sufficient"] = round(mean(cs), 4)
                r[f"{dm}:context_sufficient>=0.5"] = round(mean(c >= 0.5 for c in cs), 4)
            g = [gate_scores(D[x["uid"]]).get("gate") for x in its if x.get("answer") and x["uid"] in D]
            g = [v for v in g if v is not None]
            if g:
                r[f"{dm}:answer_gate_mean"] = round(mean(g), 4)
        if m != "ours":
            r["gptoss_score"] = round(mean(x["orig"]["gptoss_score"] for x in its), 4)
            r["qwen_rubric_correct"] = round(mean(bool(x["orig"]["qwen_rubric"]) for x in its), 4)
            q = [qwen_b[x["uid"]]["approved"] for x in its if x["uid"] in qwen_b]
            if q:
                r["qwen_regua_approved"] = round(mean(q), 4)
        out[m] = r
    # pareado ours vs mobilerag_plus no context_sufficient (mesmas perguntas)
    pair = {}
    for dm, D in deciders.items():
        a = {x["qid"]: D[x["uid"]].get("context_sufficient") for x in by_m.get("ours", []) if x["uid"] in D}
        for other in ("mobilerag_plus", "naive"):
            b = {x["qid"]: D[x["uid"]].get("context_sufficient") for x in by_m.get(other, []) if x["uid"] in D}
            ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]
            if not ks:
                continue
            win = sum(a[k] - b[k] > 0.05 for k in ks)
            loss = sum(b[k] - a[k] > 0.05 for k in ks)
            pair[f"{dm}: ours vs {other}"] = {"n": len(ks), "mean_diff": round(mean(a[k] - b[k] for k in ks), 4),
                                              "wins": win, "losses": loss, "ties": len(ks) - win - loss,
                                              "sign_p": sign_p(win, loss)}
    a = {x["qid"]: x["ev_cov"] for x in by_m.get("ours", [])}
    for other in ("mobilerag_plus", "naive"):
        b = {x["qid"]: x["ev_cov"] for x in by_m.get(other, [])}
        ks = [k for k in a if k in b]
        win = sum(a[k] - b[k] > 0.05 for k in ks)
        loss = sum(b[k] - a[k] > 0.05 for k in ks)
        pair[f"lexical ev_cov: ours vs {other}"] = {"n": len(ks), "mean_diff": round(mean(a[k] - b[k] for k in ks), 4),
                                                    "wins": win, "losses": loss, "sign_p": sign_p(win, loss)}
    return {"by_method": out, "paired": pair}


def _test_split(items, seed=7):
    groups = sorted({it["base_qa_id"] for it in items})
    calib = set(random.Random(seed).sample(groups, int(len(groups) * 0.5)))
    return [it for it in items if it["base_qa_id"] not in calib]


def arbitration_block(b_ans, qB, dB, gpt, cons):
    """Árbitro independente (gpt-oss-120b, 'correct' = aprovado) sobre o split de teste do B.
    (1) limiar TRANSFERIDO: calibrado no R contra o Qwen (nunca viu o árbitro) e aplicado no B;
    (2) em TODAS as discordâncias decider×Qwen-régua, quem concorda com o gpt-oss."""
    tst = _test_split(b_ans)
    out = {}
    ref = [gpt[it["uid"]] for it in tst]
    qv = [qB[it["uid"]]["approved"] for it in tst]
    out["qwen_regua"] = {"kappa": round(kappa(qv, ref), 4), "accuracy": round(acc(qv, ref), 4),
                         "approve_rate": round(mean(qv), 4)}
    qr = [bool(it["orig"]["qwen_rubric"]) for it in tst]
    out["qwen_rubric_outro_time"] = {"kappa": round(kappa(qr, ref), 4), "accuracy": round(acc(qr, ref), 4),
                                     "approve_rate": round(mean(qr), 4)}
    for m, D in dB.items():
        t = cons["agree_R_vs_qwen"][f"{m}:gate"]["threshold"]
        pv = [gate_scores(D[it["uid"]])["gate"] >= t for it in tst]
        out[f"{m}:limiar_transferido_R"] = {"threshold": t, "kappa": round(kappa(pv, ref), 4),
                                            "kappa_ci95": boot_ci(lambda p: kappa([a for a, _ in p], [b for _, b in p]), list(zip(pv, ref))),
                                            "accuracy": round(acc(pv, ref), 4), "approve_rate": round(mean(pv), 4)}
        dis = [(p, q, r) for p, q, r in zip(pv, qv, ref) if p != q]
        out[f"{m}:discordancias_vs_qwen"] = {"n": len(dis), "decider_certo": sum(p == r for p, _, r in dis),
                                             "qwen_certo": sum(q == r for _, q, r in dis),
                                             "decider_aprova_qwen_reprova": sum(p for p, _, _ in dis),
                                             "decider_aprova_qwen_reprova__gptoss_aprova": sum(p and r for p, _, r in dis),
                                             "qwen_aprova_decider_reprova__gptoss_aprova": sum(q and r for _, q, r in dis)}
    return out


def hybrid_block(b_ans, qB, dB, gpt):
    """Triagem: decider decide quando confiante (gate <= lo ou >= hi), resto vai ao Qwen.
    Faixa (lo, hi) escolhida no split de CALIBRAÇÃO contra o QWEN (não vê o árbitro),
    exigindo concordância >= 97% com o Qwen fora da faixa; avaliada no teste contra o gpt-oss."""
    groups = sorted({it["base_qa_id"] for it in b_ans})
    calib = set(random.Random(7).sample(groups, int(len(groups) * 0.5)))
    cal = [it for it in b_ans if it["base_qa_id"] in calib]
    tst = [it for it in b_ans if it["base_qa_id"] not in calib]
    out = {}
    for m, D in dB.items():
        g = {it["uid"]: gate_scores(D[it["uid"]])["gate"] for it in b_ans}
        best = None
        grid = [x / 100 for x in range(1, 100)]
        for lo in grid:
            for hi in grid:
                if hi <= lo:
                    continue
                conf = [it for it in cal if g[it["uid"]] <= lo or g[it["uid"]] >= hi]
                if not conf:
                    continue
                agree = mean((g[it["uid"]] >= hi) == qB[it["uid"]]["approved"] for it in conf)
                cov = len(conf) / len(cal)
                if agree >= 0.97 and (best is None or cov > best[2]):
                    best = (lo, hi, cov)
        if not best:
            continue
        lo, hi, _ = best
        pred, to_qwen = [], 0
        for it in tst:
            s = g[it["uid"]]
            if s <= lo:
                pred.append(False)
            elif s >= hi:
                pred.append(True)
            else:
                pred.append(qB[it["uid"]]["approved"])
                to_qwen += 1
        ref = [gpt[it["uid"]] for it in tst]
        out[m] = {"lo": lo, "hi": hi, "frac_decidido_pelo_decider": round(1 - to_qwen / len(tst), 4),
                  "kappa_vs_gptoss": round(kappa(pred, ref), 4), "accuracy_vs_gptoss": round(acc(pred, ref), 4),
                  "n_test": len(tst)}
    return out


def context_validation(set_b, dB):
    """O eixo context_sufficient separa contexto COM evidência (ev_cov>=0,9) de SEM (ev_cov<0,3)?"""
    out = {}
    for m, D in dB.items():
        pos = [D[x["uid"]]["context_sufficient"] for x in set_b if x["ev_cov"] >= 0.9 and "context_sufficient" in D.get(x["uid"], {})]
        neg = [D[x["uid"]]["context_sufficient"] for x in set_b if x["ev_cov"] < 0.3 and "context_sufficient" in D.get(x["uid"], {})]
        # e contra o veredito do gpt-oss da resposta (proxy: contexto bom -> resposta correta), só os 4 deles
        pg = [D[x["uid"]]["context_sufficient"] for x in set_b if x["method"] != "ours" and x["orig"]["gptoss"] == "correct"]
        ng = [D[x["uid"]]["context_sufficient"] for x in set_b if x["method"] != "ours" and x["orig"]["gptoss"] == "incorrect"]
        out[m] = {"auroc_evcov_hi_vs_lo": round(auroc(pos, neg) or 0, 4), "n_pos": len(pos), "n_neg": len(neg),
                  "auroc_vs_gptoss_answer": round(auroc(pg, ng) or 0, 4)}
    return out


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 5)


def main() -> None:
    sets = {s: C.read_jsonl(C.DATA / "sets" / f"set_{s}.jsonl") for s in "VRB"}
    deciders_names = [m for m in ("pplx", "vela") if C.read_text_any(RAW / f"{m}_rep1_V.jsonl") is not None]
    cons: dict = {"deciders": deciders_names}
    # --- V
    mv = {"qwen": load_qwen("qwen", 1, "V")}
    for m in deciders_names:
        mv[m] = load_decider(m, 1, "V")
    cons["variants"] = variants_block(mv, sets["V"])
    cons["variants_n"] = dict(Counter(it["variant"] for it in sets["V"]))
    # --- R (real) e B (benchmark) — concordância vs Qwen e vs gpt-oss
    qR = load_qwen("qwen", 1, "R")
    dR = {m: load_decider(m, 1, "R") for m in deciders_names}
    cons["agree_R_vs_qwen"] = agreement_block(sets["R"], {u: r["approved"] for u, r in qR.items()}, dR)
    qB = load_qwen("qwen", 1, "B")
    dB = {m: load_decider(m, 1, "B") for m in deciders_names}
    b_ans = [it for it in sets["B"] if it.get("answer")]
    gpt = {it["uid"]: it["orig"]["gptoss"] == "correct" for it in b_ans}
    cons["agree_B_vs_gptoss"] = agreement_block(b_ans, gpt, dB)
    cons["agree_B_vs_qwen"] = agreement_block(b_ans, {u: r["approved"] for u, r in qB.items()}, dB)
    # Qwen (régua e rubrica) vs gpt-oss: no conjunto inteiro E no MESMO split de teste
    # dos deciders (calib por base_qa_id, seed 7) — comparação justa com o kappa do decider.
    groups = sorted({it["base_qa_id"] for it in b_ans})
    calib = set(random.Random(7).sample(groups, int(len(groups) * 0.5)))
    tst = [it for it in b_ans if it["uid"] in qB and it["base_qa_id"] not in calib]
    cons["qwen_vs_gptoss_B_testsplit"] = {
        "n": len(tst),
        "qwen_regua:kappa": round(kappa([qB[it["uid"]]["approved"] for it in tst], [gpt[it["uid"]] for it in tst]), 4),
        "qwen_regua:kappa_ci95": boot_ci(lambda p: kappa([a for a, _ in p], [b for _, b in p]),
                                         [(qB[it["uid"]]["approved"], gpt[it["uid"]]) for it in tst]),
        "qwen_rubric:kappa": round(kappa([bool(it["orig"]["qwen_rubric"]) for it in tst], [gpt[it["uid"]] for it in tst]), 4),
        "qwen_rubric:kappa_ci95": boot_ci(lambda p: kappa([a for a, _ in p], [b for _, b in p]),
                                          [(bool(it["orig"]["qwen_rubric"]), gpt[it["uid"]]) for it in tst])}
    # idem no R: Qwen original (F6) vs rerun é o "teto de auto-concordância"
    common = [it for it in b_ans if it["uid"] in qB]
    cons["qwen_vs_gptoss_B"] = {
        "qwen_regua:kappa": round(kappa([qB[it["uid"]]["approved"] for it in common], [gpt[it["uid"]] for it in common]), 4),
        "qwen_regua:accuracy": round(acc([qB[it["uid"]]["approved"] for it in common], [gpt[it["uid"]] for it in common]), 4),
        "qwen_rubric:kappa": round(kappa([bool(it["orig"]["qwen_rubric"]) for it in common], [gpt[it["uid"]] for it in common]), 4),
        "qwen_rubric:accuracy": round(acc([bool(it["orig"]["qwen_rubric"]) for it in common], [gpt[it["uid"]] for it in common]), 4),
        "n": len(common)}
    # kappa entre deciders (R e B)
    if len(deciders_names) == 2:
        for nm, block in (("R", cons["agree_R_vs_qwen"]), ("B", cons["agree_B_vs_gptoss"])):
            a, b = block.get("pplx:gate", {}).get("_pred", {}), block.get("vela:gate", {}).get("_pred", {})
            ks = sorted(set(a) & set(b))
            if ks:
                cons[f"pplx_vs_vela_{nm}"] = {"n": len(ks), "kappa": round(kappa([a[k] for k in ks], [b[k] for k in ks]), 4)}
    # --- determinismo
    spec = {}
    for m in deciders_names:
        for s in ("V", "R"):
            A, B = load_decider(m, 1, s), load_decider(m, 2, s)
            if A and B:
                spec[f"{m}:{s}"] = (A, B, "decider_raw")
    for tag in ("qwen", "qwen_t07"):
        for s in ("V", "R"):
            A, B = load_qwen(tag, 1, s), load_qwen(tag, 2, s)
            if A and B:
                spec[f"{tag}:{s}"] = (A, B, "qwen")
    cons["arbitration_B"] = arbitration_block(b_ans, qB, dB, gpt, cons)
    cons["hybrid_B"] = hybrid_block(b_ans, qB, dB, gpt)
    cons["context_axis_validation"] = context_validation(sets["B"], dB)
    cons["determinism"] = determinism_block(spec)
    cons["speed"] = speed_block()
    cons["retrievers"] = retrievers_block(sets["B"], dB, qB)
    # remove campos internos grandes
    for blk in ("agree_R_vs_qwen", "agree_B_vs_gptoss", "agree_B_vs_qwen"):
        for v in cons[blk].values():
            v.pop("_pred", None)
            v.pop("_score", None)
    C.write_json(C.DATA / "consolidation.json", cons)
    print(json.dumps({k: cons[k] for k in cons if k not in ("retrievers",)}, ensure_ascii=False, indent=1)[:6000])
    print(json.dumps(cons["retrievers"], ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
