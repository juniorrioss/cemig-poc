#!/usr/bin/env python3
"""
consolidate_quant.py — tabela-mestra de RETENÇÃO DE QUALIDADE por formato -> data/consolidation.json.

Formatos (todos o MESMO f4-r32-ep2 mesclado, recorte 344, célula op_d50 da F6, contextos
byte a byte idênticos):
  hf_bf16           transformers bf16 (referência-teto) = 3 rodadas op_d50 da F6 (sem regerar)
  gguf_bf16/q8_0/q4_0  llama.cpp v0.9.0 (2 rodadas, seeds 42/43)
  gguf_q4_k_m       ponto extra (k-quant, mesmo protocolo)
  gguf_q4_0_rep1.0  controle: Q4_0 sem repetition penalty (1 rodada)
  litert_<q>_<be>   LiteRT-LM 0.19.0, q ∈ {int4 (mixed int4/int8), int8}, be ∈ {cpu, gpu}

Por formato: média das rodadas (e min–max) de aprovação global/operacional/norma/manual,
cobertura, alucinação, recusa, cita item-ouro; Δ vs hf_bf16 e vs gguf_bf16 com IC 95%
bootstrap pareado por pergunta (média das rodadas por pergunta) e teste do sinal pareado
rodada a rodada; piso de ruído = Δ entre rodadas do mesmo formato.

Uso: <venv>/bin/python bench/lfm3_quant/consolidate_quant.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
F6E2E = REPO / "bench" / "lfm3_f6" / "data" / "e2e"
VOCABS = ("clean", "technical", "operational", "lay")

# tamanhos medidos (bytes, ls -l na Spark/H100) e md5/sha quando conhecidos
SIZES = {
    "hf_bf16": 3_457_635_448, "gguf_bf16": 3_466_625_568, "gguf_q8_0": 1_846_237_728,
    "gguf_q4_0": 1_049_614_880, "gguf_q4_0_rep1.0": 1_049_614_880, "gguf_q4_k_m": 1_099_627_040,
    "litert_int4_cpu": 1_442_191_536, "litert_int4_gpu": 1_442_191_536,
    "litert_int8_cpu": 1_802_113_088, "litert_int8_gpu": 1_802_113_088,
}
ORDER = ["hf_bf16", "gguf_bf16", "gguf_q8_0", "gguf_q4_k_m", "gguf_q4_0", "gguf_q4_0_rep1.0",
         "litert_int8_gpu", "litert_int8_cpu", "litert_int4_gpu", "litert_int4_cpu"]


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_rounds():
    """{fmt: [ {id: scored_row}, ... por rodada ]}"""
    fm = defaultdict(list)
    for t in ("f6-recorte", "f6-recorte-rep2", "f6-recorte-rep3"):
        rows = [r for r in read_jsonl(F6E2E / f"scored_{t}.jsonl") if r["cell"] == "op_d50"]
        fm["hf_bf16"].append({r["id"]: r for r in rows})
    for p in sorted((DATA / "e2e").glob("scored_*.jsonl")):
        tag = p.stem[len("scored_"):]
        fmt = tag.rsplit("-r", 1)[0]
        rows = read_jsonl(p)
        if len(rows) != 344:
            print(f"[skip] {tag}: {len(rows)} linhas")
            continue
        fm[fmt].append({r["id"]: r for r in rows})
    return fm


def agg(rows):
    n = len(rows)
    pct = lambda k: 100 * sum(1 for r in rows if r[k]) / n  # noqa: E731
    gi = [r for r in rows if r["has_gold_items"]]
    return {"n": n, "approved": pct("approved"), "coverage": sum(r["coverage"] for r in rows) / n,
            "hallucination": pct("hallucination"), "refused": pct("refused"),
            "truncated": pct("truncated"),
            "cite_gold_item": 100 * sum(1 for r in gi if r["cite_item"]) / max(1, len(gi)),
            "gen_ntok": sum(r["n_gen_tokens"] for r in rows) / n}


def round_metrics(rd):
    rows = list(rd.values())
    m = {"all": agg(rows)}
    for v in VOCABS:
        m[v] = agg([r for r in rows if r["vocabulary_level"] == v])
    for k in ("norma", "manual"):
        m[k] = agg([r for r in rows if r["gold_kind"] == k])
    m["operational_norma"] = agg([r for r in rows if r["vocabulary_level"] == "operational"
                                  and r["gold_kind"] == "norma"])
    m["operational_manual"] = agg([r for r in rows if r["vocabulary_level"] == "operational"
                                   and r["gold_kind"] == "manual"])
    return m


def mean_rounds(ms, slice_, key):
    vals = [m[slice_][key] for m in ms]
    return {"mean": round(sum(vals) / len(vals), 2), "min": round(min(vals), 2),
            "max": round(max(vals), 2), "rounds": [round(v, 2) for v in vals]}


def sign_p(w, l):
    n = w + l
    if not n:
        return 1.0
    k = min(w, l)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def per_q(rounds, ids):
    return {i: sum(rd[i]["approved"] for rd in rounds) / len(rounds) for i in ids}


def paired(rx, rr, ids, n_boot=4000, seed=7):
    """Δ (x − ref) em p.p. com IC 95% bootstrap por pergunta + sinal pareado rodada a rodada."""
    px, pr = per_q(rx, ids), per_q(rr, ids)
    d = [px[i] - pr[i] for i in ids]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(len(d))] for _ in d]
        boots.append(100 * sum(s) / len(s))
    boots.sort()
    w = l = 0
    for a, b in zip(rx, rr):  # rodada i vs rodada i
        for i in ids:
            w += a[i]["approved"] and not b[i]["approved"]
            l += b[i]["approved"] and not a[i]["approved"]
    return {"delta_pp": round(100 * sum(d) / len(d), 2),
            "ci95": [round(boots[int(0.025 * n_boot)], 2), round(boots[int(0.975 * n_boot)], 2)],
            "win": w, "loss": l, "sign_p": sign_p(w, l)}


def noise(rounds, ids):
    """Piso de ruído intra-formato: |Δ| e w/l entre pares de rodadas do mesmo formato."""
    out = []
    for a in range(len(rounds)):
        for b in range(a + 1, len(rounds)):
            ra, rb = rounds[a], rounds[b]
            w = sum(ra[i]["approved"] and not rb[i]["approved"] for i in ids)
            l = sum(rb[i]["approved"] and not ra[i]["approved"] for i in ids)
            out.append({"pair": [a + 1, b + 1], "delta_pp": round(100 * (w - l) / len(ids), 2),
                        "win": w, "loss": l})
    return out


def speed():
    sp = {}
    q = DATA / "speed"
    for f in ("bf16", "q8_0", "q4_k_m", "q4_0"):
        for dev in ("gpu", "cpu6"):
            p = q / f"bench_{dev}_{f}.json"
            if p.exists() and p.stat().st_size:
                for r in json.loads(p.read_text()):
                    kind = "decode" if r.get("n_gen") else "prefill"
                    if dev == "cpu6" and kind == "prefill":
                        continue  # build CUDA: -ngl 0 ainda manda o prefill em lote p/ a GPU
                    sp.setdefault(f"gguf_{f}", {})[f"{dev}_{kind}_tps"] = round(r["avg_ts"], 2)
    p = q / "speed_hf_bf16.json"
    if p.exists():
        sp.setdefault("hf_bf16", {})["gpu_decode_tps"] = json.loads(p.read_text())["decode_tps_mean"]
    p = q / "litert_bench.json"
    if p.exists():
        for k, v in json.loads(p.read_text()).items():
            sp.setdefault(k, {}).update(v)
    return sp


def gen_speed(fmt_tags):
    """decode t/s registrado DURANTE a geração (com concorrência; só indicativo)."""
    out = {}
    for p in sorted((DATA / "gen").glob("gen_*.jsonl")):
        rows = read_jsonl(p)
        fmt = rows[0]["engine_format"]
        v = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        par = sum(1 for r in rows if r.get("tok_parity"))
        out.setdefault(fmt, {"decode_tps_during_gen": [], "tok_parity": []})
        out[fmt]["decode_tps_during_gen"].append(round(sum(v) / max(1, len(v)), 2))
        out[fmt]["tok_parity"].append(f"{par}/{len(rows)}")
    return out


def main():
    fm = load_rounds()
    ids = sorted(fm["hf_bf16"][0])
    res = {"protocol": {
        "suite": "recorte344", "cell": "op_d50 (F6: thr0.7_f2+man1.75+margin -> top-5 -> compressão 50%)",
        "model": "f4-r32-ep2 mesclado (merged md5 a661ef377131b7fb222884d9f80afbe0)",
        "contexts": "prompts congelados (render HF enable_thinking=False) idênticos entre formatos",
        "sampling": "temp 0.1, top_k 50, top_p 1, rep 1.1 (janela 64 nos motores on-device)",
        "judge": "Qwen3.8-27B-FP8 (régua honesta F6, score_f6.worker)",
        "hf_bf16_note": "3 rodadas op_d50 da F6 (transformers batch 8, sampling HF); não regerado"},
        "formats": {}, "speed_single_stream": speed(), "speed_during_gen": gen_speed(None)}
    ref_hf, ref_gb = fm.get("hf_bf16"), fm.get("gguf_bf16")
    for fmt in ORDER:
        if fmt not in fm:
            continue
        rounds = fm[fmt]
        ms = [round_metrics(rd) for rd in rounds]
        d = {"n_rounds": len(rounds), "size_bytes": SIZES.get(fmt),
             "size_gb": round((SIZES.get(fmt) or 0) / 1e9, 2)}
        for sl in ("all", "operational", "norma", "manual", "clean", "technical", "lay",
                   "operational_norma", "operational_manual"):
            d[sl] = {k: mean_rounds(ms, sl, k) for k in
                     ("approved", "coverage", "hallucination", "refused", "cite_gold_item", "gen_ntok")}
        d["vs_hf_bf16"] = paired(rounds, ref_hf, ids) if fmt != "hf_bf16" else None
        op_ids = [i for i in ids if fm["hf_bf16"][0][i]["vocabulary_level"] == "operational"]
        d["vs_hf_bf16_operational"] = paired(rounds, ref_hf, op_ids) if fmt != "hf_bf16" else None
        d["vs_gguf_bf16"] = paired(rounds, ref_gb, ids) if ref_gb and fmt != "gguf_bf16" else None
        d["noise_intra"] = noise(rounds, ids)
        res["formats"][fmt] = d
    (DATA / "consolidation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{'formato':<18} {'GB':>5} {'aprov':>6} {'(min-max)':>12} {'oper':>6} {'norma':>6} "
          f"{'manual':>6} {'aluc':>5} {'recusa':>6} {'citeIt':>6} {'Δhf':>6} {'IC95':>14} {'w/l':>8} {'p':>6}")
    for fmt, d in res["formats"].items():
        a = d["all"]["approved"]
        v = d["vs_hf_bf16"] or {"delta_pp": 0, "ci95": [0, 0], "win": 0, "loss": 0, "sign_p": 1}
        print(f"{fmt:<18} {d['size_gb']:5.2f} {a['mean']:6.1f} {a['min']:5.1f}-{a['max']:<5.1f} "
              f"{d['operational']['approved']['mean']:6.1f} {d['norma']['approved']['mean']:6.1f} "
              f"{d['manual']['approved']['mean']:6.1f} {d['all']['hallucination']['mean']:5.1f} "
              f"{d['all']['refused']['mean']:6.1f} {d['all']['cite_gold_item']['mean']:6.1f} "
              f"{v['delta_pp']:+6.1f} [{v['ci95'][0]:+5.1f},{v['ci95'][1]:+5.1f}] "
              f"{v['win']:>3}/{v['loss']:<4} {v['sign_p']:6.3f}")


if __name__ == "__main__":
    main()
