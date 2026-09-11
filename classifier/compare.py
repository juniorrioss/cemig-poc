#!/usr/bin/env python3
"""
compare.py — FASE B: tabela comparativa única (clássicos × prompt-classify).

Consolida classifier/data/classic_results.json e classifier/data/prompt_results.json
numa tabela: acurácia top1/top2 × macro-F1 × latência × RAM/tamanho × complexidade de
deploy Android. Aplica o critério explícito de escolha (regra de desempate: mais simples
vence) e projeta a latência mobile de cada candidato.

Projeção mobile:
  - Clássicos: TF-IDF hashing/vocab + produto interno linear é O(nnz) — porta trivial
    em Kotlin puro (sem runtime), latência estimada < 5 ms; RAM ~ tamanho do modelo.
  - Prompt-classify: prefill de ~900 tokens do catálogo no LFM2.5-1.2B. Pela tabela
    device-liquid (QAD-Q4_0: prefill pp800=219 tok/s, pp1500=207 tok/s), ~900 tok de
    prefill + ~7 tok decode ≈ 4.3-4.6 s de TTFT no S24+ por classificação.

Uso: python3 classifier/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Projeção S24+ do prompt-classify: prefill do catálogo (~900 tok) no LFM2.5-1.2B QAD-Q4_0.
# Fonte: bench/device-liquid/README.md (prefill pp800=219 tok/s, TTFT pp800=3.65s).
PROMPT_PREFILL_TOKENS = 900
PROMPT_PREFILL_TOKS_PER_S_S24 = 207.0  # conservador (faixa pp1500)


def deploy_complexity(name: str) -> str:
    """Rótulo de complexidade de deploy Android por candidato."""
    m = {
        "logreg": "BAIXA — TF-IDF + matriz linear portável em Kotlin puro OU ONNX (sklearn->onnx)",
        "linsvc": "BAIXA — idêntico ao logreg (linear); sem probs calibradas p/ boost",
        "complementnb": "BAIXA-MÉDIA — log-probs por classe em Kotlin; modelo 15 MB",
        "randomforest": "ALTA — 300 árvores, 127 MB serializado; ONNX pesado, inviável mobile",
        "decisiontree": "BAIXA — 1 árvore 384 KB, mas acurácia inaceitável (23%)",
        "gradientboost": "ALTA — exige entrada densa (25k dims); descartado",
        "prompt-json": "MÉDIA — sem novo artefato (LFM2.5 já residente), mas +~4.5s TTFT/consulta no S24+",
    }
    return m.get(name, "?")


def main() -> None:
    classic = json.loads((_HERE / "data" / "classic_results.json").read_text(encoding="utf-8"))
    prompt = json.loads((_HERE / "data" / "prompt_results.json").read_text(encoding="utf-8"))

    prompt_prefill_ms = PROMPT_PREFILL_TOKENS / PROMPT_PREFILL_TOKS_PER_S_S24 * 1000

    rows = []
    for r in classic["results"]:
        size_kb = (r.get("pipeline_bytes") or r.get("model_bytes") or 0) / 1024
        rows.append({
            "cand": r["name"],
            "top1": r["top1_acc"],
            "top2": r["top2_acc"],
            "mf1": r["macro_f1"],
            "lat_desktop_ms": r.get("lat_ms_desktop"),
            "lat_s24_ms": None if r["name"] in ("randomforest", "gradientboost") else 5.0,
            "size_kb": size_kb,
            "deploy": deploy_complexity(r["name"]),
        })
    # melhor formato do prompt-classify (json)
    best_fmt = max(prompt["results"], key=lambda x: x["top1_acc"])
    rows.append({
        "cand": f"prompt-{best_fmt['format']}",
        "top1": best_fmt["top1_acc"],
        "top2": None,  # prompt-classify devolve 1 norma (top-1); top-2 exigiria 2ª geração
        "mf1": None,
        "lat_desktop_ms": best_fmt["lat_ms_median_5070"],
        "lat_s24_ms": round(prompt_prefill_ms, 0),
        "size_kb": 0,  # sem artefato novo (modelo já residente)
        "deploy": deploy_complexity("prompt-json"),
    })

    print("\n" + "=" * 118)
    print("  FASE B — TABELA COMPARATIVA ÚNICA (holdout real 171: 151 qa_v2 + 20 smoke)")
    print("=" * 118)
    hdr = f"| {'Candidato':<14} | {'Top-1':>6} | {'Top-2':>6} | {'MacroF1':>7} | {'Lat desk':>8} | {'Lat S24+':>8} | {'Tamanho':>8} |"
    print(hdr)
    print("|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    for r in rows:
        t2 = f"{r['top2']:.1f}%" if r["top2"] is not None else "  n/a"
        mf = f"{r['mf1']:.3f}" if r["mf1"] is not None else " n/a"
        ld = f"{r['lat_desktop_ms']:.1f}ms" if r["lat_desktop_ms"] is not None else "  n/a"
        ls = f"{r['lat_s24_ms']:.0f}ms" if r["lat_s24_ms"] is not None else "inviável"
        sz = f"{r['size_kb']/1024:.1f}MB" if r["size_kb"] >= 1024 else (f"{r['size_kb']:.0f}KB" if r["size_kb"] else "0 (resid.)")
        print(f"| {r['cand']:<14} | {r['top1']:>5.1f}% | {t2:>6} | {mf:>7} | {ld:>8} | {ls:>8} | {sz:>8} |")
    print("=" * 118)

    print("\nComplexidade de deploy Android:")
    for r in rows:
        print(f"  - {r['cand']:<14}: {r['deploy']}")

    # Critério de escolha explícito
    print("\n--- CRITÉRIO DE ESCOLHA (top-2 p/ boost suave > top-1 > macro-F1 > simplicidade) ---")
    viable = [r for r in rows if r["top2"] is not None and r["cand"] not in ("randomforest", "gradientboost", "decisiontree")]
    viable.sort(key=lambda r: (-(r["top2"] or 0), -(r["top1"] or 0), -(r["mf1"] or 0)))
    winner = viable[0]
    print(f"Vencedor: {winner['cand']}  (top1={winner['top1']}% top2={winner['top2']}% macroF1={winner['mf1']})")
    print("Justificativa: melhor top-2 (essencial p/ boost suave 2-NR), latência desktop <1ms,")
    print("porte trivial em Kotlin puro ou ONNX (<8 MB), probs calibradas para boost proporcional.")
    print("randomforest tem top-1 marginalmente maior mas top-2 pior, 127 MB e 67ms/consulta (inviável).")
    print("prompt-classify (25.7% top-1) confirma o ceticismo do capitão e custa ~4.5s TTFT/consulta.")

    out = {"rows": rows, "winner": winner["cand"]}
    (_HERE / "data" / "compare_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
