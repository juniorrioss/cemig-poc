#!/usr/bin/env python3
"""
refusal_metrics.py — PARTE 3 do SFT v3: recusa como CLASSIFICADOR BINÁRIO (pedido do capitão).

O capitão quer "uma noção estilo F1, recall x precision" para a recusa, na MESMA tabela. Este
script lê os arquivos de score da suite de recusa (score_<label>_refusal.json) já produzidos
pela régua v2 (score_v2.py) e computa a matriz de confusão + métricas de classificação.

Definição da classe POSITIVA = "deveria RECUSAR" (contexto NÃO responde: famílias recusa +
distrator, expects_refusal=True). Classe NEGATIVA = "dava para responder" (answerable=True,
família oracle). A família PARCIAL é tratada à parte (é answerable-parcial: deve responder o
que der E declarar o que falta — não é recusa binária), reportada como coluna informativa.

Matriz de confusão (predição = refused; verdade = expects_refusal):
  TP = expects_refusal ∧ refused      (recusou quando devia)
  FN = expects_refusal ∧ ¬refused     (NÃO recusou quando devia — inventa com contexto errado)
  FP = answerable(oracle) ∧ refused   (recusou quando DAVA — recusa-indevida)
  TN = answerable(oracle) ∧ ¬refused  (respondeu quando dava)

  recusa-correta (recall)   = TP / (TP+FN)
  recusa-indevida (FP rate) = FP / (FP+TN)
  precision                 = TP / (TP+FP)
  F1                        = 2·P·R / (P+R)

Uso:
  classifier/.venv/bin/python refusal_metrics.py \
     --scores sft_v2/data/score_v2_r64_q4_refusal.json:v2_r64_q4 \
              sft_v3/data/score_v3_r64_q4_refusal.json:v3_r64_q4 \
     --out sft_v3/data/refusal_confusion.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def confusion(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Matriz de confusão binária + métricas de classificação da recusa."""
    tp = fn = fp = tn = 0
    # parcial: informativa (answerable-parcial), fora da binária.
    parc_n = parc_refused = 0
    for it in items:
        fam = it.get("family")
        refused = bool(it.get("refused"))
        expects = it.get("expects_refusal")
        answerable = it.get("answerable")
        if fam == "parcial":
            parc_n += 1
            parc_refused += int(refused)
            continue
        if expects is True:  # deveria recusar (recusa/distrator)
            if refused:
                tp += 1
            else:
                fn += 1
        elif answerable is True:  # dava para responder (oracle)
            if refused:
                fp += 1
            else:
                tn += 1
    recall = tp / max(1, tp + fn)          # recusa-correta
    fp_rate = fp / max(1, fp + tn)         # recusa-indevida
    precision = tp / max(1, tp + fp)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "n_should_refuse": tp + fn, "n_answerable": fp + tn,
        "recusa_correta_recall_pct": round(100 * recall, 1),
        "recusa_indevida_fp_pct": round(100 * fp_rate, 1),
        "precision_pct": round(100 * precision, 1),
        "f1_pct": round(100 * f1, 1),
        "accuracy_pct": round(100 * accuracy, 1),
        "parcial_n": parc_n, "parcial_refused_pct": round(100 * parc_refused / max(1, parc_n), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="lista de arquivo:label (arquivo score_*_refusal.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: Dict[str, Any] = {}
    for spec in args.scores:
        path_str, _, label = spec.partition(":")
        path = Path(path_str)
        if not path.exists():
            print(f"[skip] {path} inexistente", flush=True)
            continue
        label = label or path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        rows[label] = confusion(items)
        c = rows[label]
        print(f"{label:16s} | recall(correta) {c['recusa_correta_recall_pct']:5.1f}% | "
              f"FP(indevida) {c['recusa_indevida_fp_pct']:5.1f}% | prec {c['precision_pct']:5.1f}% | "
              f"F1 {c['f1_pct']:5.1f}% | TP{c['TP']} FN{c['FN']} FP{c['FP']} TN{c['TN']}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
