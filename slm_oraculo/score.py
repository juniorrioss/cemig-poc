#!/usr/bin/env python3
"""
score.py — Aplica a RÉGUA HONESTA (bench/regua) a um arquivo de respostas de generate.py.

Régua ÚNICA (ordem do capitão): reusa, sem inventar métrica nova:
  - ruler.deterministic_pass       (cobertura de fatos + antitautologia determinística)
  - judge_regua.judge_disambiguate (desempate único 27B nos ambíguos + ALUCINAÇÃO)
  - bench/regua/data/gabarito_151.jsonl  (fatos obrigatórios por pergunta)

O trecho-ouro usado na checagem de alucinação é o do corpus CORRIGIDO (corpus_fix),
coerente com o oráculo. Métricas: aprovação%, cobertura, tautologia%, ALUCINAÇÃO%
(critério de rejeição, não nota de rodapé), citação% (informativa, fora do gate),
conversão condicional ao chunk-ouro.

Higiene: paralelo (16 workers), checkpoint incremental. Comentários PT-BR; código em inglês.
Uso (classifier/.venv; juiz via REGUA_JUDGE_URL ou default 10.100.0.111:8005):
  ../classifier/.venv/bin/python score.py --resp data/gen_27b_oracle.json --out data/score_27b_oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "bench" / "regua"))
sys.path.insert(0, str(_HERE))

from judge_regua import judge_disambiguate  # noqa: E402
from ruler import RulerResult, deterministic_pass  # noqa: E402
from corpus_fix import CorrectedGoldResolver  # noqa: E402

GABARITO = _ROOT / "bench" / "regua" / "data" / "gabarito_151.jsonl"


def load_gabarito() -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in
            (json.loads(l) for l in GABARITO.read_text(encoding="utf-8").splitlines() if l.strip())}


def load_gold_texts() -> Dict[str, str]:
    """Trecho-ouro do corpus CORRIGIDO por pergunta (p/ checagem de alucinação)."""
    resolver = CorrectedGoldResolver()
    sys.path.insert(0, str(_ROOT / "bench" / "regua"))
    from common import load_holdout  # noqa: E402
    out = {}
    for q in load_holdout():
        gc = resolver.resolve(q["doc"], q["section"], q["question"])
        out[q["id"]] = "\n\n".join(c["text"] for c in gc)
    return out


def score_item(it: Dict[str, Any], gab: Dict[str, Any], gold_text: str,
               threshold: float, use_judge: bool) -> Dict[str, Any]:
    facts = gab["facts"]
    r: RulerResult = deterministic_pass(it["question"], it.get("response", ""), facts, threshold)
    judged = False
    if use_judge and (r.needs_judge or (r.approved and not r.empty)):
        try:
            jr = judge_disambiguate(it["question"], it.get("response", ""),
                                    r.facts_ambiguous, gold_text)
            judged = True
            confirmed = set()
            amb_low = {a.lower(): a for a in r.facts_ambiguous}
            for fp in jr.get("fatos_presentes", []):
                key = fp.strip().lower()
                if key in amb_low:
                    confirmed.add(amb_low[key])
                else:
                    for al, orig in amb_low.items():
                        if al.startswith(key[:12]) or key.startswith(al[:12]):
                            confirmed.add(orig)
            r.facts_present = list(set(r.facts_present) | confirmed)
            r.facts_ambiguous = [a for a in r.facts_ambiguous if a not in confirmed]
            r.hallucination = bool(jr.get("alucinacao", False))
        except Exception:  # noqa: BLE001
            pass
    nn = max(1, len(facts))
    r.coverage = len(r.facts_present) / nn
    r.approved = (r.coverage >= threshold and not r.tautology
                  and not r.hallucination and not r.empty)
    return {
        "item_id": it["item_id"], "coverage": round(r.coverage, 3),
        "approved": r.approved, "tautology": r.tautology,
        "hallucination": r.hallucination, "empty": r.empty,
        "citation": r.citation_present, "retrieval_hit": it.get("retrieval_hit"),
        "n_facts": len(facts), "facts_present": r.facts_present,
        "facts_absent": r.facts_absent, "judged": judged,
    }


def run(args: argparse.Namespace) -> None:
    resp = json.loads(Path(args.resp).read_text(encoding="utf-8"))
    gabarito = load_gabarito()
    gold_texts = load_gold_texts()

    items = [it for it in resp["items"] if it["item_id"] in gabarito]
    captain = [it for it in resp["items"] if it.get("captain_case")]

    def work(it: Dict[str, Any]) -> Dict[str, Any]:
        return score_item(it, gabarito[it["item_id"]],
                          gold_texts.get(it["item_id"], ""),
                          args.threshold, not args.no_judge)

    scored: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it["item_id"] for it in items}
        for fut in as_completed(futs):
            scored.append(fut.result())

    n = len(scored)
    n_appr = sum(1 for s in scored if s["approved"])
    n_taut = sum(1 for s in scored if s["tautology"])
    n_hall = sum(1 for s in scored if s["hallucination"])
    n_cit = sum(1 for s in scored if s["citation"])
    n_empty = sum(1 for s in scored if s["empty"])
    cov = sum(s["coverage"] for s in scored) / max(1, n)

    hit = [s for s in scored if s.get("retrieval_hit") is True]
    miss = [s for s in scored if s.get("retrieval_hit") is False]
    conv_hit = (100 * sum(1 for s in hit if s["approved"]) / len(hit)) if hit else None
    conv_miss = (100 * sum(1 for s in miss if s["approved"]) / len(miss)) if miss else None

    metrics = {
        "resp_file": str(Path(args.resp).name),
        "label": resp["metadata"].get("label"),
        "model": resp["metadata"].get("model"),
        "variant": resp["metadata"].get("variant"),
        "context_source": resp["metadata"].get("context_source"),
        "threshold": args.threshold, "n": n,
        "approval_pct": round(100 * n_appr / max(1, n), 1),
        "mean_coverage": round(cov, 3),
        "tautology_pct": round(100 * n_taut / max(1, n), 1),
        "hallucination_pct": round(100 * n_hall / max(1, n), 1),
        "empty_pct": round(100 * n_empty / max(1, n), 1),
        "citation_pct": round(100 * n_cit / max(1, n), 1),
        "n_approved": n_appr,
        "conv_chunk_hit_pct": (round(conv_hit, 1) if conv_hit is not None else None),
        "conv_chunk_miss_pct": (round(conv_miss, 1) if conv_miss is not None else None),
        "n_hit": len(hit), "n_miss": len(miss),
        "avg_completion_tokens": round(
            sum(it.get("completion_tokens", 0) for it in resp["items"]) / max(1, len(resp["items"])), 1),
    }
    out = {"metrics": metrics, "items": scored,
           "captain_cases": [{"item_id": c["item_id"], "question": c["question"],
                              "response": c.get("response", "")} for c in captain]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
