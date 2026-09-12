"""
rejudge.py — PASSO 3 da régua honesta: rejulgamento retroativo de TODOS os candidatos.

Varre as respostas já salvas (loaders.discover_candidates), aplica a régua honesta
(ruler.deterministic_pass + desempate único do juiz 27B nos ambíguos/alucinação) e
produz uma tabela única comparável:
  aprovacao%, cobertura média, taxa de TAUTOLOGIA, citação (informativa), hit médio.

Higiene: paralelo (ThreadPoolExecutor), checkpoint incremental por candidato,
não gera resposta nova (só lê disco), procedência registrada.

Uso:
    python3 bench/regua/rejudge.py [--workers 16] [--threshold 0.5] [--no-judge]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from common import GoldResolver, JUDGE_MODEL, JUDGE_URL
from judge_regua import judge_disambiguate
from loaders import discover_candidates
from ruler import RulerResult, deterministic_pass

DATA_DIR = Path(__file__).resolve().parent / "data"
GABARITO = DATA_DIR / "gabarito_151.jsonl"
OUT_PER_ITEM = DATA_DIR / "rejudge_items.jsonl"
OUT_TABLE = DATA_DIR / "rejudge_table.json"
CKPT = DATA_DIR / "rejudge.ckpt.json"


def load_gabarito() -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in
            (json.loads(l) for l in GABARITO.read_text(encoding="utf-8").splitlines() if l.strip())}


def load_gold_texts() -> Dict[str, str]:
    """Texto do trecho-ouro por item_id (para o juiz checar alucinação)."""
    from common import load_holdout
    resolver = GoldResolver()
    out = {}
    for q in load_holdout():
        gc = resolver.resolve(q["doc"], q["section"])
        out[q["id"]] = "\n\n".join(c["text"] for c in gc)
    return out


def evaluate_candidate(cand: Dict[str, Any], gabarito: Dict[str, Any],
                       gold_texts: Dict[str, str], threshold: float,
                       use_judge: bool) -> Dict[str, Any]:
    """Aplica a régua a um candidato; retorna métricas agregadas + itens."""
    items_out = []
    n = 0
    n_approved = 0
    n_taut = 0
    n_halluc = 0
    n_citation = 0
    cov_sum = 0.0
    hit_sum = 0
    hit_count = 0

    for item_id, it in cand["items"].items():
        gab = gabarito.get(item_id)
        if not gab:
            continue
        facts = gab["facts"]
        r: RulerResult = deterministic_pass(it["question"], it["response"], facts, threshold)

        # Desempate do juiz: só quando pode passar (ambíguos) ou risco de alucinação
        # em resposta já aprovada deterministicamente.
        judged = None
        if use_judge and (r.needs_judge or (r.approved and not r.empty)):
            try:
                jr = judge_disambiguate(
                    it["question"], it["response"],
                    r.facts_ambiguous, gold_texts.get(item_id, ""))
                judged = jr
                # incorpora fatos ambíguos confirmados presentes
                confirmed = set()
                amb_low = {a.lower(): a for a in r.facts_ambiguous}
                for fp in jr.get("fatos_presentes", []):
                    key = fp.strip().lower()
                    if key in amb_low:
                        confirmed.add(amb_low[key])
                    else:
                        # casamento aproximado por prefixo
                        for al, orig in amb_low.items():
                            if al.startswith(key[:12]) or key.startswith(al[:12]):
                                confirmed.add(orig)
                r.facts_present = list(set(r.facts_present) | confirmed)
                r.facts_ambiguous = [a for a in r.facts_ambiguous if a not in confirmed]
                r.hallucination = bool(jr.get("alucinacao", False))
            except Exception as e:  # noqa: BLE001
                judged = {"__error__": str(e)}

        nn = max(1, len(facts))
        r.coverage = len(r.facts_present) / nn
        # Aprovação final: cobertura>=limiar E não-tautologia E sem alucinação E não-vazia.
        r.approved = (r.coverage >= threshold and not r.tautology
                      and not r.hallucination and not r.empty)

        n += 1
        cov_sum += r.coverage
        if r.approved:
            n_approved += 1
        if r.tautology:
            n_taut += 1
        if r.hallucination:
            n_halluc += 1
        if r.citation_present:
            n_citation += 1
        if it.get("retrieval_hit") is not None:
            hit_count += 1
            hit_sum += 1 if it["retrieval_hit"] else 0

        items_out.append({
            "candidate": cand["candidate"], "item_id": item_id,
            "coverage": round(r.coverage, 3), "approved": r.approved,
            "tautology": r.tautology, "hallucination": r.hallucination,
            "empty": r.empty, "citation": r.citation_present,
            "facts_present": r.facts_present, "facts_absent": r.facts_absent,
            "n_facts": len(facts), "q_overlap": r.detail.get("q_overlap"),
            "judged": bool(judged),
        })

    metrics = {
        "candidate": cand["candidate"], "source": cand["source"],
        "family": cand["family"], "n": n,
        "approval_pct": round(100 * n_approved / max(1, n), 1),
        "mean_coverage": round(cov_sum / max(1, n), 3),
        "tautology_pct": round(100 * n_taut / max(1, n), 1),
        "hallucination_pct": round(100 * n_halluc / max(1, n), 1),
        "citation_pct": round(100 * n_citation / max(1, n), 1),  # informativa
        "retrieval_hit_pct": (round(100 * hit_sum / hit_count, 1)
                              if hit_count else None),
        "n_approved": n_approved,
    }
    return {"metrics": metrics, "items": items_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-judge", action="store_true", help="só passe determinístico")
    ap.add_argument("--limit-cands", type=int, default=0)
    args = ap.parse_args()

    gabarito = load_gabarito()
    gold_texts = load_gold_texts()
    cands = discover_candidates()
    if args.limit_cands:
        cands = cands[:args.limit_cands]
    print(f"[rejudge] {len(cands)} candidatos, threshold={args.threshold}, "
          f"judge={'OFF' if args.no_judge else 'ON'}")

    # Checkpoint por candidato (idempotente).
    done: Dict[str, Any] = {}
    if CKPT.exists():
        done = json.loads(CKPT.read_text(encoding="utf-8"))
        print(f"[ckpt] {len(done)} candidatos já avaliados")

    pending = [c for c in cands if c["candidate"] not in done]

    def work(c: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_candidate(c, gabarito, gold_texts, args.threshold,
                                  use_judge=not args.no_judge)

    # Paraleliza no nível de CANDIDATO; cada candidato dispara suas chamadas ao juiz
    # em série (mas há muitos candidatos, então o pool satura o juiz).
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c["candidate"] for c in pending}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                res = fut.result()
                done[name] = res
            except Exception as e:  # noqa: BLE001
                done[name] = {"__error__": str(e)}
                print(f"  ERRO {name}: {e}")
            completed += 1
            m = done[name].get("metrics", {})
            print(f"  [{completed}/{len(pending)}] {name:<46} "
                  f"aprov={m.get('approval_pct')}% cov={m.get('mean_coverage')} "
                  f"taut={m.get('tautology_pct')}%")
            CKPT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")

    # Grava tabela e itens.
    table = []
    all_items = []
    for name, res in done.items():
        if "metrics" in res:
            table.append(res["metrics"])
            all_items.extend(res.get("items", []))
    table.sort(key=lambda m: (-m["approval_pct"], -m["mean_coverage"]))

    OUT_TABLE.write_text(json.dumps({
        "threshold": args.threshold,
        "judge_model": JUDGE_MODEL, "judge_url": JUDGE_URL,
        "provenance": "rejulgamento retroativo de respostas salvas (nenhuma gerada)",
        "n_candidates": len(table),
        "table": table,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    with OUT_PER_ITEM.open("w", encoding="utf-8") as fh:
        for it in all_items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n[ok] tabela -> {OUT_TABLE}  ({len(table)} candidatos)")
    print(f"[ok] itens  -> {OUT_PER_ITEM}")
    print("\nTOP 15 por aprovação:")
    for m in table[:15]:
        print(f"  {m['candidate']:<46} aprov={m['approval_pct']:>5}% "
              f"cov={m['mean_coverage']:.3f} taut={m['tautology_pct']:>5}% "
              f"cit={m['citation_pct']:>5}% n={m['n']}")


if __name__ == "__main__":
    main()
