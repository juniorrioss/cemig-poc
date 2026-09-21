#!/usr/bin/env python3
"""
score_v2.py — Régua honesta v2 + métrica de RECUSA (Partes 2 e 3).

Régua ÚNICA (ordem do capitão): reusa bench/regua (ruler.deterministic_pass +
judge_regua.judge_disambiguate). Mede, por arquivo de respostas:
  - APROVAÇÃO / cobertura / ALUCINAÇÃO (critério de rejeição) — famílias com fatos;
  - CITAÇÃO (informativa, fora do gate);
  - RECUSA: recusa-correta% (recusou nos UNANSWERABLE) e recusa-indevida% (recusou nos
    ANSWERABLE — dano igualmente grave). É a NOVA MÉTRICA obrigatória da Parte 3.

Duas fontes de gabarito (auto-detectadas):
  (A) itens com "facts"/"gold_text" embutidos (val_pack/refusal_test_pack) -> usa direto;
  (B) itens das 151 (oracle_pack_151/retrieved_pack_151) -> carrega gabarito_151 + o
      trecho-ouro do corpus CORRIGIDO (corpus_fix), igual ao v1.

Para as famílias SEM oráculo (recusa/distrator/parcial) do val/refusal pack, a "aprovação"
usa o critério de recusa correta (judge_nonoracle + approve_nonoracle), NÃO cobertura de
fatos — a métrica de qualidade dessas famílias é recusar/delimitar certo.

Higiene: paralelo, checkpoint-free (idempotente), não-interativo. Comentários PT-BR.

Uso (juiz via REGUA_JUDGE_URL ou default):
  ../classifier/.venv/bin/python score_v2.py --resp data/gen_x.json --out data/score_x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
import common_v2 as C  # noqa: E402
sys.path.insert(0, str(_ROOT))  # raiz do repo p/ o pacote `corpus`

GABARITO = _ROOT / "bench" / "regua" / "data" / "gabarito_151.jsonl"


def load_gabarito_151() -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in
            (json.loads(l) for l in GABARITO.read_text(encoding="utf-8").splitlines() if l.strip())}


def load_gold_texts_151() -> Dict[str, str]:
    """Trecho-ouro do corpus CORRIGIDO por pergunta das 151 (checagem de alucinação)."""
    from corpus.corpus_fix import CorrectedGoldResolver  # movido p/ corpus/ na limpeza
    sys.path.insert(0, str(_ROOT / "bench" / "regua"))
    from common import load_holdout
    resolver = CorrectedGoldResolver()
    out = {}
    for q in load_holdout():
        gc = resolver.resolve(q["doc"], q["section"], q["question"])
        out[q["id"]] = "\n\n".join(c["text"] for c in gc)
    return out


def score_factbased(question: str, answer: str, facts: List[str], gold_text: str,
                    threshold: float, use_judge: bool) -> Dict[str, Any]:
    """Régua de cobertura de fatos (oráculo/parcial/151) — igual ao v1."""
    r = C.deterministic_pass(question, answer, facts, threshold)
    judged = False
    if use_judge and (r.needs_judge or (r.approved and not r.empty)):
        try:
            jr = C.judge_disambiguate(question, answer, r.facts_ambiguous, gold_text)
            judged = True
            amb_low = {a.lower(): a for a in r.facts_ambiguous}
            confirmed = set()
            for fp in jr.get("fatos_presentes", []):
                key = fp.strip().lower()
                for al, orig in amb_low.items():
                    if key == al or al.startswith(key[:12]) or key.startswith(al[:12]):
                        confirmed.add(orig)
            r.facts_present = list(set(r.facts_present) | confirmed)
            r.hallucination = bool(jr.get("alucinacao", False))
        except Exception:
            pass
    nn = max(1, len(facts))
    r.coverage = len(r.facts_present) / nn
    r.approved = (r.coverage >= threshold and not r.tautology
                  and not r.hallucination and not r.empty)
    return {"coverage": round(r.coverage, 3), "approved": r.approved,
            "tautology": r.tautology, "hallucination": r.hallucination,
            "empty": r.empty, "citation": r.citation_present, "judged": judged}


def score_item(it: Dict[str, Any], gabarito: Dict[str, Dict[str, Any]],
               gold_151: Dict[str, str], threshold: float, use_judge: bool) -> Dict[str, Any]:
    iid = it["item_id"]
    question = it.get("question", "")
    answer = it.get("response", "")
    family = it.get("family")
    answerable = it.get("answerable")
    expects_refusal = it.get("expects_refusal")

    # fatos + trecho-ouro: embutidos (val/refusal) ou do gabarito_151 (151).
    if "facts" in it and it.get("facts") is not None:
        facts = it["facts"]
        gold_text = it.get("gold_text", "")
    elif iid in gabarito:
        facts = gabarito[iid]["facts"]
        gold_text = gold_151.get(iid, "")
        answerable = True  # oráculo/retrieved das 151 são answerable por construção
        family = family or "oracle"
    else:
        facts, gold_text = [], ""

    # detecção de recusa (barata + confirmação do juiz nos ambíguos).
    refused_det = C.looks_like_refusal(answer)
    refused = refused_det
    refusal_judged = False
    # confirma com o juiz quando o barato diz "respondeu" mas era p/ recusar, ou vice-versa
    if use_judge and expects_refusal is not None and (refused_det != bool(expects_refusal)):
        try:
            refused = C.judge_is_refusal(question, answer)
            refusal_judged = True
        except Exception:
            refused = refused_det

    out: Dict[str, Any] = {
        "item_id": iid, "family": family, "answerable": answerable,
        "expects_refusal": expects_refusal, "refused": refused,
        "refusal_judged": refusal_judged, "n_facts": len(facts),
    }

    # aprovação: famílias sem oráculo (recusa/distrator) usam critério de recusa correta;
    # oráculo/parcial/151 usam cobertura de fatos.
    if family in ("recusa", "distrator") and expects_refusal:
        # aprovado = recusou corretamente (juiz de recusa útil, se disponível).
        approved = refused
        # tenta o juiz detalhado p/ registrar por que (útil no relatório).
        verdict = None
        if use_judge:
            try:
                verdict = C.judge_nonoracle(family, question, _given_ctx(it), answer, gold_text)
                if verdict:
                    approved = C.approve_nonoracle(family, verdict)
            except Exception:
                pass
        out.update({"approved": approved, "coverage": None,
                    "hallucination": (verdict["inventou_dado"] if verdict else (not refused)),
                    "citation": bool(C.deterministic_pass(question, answer, [], threshold).citation_present),
                    "verdict": verdict})
    else:
        fb = score_factbased(question, answer, facts, gold_text, threshold, use_judge)
        out.update(fb)
        # para a família parcial, registra também se declarou o que faltou (recusou parcial).
        if family == "parcial":
            out["declared_gap"] = refused
    return out


def _given_ctx(it: Dict[str, Any]) -> str:
    """Recupera o contexto DADO ao modelo, se o generate_v2 propagou; senão vazio."""
    return it.get("context", "") or ""


def run(args: argparse.Namespace) -> None:
    resp = json.loads(Path(args.resp).read_text(encoding="utf-8"))
    items = resp["items"]
    is_151 = any(it["item_id"] in load_gabarito_151() for it in items[:5]) \
        and not any("facts" in it and it.get("family") in ("recusa", "distrator") for it in items[:5])
    gabarito = load_gabarito_151() if is_151 else {}
    gold_151 = load_gold_texts_151() if is_151 else {}

    def work(it: Dict[str, Any]) -> Dict[str, Any]:
        return score_item(it, gabarito, gold_151, args.threshold, not args.no_judge)

    scored: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it["item_id"] for it in items
                if not it.get("__error__")}
        for fut in as_completed(futs):
            scored.append(fut.result())

    metrics = summarize(scored, resp["metadata"], Path(args.resp).name, args.threshold, items)
    out = {"metrics": metrics, "items": scored}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def summarize(scored: List[Dict[str, Any]], resp_meta: Dict[str, Any], fname: str,
              threshold: float, raw_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scored)
    # métricas globais de aprovação (todas as famílias com veredito booleano).
    appr = [s for s in scored if s.get("approved") is not None]
    n_appr = sum(1 for s in appr if s["approved"])
    # cobertura/alucinação sobre as famílias fact-based (coverage != None).
    fb = [s for s in scored if s.get("coverage") is not None]
    cov = sum(s["coverage"] for s in fb) / max(1, len(fb)) if fb else None
    hall = [s for s in scored if s.get("hallucination") is not None]
    n_hall = sum(1 for s in hall if s["hallucination"])

    # métrica de RECUSA (Parte 3).
    unans = [s for s in scored if s.get("expects_refusal") is True]
    ans = [s for s in scored if s.get("answerable") is True and s.get("family") == "oracle"]
    recusa_correta = (100 * sum(1 for s in unans if s["refused"]) / len(unans)) if unans else None
    recusa_indevida = (100 * sum(1 for s in ans if s["refused"]) / len(ans)) if ans else None

    # por família.
    by_fam: Dict[str, Dict[str, Any]] = {}
    fams = sorted({s.get("family") for s in scored if s.get("family")})
    for fam in fams:
        fs = [s for s in scored if s.get("family") == fam]
        fa = [s for s in fs if s.get("approved") is not None]
        n_fa = sum(1 for s in fa if s["approved"])
        fcov = [s for s in fs if s.get("coverage") is not None]
        by_fam[fam] = {
            "n": len(fs),
            "approval_pct": round(100 * n_fa / max(1, len(fa)), 1) if fa else None,
            "mean_coverage": (round(sum(s["coverage"] for s in fcov) / len(fcov), 3)
                              if fcov else None),
            "hallucination_pct": (round(100 * sum(1 for s in fs if s.get("hallucination"))
                                        / max(1, len(fs)), 1)),
            "refused_pct": round(100 * sum(1 for s in fs if s.get("refused")) / max(1, len(fs)), 1),
        }

    avg_tok = (sum(it.get("completion_tokens", 0) for it in raw_items)
               / max(1, len(raw_items)))
    return {
        "resp_file": fname, "label": resp_meta.get("label"),
        "model": resp_meta.get("model"), "system": resp_meta.get("system"),
        "pack": resp_meta.get("pack"), "threshold": threshold, "n": n,
        "approval_pct": round(100 * n_appr / max(1, len(appr)), 1) if appr else None,
        "mean_coverage": (round(cov, 3) if cov is not None else None),
        "hallucination_pct": round(100 * n_hall / max(1, len(hall)), 1) if hall else None,
        "n_approved": n_appr,
        "recusa_correta_pct": (round(recusa_correta, 1) if recusa_correta is not None else None),
        "recusa_indevida_pct": (round(recusa_indevida, 1) if recusa_indevida is not None else None),
        "n_unanswerable": len(unans), "n_answerable_oracle": len(ans),
        "by_family": by_fam,
        "avg_completion_tokens": round(avg_tok, 1),
    }


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
