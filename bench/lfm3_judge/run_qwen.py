#!/usr/bin/env python3
"""
run_qwen.py — o juiz ATUAL (Qwen3.8-27B-FP8 via vLLM) aplicado aos MESMOS itens, com a
MESMA régua honesta de produção (score_f6.worker): passe determinístico de fatos
(bench/regua/ruler.py) + 1 desempate do 27B (judge_regua.judge_disambiguate, temp 0.0,
thinking-OFF) + heurística de recusa (lfm3_f0.looks_like_refusal).

APROVADO = cobertura >= 0,5 ∧ ¬tautologia ∧ ¬alucinação ∧ ¬recusa (idêntico à F6).
`--rep 2` regrava em arquivo separado para medir determinismo (vLLM com batching
contínuo NÃO garante bit-identidade mesmo em temp 0 — é exatamente o que medimos).
`--temperature` permite a célula "Qwen com temp>0" do brief.

Uso: classifier/.venv/bin/python bench/lfm3_judge/run_qwen.py --sets V R B --rep 1 --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_j as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "bench" / "regua"))
sys.path.insert(0, str(C.REPO / "bench" / "lfm3_f0"))
import judge_regua as JR  # noqa: E402
from common import judge_call, parallel_map  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402
from ruler import deterministic_pass  # noqa: E402


def make_worker(temperature: float):
    def worker(it):
        facts = it.get("facts") or []
        answer = (it["answer"] or "").replace("\u2011", "-").replace("\u2013", "-")
        r = deterministic_pass(it["question"], answer, facts)
        refused = looks_like_refusal(answer)
        judge, raw = None, ""
        t0 = time.perf_counter()
        if answer.strip() and not r.empty:
            facts_block = "\n".join(f"- {f}" for f in r.facts_ambiguous) or "- (nenhum)"
            prompt = JR.JUDGE_PROMPT.format(question=it["question"], gold_text=(it["evidence"] or "")[:3000],
                                            facts=facts_block, answer=answer[:2500])
            raw = judge_call(prompt, max_tokens=768, temperature=temperature)
            judge = JR._parse(raw)
            conf = {f.lower() for f in judge["fatos_presentes"]}
            extra = [f for f in r.facts_ambiguous if f.lower() in conf]
            r.coverage = (len(r.facts_present) + len(extra)) / max(1, len(facts))
            r.hallucination = bool(judge["alucinacao"])
        approved = r.coverage >= 0.5 and not r.tautology and not r.hallucination and not refused
        return {"uid": it["uid"], "coverage": round(r.coverage, 4), "tautology": r.tautology,
                "hallucination": r.hallucination, "refused": refused, "approved": approved,
                "n_ambiguous": len(r.facts_ambiguous), "called_judge": judge is not None,
                "judge_seconds": round(time.perf_counter() - t0, 3), "raw": raw,
                "note": (judge or {}).get("justificativa", "")}
    return worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["V", "R", "B"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="qwen")
    args = ap.parse_args()
    raw_dir = C.DATA / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        items = [it for it in C.read_jsonl(C.DATA / "sets" / f"set_{s}.jsonl") if it.get("answer")]
        ck = C.CACHE / f"{args.tag}_rep{args.rep}_{s}.ckpt.json"
        existing = json.loads(ck.read_text()) if ck.exists() else {}
        t0 = time.time()
        res = parallel_map(make_worker(args.temperature), items, max_workers=args.workers,
                           checkpoint_path=ck, key_fn=lambda x: x["uid"], existing=existing)
        wall = time.time() - t0
        ok = [res[it["uid"]] for it in items if "__error__" not in res.get(it["uid"], {"__error__": 1})]
        C.write_jsonl(raw_dir / f"{args.tag}_rep{args.rep}_{s}.jsonl", ok)
        n_new = len(items) - len(existing)
        print(f"[{args.tag} rep{args.rep} {s}] {len(ok)}/{len(items)} ok em {wall:.0f}s", flush=True)
        if n_new > 0:
            with (raw_dir / "timing.jsonl").open("a") as fh:
                fh.write(json.dumps({"model": args.tag, "rep": args.rep, "set": s, "items": n_new,
                                     "wall_seconds": round(wall, 1), "workers": args.workers,
                                     "temperature": args.temperature, "ts": time.time()}) + "\n")


if __name__ == "__main__":
    main()
