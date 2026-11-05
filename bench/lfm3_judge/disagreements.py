#!/usr/bin/env python3
"""
disagreements.py — amostra as discordâncias decider × Qwen para leitura manual.

Saída: data/disagreements_<set>.jsonl com evidência-ouro, gabarito de fatos, resposta,
veredito/justificativa do Qwen, nota/eixos do decider e (no B) o veredito do gpt-oss.
A classificação "quem tem razão" é feita por leitura do worker e gravada em
data/disagreements_labels.json (chave uid -> {winner: decider|qwen|both_wrong|ambiguous, why}).

Uso: python disagreements.py --model pplx --set R --n 15
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import common_j as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pplx")
    ap.add_argument("--set", default="R")
    ap.add_argument("--n", type=int, default=15, help="por direção")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    items = {it["uid"]: it for it in C.read_jsonl(C.DATA / "sets" / f"set_{args.set}.jsonl") if it.get("answer")}
    q = A.load_qwen("qwen", 1, args.set)
    d = A.load_decider(args.model, 1, args.set)
    import json
    cj = json.loads((C.DATA / "consolidation.json").read_text())
    blk = "agree_R_vs_qwen" if args.set == "R" else "agree_B_vs_qwen"
    thr = cj[blk][f"{args.model}:gate"]["threshold"]
    dec_yes_q_no, dec_no_q_yes = [], []
    for u, it in items.items():
        if u not in q or u not in d:
            continue
        g = A.gate_scores(d[u]).get("gate")
        if g is None:
            continue
        dy, qy = g >= thr, q[u]["approved"]
        row = {"uid": u, "question": it["question"], "gold": it["gold"], "evidence": it["evidence"][:1500],
               "facts": it["facts"], "answer": it["answer"], "context_has_gold": it.get("ctx_has_gold"),
               "qwen": {k: q[u][k] for k in ("approved", "coverage", "hallucination", "refused", "tautology", "note")},
               "decider": {"gate": round(g, 3), "threshold": thr,
                           **{ax: round(d[u][ax], 3) for ax in ("verdict", "hallucination", "refusal", "grounded") if ax in d[u]},
                           "facts_p": [round(x, 2) for x in d[u]["facts"]]}}
        if args.set == "B":
            row["gptoss"] = {"verdict": it["orig"]["gptoss"], "reason": it["orig"]["gptoss_reason"]}
        if dy and not qy:
            dec_yes_q_no.append(row)
        elif qy and not dy:
            dec_no_q_yes.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(dec_yes_q_no)
    rng.shuffle(dec_no_q_yes)
    out = ([dict(r, direction="decider_aprova_qwen_reprova") for r in dec_yes_q_no[: args.n]]
           + [dict(r, direction="qwen_aprova_decider_reprova") for r in dec_no_q_yes[: args.n]])
    C.write_jsonl(C.DATA / f"disagreements_{args.model}_{args.set}.jsonl", out)
    print(f"{args.set}: decider-sim/qwen-não={len(dec_yes_q_no)}  qwen-sim/decider-não={len(dec_no_q_yes)} "
          f"-> amostrados {len(out)}")


if __name__ == "__main__":
    main()
