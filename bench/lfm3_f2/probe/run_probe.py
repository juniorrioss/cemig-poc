"""
run_probe.py — probe de capacidade geral (guarda anti-catastrophic-forgetting).

Reusa as réguas CONGELADAS da POC (ood_general.json 100 itens + 
ood_behavioral.json 30; NUNCA regenerar). Lição histórica: 1.2B esqueceu
(-0.47), 2.6B não; o 1.7B é o teste.

Fase 1 (na Spark): gerar respostas do checkpoint com transformers
    python run_probe.py gen --model LiquidAI/LFM3-1.7B [--adapter ...] --tag A-r8-ep6
Fase 2 (local): julgar com o 27B (qualidade/pt/segurança 1-5)
    classifier/.venv/bin/python run_probe.py judge --tag A-r8-ep6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F2 = HERE.parent

JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso de respostas de um assistente em português.\n"
    "Avalie a RESPOSTA para o PROMPT segundo a RUBRICA. Dê notas 1-5:\n"
    "- 'qualidade': cumpre a tarefa/rubrica de forma correta, útil e completa?\n"
    "- 'pt': português natural e correto?\n"
    "Responda SOMENTE um JSON: {\"qualidade\": n, \"pt\": n, \"justificativa\": \"...\"}.")


def load_items() -> list[dict]:
    g = json.loads((HERE / "ood_general.json").read_text(encoding="utf-8"))
    b = json.loads((HERE / "ood_behavioral.json").read_text(encoding="utf-8"))
    items = [dict(it, suite="general") for it in g["items"]]
    items += [dict(it, suite="behavioral") for it in b["items"]]
    return items


def cmd_gen(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = load_items()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    out_path = HERE / f"probe_gen_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            add_generation_prompt=True, tokenize=False) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=2048, do_sample=True,
                                 temperature=0.1, top_k=50, repetition_penalty=1.1,
                                 pad_token_id=tok.pad_token_id, eos_token_id=124900)
        for it, o in zip(batch, out):
            gen = o[enc["input_ids"].shape[1]:].tolist()
            if 124900 in gen:
                gen = gen[: gen.index(124900)]
            raw = tok.decode(gen, skip_special_tokens=False)
            answer = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": it["id"], "suite": it["suite"],
                                     "answer": answer, "raw": raw},
                                    ensure_ascii=False) + "\n")
        print(f"{start+len(batch)}/{len(todo)}", flush=True)
    print(f"[ok] {out_path}")


def cmd_judge(args) -> None:
    sys.path.insert(0, str(F2 / "synth"))
    from common_synth import gen_json, parallel_map, write_jsonl
    items = {it["id"]: it for it in load_items()}
    gens = [json.loads(l) for l in
            (HERE / f"probe_gen_{args.tag}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ckpt = HERE / f"probe_judge_{args.tag}.ckpt.json"

    def worker(g: dict) -> dict:
        it = items[g["id"]]
        rub = it.get("rubrica", it.get("rubric", ""))
        v = gen_json(f"PROMPT:\n{it['prompt']}\n\nRUBRICA:\n{rub}\n\n"
                     f"RESPOSTA:\n{g['answer'][:4000]}",
                     system=JUDGE_SYSTEM, temperature=0.0, max_tokens=512)
        return {"qualidade": int(v.get("qualidade", 0)), "pt": int(v.get("pt", 0))}

    results = parallel_map(worker, gens, max_workers=16, checkpoint_path=ckpt,
                           key_fn=lambda g: g["id"])
    rows = []
    for g in gens:
        r = results.get(g["id"])
        if r and "__error__" not in r:
            rows.append({**g, **r, "suite": g["suite"]})
    write_jsonl(HERE / f"probe_scored_{args.tag}.jsonl", rows)
    for suite in ("general", "behavioral"):
        sub = [r for r in rows if r["suite"] == suite]
        if sub:
            q = sum(r["qualidade"] for r in sub) / len(sub)
            p = sum(r["pt"] for r in sub) / len(sub)
            print(f"{args.tag} {suite}: qualidade {q:.2f}, pt {p:.2f} (n={len(sub)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--adapter", default="")
    g.add_argument("--tag", required=True)
    g.add_argument("--batch", type=int, default=32)
    j = sub.add_parser("judge")
    j.add_argument("--tag", required=True)
    args = ap.parse_args()
    (cmd_gen if args.cmd == "gen" else cmd_judge)(args)


if __name__ == "__main__":
    main()
