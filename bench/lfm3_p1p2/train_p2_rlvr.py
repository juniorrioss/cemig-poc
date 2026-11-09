#!/usr/bin/env python3
"""
train_p2_rlvr.py — P2: mini-RLVR de groundedness sobre o f4-r32-ep2 (TRL GRPO).

- Política de partida: f4-r32-ep2 MESCLADO (`merge_policy.py`, md5 publicado);
  LoRA NOVO r32 alpha 64 MLP w1/w2/w3 (asserts de sempre: 72 lora_A em
  feed_forward, 0 em self_attn/gate_proj/conv).
- Prompts: `p2_prompts.jsonl` (banda produtiva + difícil do P1 sobre o POOL RLVR:
  contexto REAL do retriever op, top-5 -> 50%, formato do APP com turno tool;
  ouro presente E ausente). O prompt vai como TEXTO já renderizado pelo
  `apply_chat_template(..., enable_thinking=False)` — byte a byte o que o
  app/avaliação entrega (inclui o `<think></think>` vazio do thinking-OFF).
- Reward: `verifier.verify` (binário, regra, zero juiz) — o CONTRATO publicado
  com testes ANTES do treino. Variante `--concision`: r -= λ·max(0, tok-T)/T só
  quando r=1 (adianta o P3; opcional, secundário).
- Filtro dinâmico DAPO-style: (1) OFFLINE — os prompts já vêm filtrados pelo
  P1 (banda 0<acc<0,8 + difíceis que o 27B resolve), que é onde a política
  tem variância de reward; (2) ONLINE — grupos todo-0/todo-1 têm vantagem
  exatamente zero (`scale_rewards="group"`) e a loss "dapo" normaliza pelos
  tokens ativos, então eles NÃO contribuem gradiente (descarte efetivo; o
  TRL 1.17 não reamostra para repor o lote — a fração é logada como
  `frac_reward_zero_std`). Clip-higher (ε_high 0,28) e overlong filtering
  (`mask_truncated_completions`) também do DAPO.

Uso (container, GPU 3 compartilhada com o Qwen):
  python train_p2_rlvr.py --run lfm3-p2-rlvr --max-steps 400
  python train_p2_rlvr.py --preflight
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/cemig/p1p2/f4-r32-ep2-merged")
    ap.add_argument("--prompts", default="/workspace/cemig/p1p2/p2_prompts.jsonl")
    ap.add_argument("--run", default="lfm3-p2-rlvr")
    ap.add_argument("--out", default="/workspace/cemig/checkpoints")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--x", type=float, default=None, help="limiar de cobertura (default do verifier)")
    ap.add_argument("--concision", type=float, default=0.0, help="λ da penalidade de comprimento (0=off)")
    ap.add_argument("--concision-target", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refuse-oversample", type=int, default=2)
    ap.add_argument("--max-prompt-tokens", type=int, default=100000)
    ap.add_argument("--contract", choices=["v2", "v4"], default="v2",
                    help="v2 = binário conjuntivo; v4 = escalonado (gate de números + cobertura contínua + concisão)")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_PROJECT", "lfm3-1.7b-cemig")
    os.environ.setdefault("WANDB_DIR", "/workspace/cemig/wandb")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    import verifier as V

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    # recusa é minoria no pool filtrado (~11%): sobreamostra p/ ~20% (brief: recusa
    # correta com ouro ausente deve ser recompensada com frequência suficiente)
    rows = rows + [r for r in rows if r["expect"] == "refuse"] * (args.refuse_oversample - 1)
    random.Random(args.seed).shuffle(rows)
    recs = []
    n_long = 0
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        # memória: 8x8 gerações com prompt de 6k tok (k=8 estático) estoura a folga da
        # GPU compartilhada com o Qwen; os prompts do app têm ~1,1k tok
        if len(tok(text, add_special_tokens=False)["input_ids"]) > args.max_prompt_tokens:
            n_long += 1
            continue
        # o TRL tokeniza com add_special_tokens=False -> o BOS do template fica
        recs.append({"prompt": text, "uid": r["uid"], "expect": r["expect"],
                     "facts": json.dumps(r.get("facts") or [], ensure_ascii=False),
                     "gold_items": json.dumps(r.get("gold_items") or []),
                     "context": r.get("context") or "", "question": r["question"]})
    ds = Dataset.from_list(recs)
    n_ans = sum(1 for r in recs if r["expect"] == "answer")
    print(f"[data] descartados por comprimento > {args.max_prompt_tokens}: {n_long}")
    print(f"[data] {len(recs)} prompts (answer={n_ans}, refuse={len(recs) - n_ans})", flush=True)

    out_dir = Path(args.out) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = (out_dir / "reward_dump.jsonl").open("a", encoding="utf-8")
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    def reward_rlvr(prompts, completions, completion_ids, expect, facts, gold_items,
                    context, question, uid, **kw):
        out = []
        for i, c in enumerate(completions):
            ids = completion_ids[i]
            stopped = len(ids) > 0 and (ids[-1] == eos_id or len(ids) < args.max_completion)
            item = {"expect": expect[i], "facts": json.loads(facts[i]),
                    "gold_items": json.loads(gold_items[i]), "context": context[i],
                    "question": question[i]}
            if args.contract == "v4":
                v = V.verify_v4(c, item, stopped=stopped, n_tok=len(ids))
            else:
                v = V.verify(c, item, stopped=stopped, x=args.x)
            r = float(v.reward)
            if args.concision and r > 0:
                over = max(0, len(ids) - args.concision_target) / args.concision_target
                r -= min(0.5, args.concision * over)
            out.append(r)
            if i % args.num_generations == 0:
                dump.write(json.dumps({"uid": uid[i], "expect": expect[i], "r": r,
                                       "reason": v.reason, "n_tok": len(ids),
                                       "text": c[-500:]}, ensure_ascii=False) + "\n")
        dump.flush()
        return out

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["w1", "w2", "w3"])
    gen_bs = args.prompts_per_step * args.num_generations
    cfg = GRPOConfig(
        output_dir=str(out_dir), run_name=args.run, report_to=["wandb"],
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=gen_bs // args.micro_batch,
        generation_batch_size=gen_bs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=1.0, top_p=1.0, top_k=0,
        max_steps=args.max_steps, save_steps=args.save_steps, save_total_limit=20,
        logging_steps=1, bf16=True, seed=args.seed,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.beta, loss_type="dapo", scale_rewards="group",
        epsilon=0.2, epsilon_high=0.28,              # clip-higher (DAPO)
        mask_truncated_completions=True,             # overlong filtering (DAPO)
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"},
        log_completions=False,
    )
    trainer = GRPOTrainer(model=args.model, args=cfg, reward_funcs=[reward_rlvr],
                          train_dataset=ds, processing_class=tok, peft_config=lora)

    # ---- ASSERTS obrigatórios de módulos -------------------------------------
    lora_a = [n for n, _ in trainer.model.named_parameters() if "lora_A" in n]
    ff = [n for n in lora_a if ".feed_forward." in n]
    attn = [n for n in lora_a if ".self_attn." in n]
    gate = [n for n in lora_a if "gate_proj" in n]
    conv = [n for n in lora_a if ".conv." in n]
    assert not gate, f"lora_A em gate_proj (PROIBIDO): {gate[:4]}"
    assert not conv, f"lora_A em conv (PROIBIDO): {conv[:4]}"
    assert len(ff) == 72, f"esperava 72 lora_A em feed_forward, veio {len(ff)}"
    assert not attn, f"lora_A em self_attn (PROIBIDO): {attn[:4]}"
    assert len(lora_a) == 72, f"lora_A total != 72: {len(lora_a)}"
    trainer.model.print_trainable_parameters()
    premises = {k: v for k, v in vars(args).items()}
    premises.update({"n_lora_A": len(lora_a), "n_feed_forward": len(ff), "n_self_attn": 0,
                     "trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                     "n_prompts": len(recs), "n_answer": n_ans, "coverage_x": args.x or V.COVERAGE_X,
                     "contract": args.contract, "loss_type": "dapo", "scale_rewards": "group", "epsilon_high": 0.28,
                     "generation_batch_size": gen_bs, "enable_thinking": False})
    (out_dir / "premises.json").write_text(json.dumps(premises, indent=2))
    print("[premises]", json.dumps(premises, indent=2), flush=True)
    if args.preflight:
        print("[preflight-ok]")
        return

    class Probe(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "reward" in logs:
                print(f"[step {state.global_step}] reward={logs.get('reward'):.3f} "
                      f"zero_std={logs.get('frac_reward_zero_std', float('nan')):.2f} "
                      f"len={logs.get('completions/mean_length', float('nan')):.0f}", flush=True)

    trainer.add_callback(Probe())
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print("[ok]", flush=True)


if __name__ == "__main__":
    main()
