"""
train_grpo_k2v_h100.py — braço C (K2V via TRL GRPOTrainer) adaptado p/ H100.

Derivado de bench/lfm3_f2/train_grpo_k2v.py (Spark) com as mudanças exigidas
pelo contrato do worker H100:

  1. LoRA (peft_config no GRPOTrainer) em target_modules=["w1","w2","w3"]
     — 72 matrizes lora_A, TODAS em *.feed_forward.*, ZERO em *.self_attn.*
     (no LFM3, gate_proj é gate da ATENÇÃO, não MLP). Asserts obrigatórios
     + print_trainable_parameters publicados no log ANTES do treino.
     O script da Spark treinava full-model; aqui o C fica comparável aos
     braços A/B (LoRA nos mesmos módulos).
  2. Judge do checklist PARALELO (ThreadPoolExecutor, 8 workers) — o 27B
     está no MESMO nó (localhost:8005); serial custava até 64 chamadas
     sequenciais por step.
  3. Paths do container cemig_mobile_walcy: dados em /workspace/cemig/data,
     saída em /workspace/cemig/checkpoints, wandb em /workspace/cemig/wandb.
  4. wandb ONLINE (projeto lfm3-1.7b-cemig, runs lfm3-f2-C-k2v-{smoke,grpo}),
     .netrc copiado da Spark para o container.
  5. --preflight: só carrega modelo+LoRA, publica asserts e sai (sem treino).
  6. THINKING-OFF (achado do smoke #1): o LFM3-1.7B é modelo de raciocínio
     com thinking ON por default no chat_template; com ON ele gasta os 1024
     tokens de completion pensando em INGLÊS, nunca emite EOS nem chega a
     'Resposta:' => reward 0 em TODOS os steps (clipped_ratio=1). Fix:
     chat_template_kwargs={'enable_thinking': False} no GRPOConfig (o template
     então injeta '<think></think>' vazio no generation prompt).
  7. FEW-SHOT + dica de nós adjacentes (achado do smoke #2): thinking-OFF
     sozinho ainda dá ~0-4% de acerto base (sem variancia => sem gradiente
     GRPO). Com 1 exemplo few-shot (formato 'Resposta: <entidade>') + os
     nós adjacentes do grafo K2V como contexto (dado que já está no dataset,
     espírito graphgen do paper), medido ~10% acerto base @ temp 1.0 =>
     P(≥1 acerto no grupo de 8) ≈ 57% dos grupos com gradiente.
     max_completion 512 (em 1024 o modelo divaga; força concisão e dobra o
     throughput), temperature 1.0 (mais exploração p/ GRPO).

Reward (inalterado, espírito do paper K2V, answer-gated):
  r = 1.0 * answer_correct + 0.5 * checklist_pass_rate * answer_correct

Uso (DENTRO do container, GPU 2 do nó = cuda:0):
    python train_grpo_k2v_h100.py --preflight
    python train_grpo_k2v_h100.py --smoke     # 50 steps
    python train_grpo_k2v_h100.py             # corrida completa (400 steps)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE_URL = os.environ.get("REGUA_JUDGE_URL", "http://10.100.0.111:8005/v1")
JUDGE_MODEL = os.environ.get("REGUA_JUDGE_MODEL", "Qwen/Qwen3.8-27B-FP8")
JUDGE_WORKERS = int(os.environ.get("K2V_JUDGE_WORKERS", "8"))

SYSTEM = ("Você é um assistente técnico de segurança do trabalho especialista "
          "em Normas Regulamentadoras (NRs) brasileiras. Responda em português "
          "do Brasil. Seja CONCISO: no máximo 4 frases de raciocínio e termine "
          "SEMPRE com a resposta final na última linha, no formato "
          "'Resposta: <entidade>'.")

USER_HDR = ("Preencha a lacuna { } no texto abaixo com a entidade correta "
            "(nome exato). Raciocine brevemente e termine com "
            "'Resposta: <entidade>'.\n\n")

FEWSHOT_U = (USER_HDR +
             "Durante a soldagem, o { } é o equipamento que protege os olhos "
             "do trabalhador contra a radiação intensa do arco elétrico.")
FEWSHOT_A = ("A soldagem gera radiação luminosa intensa que exige proteção "
             "ocular específica. O equipamento que cumpre essa função é a "
             "máscara de solda.\nResposta: máscara de solda")


def build_user(row: dict) -> str:
    adj = row.get("adjacent_nodes", {}) or {}
    ctx = sorted({a for lv in adj.values() for a in lv})
    hint = (("Conceitos relacionados no grafo de conhecimento (a resposta NÃO "
             "está necessariamente entre eles, use-os como contexto): "
             + "; ".join(ctx) + "\n\n") if ctx else "")
    return USER_HDR + hint + row["question"]

JUDGE_PROMPT = """Avalie o RACIOCÍNIO abaixo contra o checklist. Para cada item do checklist, responda se o raciocínio o satisfaz (true/false). Seja rigoroso.

QUESTÃO (preencher lacuna): {question}
RESPOSTA CORRETA: {answer}

RACIOCÍNIO DO MODELO:
{reasoning}

CHECKLIST:
{checklist}

Responda APENAS com JSON: {{"passes": [true, false, ...]}} (mesma ordem do checklist)."""


def deaccent(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", deaccent(t.lower())).strip()


def answer_correct(completion: str, gold: str) -> bool:
    return norm(gold) in norm(completion)


def judge_checklist(question: str, gold: str, reasoning: str,
                    checklist: list[str]) -> float:
    """1 chamada ao 27B; retorna pass-rate [0,1]. Falha => 0 (conservador).
    Parser: pega o ÚLTIMO bloco JSON válido (o 27B pensa antes do JSON)."""
    import requests
    cl = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checklist))
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=gold, reasoning=reasoning[:6000],
            checklist=cl)}],
        "max_tokens": 2048, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{JUDGE_URL}/chat/completions", json=payload,
                          timeout=120)
        content = r.json()["choices"][0]["message"]["content"]
        passes = None
        for m in re.finditer(r"\{[^{}]*\}", content, re.DOTALL):
            try:
                cand = json.loads(m.group(0))
                if "passes" in cand:
                    passes = cand["passes"]
            except Exception:  # noqa: BLE001
                continue
        if passes is None:
            return 0.0
        return sum(bool(p) for p in passes) / max(1, len(passes))
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--data", default="/workspace/cemig/data/k2v_qa_checklist.jsonl")
    ap.add_argument("--run", default="lfm3-f2-C-k2v-grpo")
    ap.add_argument("--smoke", action="store_true", help="50 steps + run smoke")
    ap.add_argument("--preflight", action="store_true",
                    help="só carrega modelo+LoRA, publica asserts e sai")
    ap.add_argument("--out", default="/workspace/cemig/checkpoints")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--wandb-dir", default="/workspace/cemig/wandb")
    args = ap.parse_args()
    if args.smoke:
        args.run = "lfm3-f2-C-k2v-smoke"

    os.environ.setdefault("WANDB_MODE", "online")
    os.environ["WANDB_PROJECT"] = "lfm3-1.7b-cemig"
    os.environ["WANDB_DIR"] = args.wandb_dir
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in
            Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = Dataset.from_list([{
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": FEWSHOT_U},
                   {"role": "assistant", "content": FEWSHOT_A},
                   {"role": "user", "content": build_user(r)}],
        "gold": r["answer"],
        "question": r["question"],
        "checklist": r["checklist"],
    } for r in rows])

    def reward_k2v(completions, gold, question, checklist, **kw):
        texts = [c[0]["content"] if isinstance(c, list) else c
                 for c in completions]
        oks = [answer_correct(t, g) for t, g in zip(texts, gold)]
        rewards = [1.0 if ok else 0.0 for ok in oks]
        # answer-gated: julga APENAS as corretas, em paralelo (27B local)
        idx = [i for i, ok in enumerate(oks) if ok]
        if idx:
            with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
                rates = list(ex.map(
                    lambda i: judge_checklist(question[i], gold[i],
                                              texts[i], checklist[i]), idx))
            for i, rate in zip(idx, rates):
                rewards[i] += 0.5 * rate
        return rewards

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["w1", "w2", "w3"],
    )

    cfg = GRPOConfig(
        output_dir=str(Path(args.out) / args.run),
        run_name=args.run,
        report_to=["wandb"],
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        max_steps=50 if args.smoke else args.max_steps,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        seed=42,
        temperature=1.0,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"},
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        reward_funcs=[reward_k2v],
        train_dataset=ds,
        peft_config=lora,
    )

    # ---- ASSERTS obrigatórios do contrato (publicados no log) -------------
    lora_a = [n for n, _ in trainer.model.named_parameters() if "lora_A" in n]
    ff = [n for n in lora_a if ".feed_forward." in n]
    attn = [n for n in lora_a if ".self_attn." in n]
    assert len(ff) == 72, f"esperava 72 lora_A em feed_forward, veio {len(ff)}"
    assert len(attn) == 0, f"lora_A em self_attn (PROIBIDO): {attn[:4]}"
    assert len(lora_a) == 72, f"lora_A total != 72: {len(lora_a)}"
    trainer.model.print_trainable_parameters()
    premises = {
        "model": args.model, "rank": args.rank, "run": args.run,
        "n_lora_A": len(lora_a), "n_feed_forward": len(ff),
        "n_self_attn": len(attn),
        "trainable": sum(p.numel() for p in trainer.model.parameters()
                         if p.requires_grad),
        "lr": args.lr, "batch": args.batch,
        "num_generations": args.num_generations,
        "max_steps": cfg.max_steps,
        "judge_url": JUDGE_URL, "judge_workers": JUDGE_WORKERS,
    }
    out_dir = Path(args.out) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "premises.json").write_text(json.dumps(premises, indent=2))
    print("[premises]", json.dumps(premises, indent=2), flush=True)

    if args.preflight:
        print("[preflight-ok] asserts passaram; saindo sem treinar.")
        return

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print("[ok]", flush=True)


if __name__ == "__main__":
    main()
