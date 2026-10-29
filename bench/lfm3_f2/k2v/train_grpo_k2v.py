"""
train_grpo_k2v.py — braço C: implementação do método K2V via TRL GRPOTrainer.

NOTA de implementação (registrada no RESULTS.md): o rollout vllm/sglang do
verl NÃO suporta o LFM3 (arquitetura conv híbrida; erro concreto:
"ValueError: Layer 0 (conv) does not dispatch through the Transformers
attention interface and vLLM has no other way to handle it", vllm 0.32.0).
Decisão do firstmate: manter o MÉTODO K2V (dados graphgen-mask fill-blank +
checklist por questão + answer-gated reward) trocando a lib de RL por TRL
GRPOTrainer, que gera com transformers.generate (suporta LFM3 nativamente).

Reward (espírito do paper, answer-gated):
  r = 1.0 * answer_correct + 0.5 * checklist_pass_rate * answer_correct
  - answer_correct: a entidade mascarada (ground-truth verificável) aparece na
    resposta final (match normalizado, acento/caixa-insensível).
  - checklist_pass_rate: fração dos itens do checklist que o raciocínio da
    resposta satisfaz, julgada pelo 27B (1 chamada por amostra; APENAS quando
    a resposta está correta — gate anti reward-hacking do paper).

Logging: wandb (report_to), run lfm3-f2-C-k2v-{smoke|full}.

Uso (na Spark, venv lfm3_f2/.venv):
    python train_grpo_k2v.py --smoke               # 50 steps obrigatórios
    python train_grpo_k2v.py --run lfm3-f2-C-k2v-full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent

JUDGE_URL = os.environ.get("REGUA_JUDGE_URL", "http://10.100.0.111:8005/v1")
JUDGE_MODEL = os.environ.get("REGUA_JUDGE_MODEL", "Qwen/Qwen3.8-27B-FP8")

SYSTEM = ("Você é um assistente técnico de segurança do trabalho especialista "
          "em Normas Regulamentadoras (NRs) brasileiras. Responda em português "
          "do Brasil. Pense passo a passo e termine com a resposta final na "
          "última linha, no formato 'Resposta: <entidade>'.")

USER_TMPL = ("Preencha a lacuna {{ }} no texto abaixo com a entidade correta "
             "(nome exato). Explique seu raciocínio e termine com "
             "'Resposta: <entidade>'.\n\n{question}")

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
    """1 chamada ao 27B; retorna pass-rate [0,1]. Falha => 0 (conservador)."""
    import requests
    cl = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checklist))
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=gold, reasoning=reasoning[:6000],
            checklist=cl)}],
        "max_tokens": 512, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{JUDGE_URL}/chat/completions", json=payload, timeout=90)
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        passes = json.loads(m.group(0))["passes"]
        return sum(bool(p) for p in passes) / max(1, len(passes))
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--data", default=str(HERE / "data" / "k2v_qa_checklist.jsonl"))
    ap.add_argument("--run", default="lfm3-f2-C-k2v-full")
    ap.add_argument("--smoke", action="store_true", help="50 steps + run smoke")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--max-completion", type=int, default=1024)
    ap.add_argument("--wandb-dir", default=os.path.expanduser("~/cemig-poc/lfm3_f2/wandb"))
    args = ap.parse_args()
    if args.smoke:
        args.run = "lfm3-f2-C-k2v-smoke"

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "cemig-lfm3-f2")
    os.environ["WANDB_DIR"] = args.wandb_dir
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    # prompt conversacional; metadados viajam em colunas extras (TRL repassa como kwargs)
    ds = Dataset.from_list([{
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": USER_TMPL.format(question=r["question"])}],
        "gold": r["answer"],
        "question": r["question"],
        "checklist": r["checklist"],
    } for r in rows])

    def reward_k2v(completions, gold, question, checklist, **kw):
        rewards = []
        for comp, g, q, cl in zip(completions, gold, question, checklist):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            ok = answer_correct(text, g)
            r = 1.0 if ok else 0.0
            if ok:  # answer-gated: raciocínio só pontua com resposta certa
                r += 0.5 * judge_checklist(q, g, text, cl)
            rewards.append(r)
        return rewards

    cfg = GRPOConfig(
        output_dir=str(Path(args.out) / args.run),
        run_name=args.run,
        report_to=["wandb"],
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        max_steps=50 if args.smoke else 400,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        seed=42,
        temperature=0.8,
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"},
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        reward_funcs=[reward_k2v],
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(str(Path(args.out) / args.run / "final"))
    print("[ok]")


if __name__ == "__main__":
    main()
