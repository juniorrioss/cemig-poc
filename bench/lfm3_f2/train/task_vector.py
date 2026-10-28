"""
task_vector.py — braço B (experimento do capitão): task vector do post-training.

delta = pesos(INSTRUCT) - pesos(BASE)  [mesma arquitetura; shapes verificados]

Pipeline: o LoRA do braço A treinado SOBRE O BASE é mergeado no base
(base_treinado) e então o delta é re-aplicado:

  variante "sum"  : W_final = W_base_treinado + delta
                    (soma direta; nos módulos não treinados equivale ao
                     instruct puro, nos w1/w2/w3 = instruct + contribuição LoRA)

  variante "prop" : escala do delta por proporção da média dos módulos (ideia
                    literal do capitão: "proporção da média"). Para cada tensor
                    treinado t:  s_t = mean|W_base_t| / mean|W_base_treinado_t|
                    W_final_t = W_base_treinado_t + s_t * delta_t
                    (o delta é re-escalado para compensar a mudança de
                     magnitude média que o merge do LoRA introduziu no base;
                     nos tensores não treinados s_t = 1 -> igual à soma direta)

Fórmulas exatas documentadas no README. Saída: diretório HF completo por
variante, pronto para avaliação.

Uso (na Spark):
    python task_vector.py --base LiquidAI/LFM3-1.7B-Base \
        --instruct LiquidAI/LFM3-1.7B \
        --adapter runs/lfm3-f2-B-base-r8/checkpoint-XXX \
        --variant sum --out models/lfm3-f2-B-sum-r8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--adapter", required=True, help="dir do LoRA treinado sobre o BASE")
    ap.add_argument("--variant", choices=["sum", "prop"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[1/4] carregando BASE + LoRA e mergeando...")
    base_trained = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.float32)
    base_trained = PeftModel.from_pretrained(base_trained, args.adapter)
    base_trained = base_trained.merge_and_unload()
    sd_trained = base_trained.state_dict()

    print("[2/4] carregando BASE puro e INSTRUCT...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.float32)
    sd_base = base.state_dict()
    instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct, trust_remote_code=True, dtype=torch.float32)
    sd_instr = instruct.state_dict()

    # verificação de shapes (ordem do brief)
    assert set(sd_base.keys()) == set(sd_instr.keys()), "state dicts divergem em chaves"
    for k in sd_base:
        assert sd_base[k].shape == sd_instr[k].shape, f"shape difere em {k}"
    assert set(sd_trained.keys()) == set(sd_base.keys()), "merge mudou as chaves"

    print(f"[3/4] aplicando delta (variante={args.variant})...")
    stats = {"variant": args.variant, "adapter": args.adapter,
             "changed_tensors": 0, "scales": {}}
    new_sd = {}
    for k in sd_base:
        delta = sd_instr[k] - sd_base[k]
        wt = sd_trained[k]
        trained_here = not torch.equal(wt, sd_base[k])
        if args.variant == "sum" or not trained_here:
            new_sd[k] = wt + delta
        else:  # prop, tensor treinado
            m_base = sd_base[k].abs().mean()
            m_trained = wt.abs().mean()
            s = (m_base / m_trained).item() if m_trained > 0 else 1.0
            new_sd[k] = wt + s * delta
            stats["scales"][k] = round(s, 6)
        if trained_here:
            stats["changed_tensors"] += 1

    print(f"  tensores alterados pelo treino: {stats['changed_tensors']}")

    print("[4/4] salvando...")
    out = Path(args.out)
    # reusa o esqueleto do instruct (config/template de chat do instruct!)
    instruct.load_state_dict(new_sd)
    instruct = instruct.to(torch.bfloat16)
    instruct.save_pretrained(str(out))
    tok = AutoTokenizer.from_pretrained(args.instruct, trust_remote_code=True)
    tok.save_pretrained(str(out))
    (out / "task_vector_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
