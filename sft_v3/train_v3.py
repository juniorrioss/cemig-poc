#!/usr/bin/env python3
"""
train_v3.py — SFT v3 do LFM2.5-2.6B (roda na DGX Spark). Ranks r64 e r128, 1 época.

É o train_v2.py com 3 ajustes pedidos pelo capitão para o v3:
  1. dados REBALANCEADOS (recusa <= 25%): train_v3.jsonl (oráculo 75%, recusa/parcial/distrator).
  2. ranks r64 e r128 (no v2 o rank ainda estava subindo — r64 foi o melhor em tudo). alpha=2r.
  3. wandb ONLINE (a chave existe no ~/.netrc da Spark; sync retroativo dispensado).

Mantém INTACTO do v2 (exigência permanente do capitão): 1 época, bs8×ga4, max_len 2048,
lr 1e-4, cosine, bf16, gradient checkpointing, targets por REGEX (MLP+atenção, conv fora),
assistant_only_loss, validation set interno (eval_loss ao fim da época), e os 3 ASSERTS que
ABORTAM (mlp presente, attn presente, conv AUSENTE) + print_trainable_parameters publicados
ANTES de treinar.

Se r128 mostrar instabilidade (grad_norm explodindo / eval_loss pior que r64), o run_train_v3
detecta e o relatório propõe lr menor — NÃO se decide sozinho baixar o lr (ordem do brief).

Comentários PT-BR; código em inglês.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reusa integralmente o motor de treino do v2 (rank-configurável, com os asserts).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "sft_v2"))

from train_v2 import main  # noqa: E402

if __name__ == "__main__":
    # O train_v2.main() lê --lora-r/--tag/--data da CLI; o run_train_v3.sh passa r64/r128,
    # tag sft_v3_r{r}, e o dataset rebalanceado train_v3.jsonl.
    main()
