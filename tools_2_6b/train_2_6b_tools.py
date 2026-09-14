#!/usr/bin/env python3
"""
train_2_6b_tools.py — SFT de TOOL-CALLING do LFM2.5-2.6B (roda na DGX Spark).

Reusa INTEGRALMENTE o motor de treino do sft_v2/train_v2 (train_v2.main): 1 época, bs8×ga4,
lr 1e-4, cosine, bf16, grad ckpt, LoRA por REGEX (MLP + atenção; ShortConv fora),
assistant_only_loss, validation set interno e os 3 ASSERTS que ABORTAM (mlp✓ attn✓ conv✗;
122 alvos no 2.6B). O train_v2 JÁ grava o GGUF com prefixo lfm2.5-2.6b- (é o motor original do
2.6B) — nenhum monkeypatch é necessário (ao contrário do tools_v1, que teve de renomear p/ 1.2b).

Dados: os MESMOS 4.236 diálogos do tools_v1 (mesmas 5 famílias, mesmo split) — o valor desta
task é a COMPARABILIDADE maçã-com-maçã com o 1.2B. tools ANEXADAS ao system (baked); o chat
template oficial do 2.6B renderiza a região {% generation %} mascarando o turno 'tool'
(assistant_only_loss cobre DECISÃO + ARGUMENTOS + SÍNTESE, e NÃO o retorno da tool). O corpo dos
turnos é BYTE-A-BYTE idêntico ao 1.2B (provado em verify_2_6b.py); o 2.6B é reasoning, mas os
alvos assistant do dataset não têm campo thinking, então o modelo aprende a responder direto
(thinking-OFF), como o sft_v3.

Comentários PT-BR; código em inglês.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# o motor train_v2 mora em sft_v2/ (repo) ou slm_scripts_v2/ (Spark) — procura em ambos.
for _cand in (_HERE.parent / "sft_v2", _HERE.parent / "slm_scripts_v2",
              Path.home() / "cemig-poc" / "sft_v2",
              Path.home() / "cemig-poc" / "slm_scripts_v2"):
    if (_cand / "train_v2.py").exists():
        sys.path.insert(0, str(_cand))
        break

import train_v2  # noqa: E402

if __name__ == "__main__":
    train_v2.main()
