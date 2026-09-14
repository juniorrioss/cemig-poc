#!/usr/bin/env python3
"""
train_1_2b_tools.py — SFT de TOOL-CALLING do LFM2.5-1.2B-Instruct (roda na DGX Spark).

Reusa INTEGRALMENTE o motor de treino do sft_v2/train_v2 (train_v2.main): 1 época, bs8×ga4,
lr 1e-4, cosine, bf16, grad ckpt, LoRA por REGEX (MLP + atenção; ShortConv fora),
assistant_only_loss, validation set interno e os 3 ASSERTS que ABORTAM (mlp✓ attn✓ conv✗;
72 alvos no 1.2B). Único ajuste: o NOME do artefato GGUF (lfm2.5-1.2b-tools_*), via
monkeypatch de train_v2.convert_and_quant — SEM tocar no motor compartilhado.

Dados: MULTITURNO com tool_calls e role 'tool', tools ANEXADAS ao system (paridade byte-a-byte
com tools=[...], provada em test_render_parity). O chat_template oficial renderiza a região de
geração ({% generation %}) mascarando o turno 'tool' — assistant_only_loss nativo cobre a
DECISÃO de chamar, a CHAMADA (argumentos) e a SÍNTESE, e NÃO o retorno da ferramenta.

Comentários PT-BR; código em inglês.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent / "sft_v2", _HERE.parent / "slm_scripts_v2",
              Path.home() / "cemig-poc" / "sft_v2", Path.home() / "cemig-poc" / "slm_scripts_v2"):
    if (_cand / "train_v2.py").exists():
        sys.path.insert(0, str(_cand))
        break

import train_v2  # noqa: E402


def _convert_and_quant_tools(merged_dir: Path, tag: str, llama_dir: Path, out_dir: Path) -> None:
    """Idêntico ao train_v2.convert_and_quant, mas grava GGUF com prefixo lfm2.5-1.2b-."""
    convert = llama_dir / "convert_hf_to_gguf.py"
    quant = llama_dir / "build-cuda" / "bin" / "llama-quantize"
    bf16 = out_dir / f"lfm2.5-1.2b-{tag}-bf16.gguf"
    q4 = out_dir / f"lfm2.5-1.2b-{tag}-Q4_0.gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(convert), str(merged_dir),
                    "--outfile", str(bf16), "--outtype", "bf16"], check=True)
    subprocess.run([str(quant), str(bf16), str(q4), "Q4_0"], check=True)
    train_v2.logger.info("GGUF exportado: %s / %s", bf16.name, q4.name)


if __name__ == "__main__":
    train_v2.convert_and_quant = _convert_and_quant_tools
    train_v2.main()
