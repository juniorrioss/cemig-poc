#!/usr/bin/env python3
"""
train_1_2b.py — SFT do LFM2.5-1.2B-Instruct (roda na DGX Spark). Varredura de RANK, 1 época.

Ordem do capitão (após o v3 do 2.6B): "dispare a MESMA lógica para o modelo de 1.2B, mantenha
os dados utilizados nessa última versão e dispare algumas variantes de rank de forma similar".

Reusa INTEGRALMENTE o motor de treino do v2/v3 (train_v2.main): 1 época, bs8×ga4, max_len 2048,
lr 1e-4, cosine, bf16, grad ckpt, targets por REGEX (MLP + atenção; ShortConv fora),
assistant_only_loss, validation set interno (eval_loss ao fim da época) e os 3 ASSERTS que
ABORTAM (mlp presente, attn presente, conv AUSENTE). Único ajuste: o NOME do artefato GGUF
(lfm2.5-1.2b-* em vez de lfm2.5-2.6b-*) — feito por monkeypatch de train_v2.convert_and_quant,
SEM tocar no motor compartilhado.

PASSO 0 (exigência permanente do capitão) — arquitetura do LFM2.5-1.2B-Instruct CONFIRMADA:
  16 camadas (10 conv + 6 full_attention), MESMOS nomes de módulo do 2.6B
  (self_attn.{q,k,v,out}_proj + feed_forward.{w1,w2,w3} + conv.{in_proj,conv,out_proj}).
  Logo o REGEX do v2 é reaproveitável sem mudança; targets esperados = 6*4 + 16*3 = 72
  (contra 122 no 2.6B). Os 3 asserts do train_v2 continuam válidos e ABORTAM se errar.

RISCO QAD (ordem do capitão): o modelo EMBARCADO é a variante QAD-Q4_0 (quantization-aware
distilled) — que NÃO tem checkpoint HF em alta precisão (é só GGUF). O treino é feito sobre o
base HF LFM2.5-1.2B-Instruct (bf16, o mesmo que a Liquid usou p/ gerar o QAD). Treinar em bf16
e requantizar para Q4_0 pode NÃO recuperar a vantagem do QAD-de-fábrica; isso é medido
explicitamente na Parte 2 (bf16 vs Q4 do treinado vs QAD original embarcado).

Comentários PT-BR; código em inglês.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Reusa integralmente o motor de treino do v2 (rank-configurável, com os asserts).
# Layout local: <root>/sft_v2/train_v2.py; layout Spark: ~/cemig-poc/slm_scripts_v2/train_v2.py.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent / "sft_v2", _HERE.parent / "slm_scripts_v2"):
    if (_cand / "train_v2.py").exists():
        sys.path.insert(0, str(_cand))
        break

import train_v2  # noqa: E402


def _convert_and_quant_1_2b(merged_dir: Path, tag: str, llama_dir: Path, out_dir: Path) -> None:
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
    # Monkeypatch APENAS o nome do artefato (o motor, asserts e LoRA ficam intactos).
    train_v2.convert_and_quant = _convert_and_quant_1_2b
    train_v2.main()
