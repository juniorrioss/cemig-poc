---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM3-1.7B
language:
- pt
pipeline_tag: text-generation
library_name: transformers
tags:
- rag
- tool-calling
- seguranca-do-trabalho
- normas-regulamentadoras
- gguf
- lfm3
---

# cemig-lfm3-1.7b-nr — RAG offline de Normas Regulamentadoras (PT-BR)

Melhor configuração da POC CEMIG/CEIA de assistente de segurança do trabalho **offline** para
celular: um **LFM3-1.7B** ajustado (`f4-r32-ep2`) para responder com citação a partir de
trechos das 36 NRs brasileiras e de 9 manuais oficiais do MTE, entregues por uma ferramenta
`buscar_norma`. Este repo traz os **pesos** (safetensors bf16 + GGUF Q4_0) e o **código de
inferência autocontido** (`inference.py`), que baixa os artefatos de busca do dataset
[`juniorrios/cemig-nr-retrieval-assets`](https://huggingface.co/datasets/juniorrios/cemig-nr-retrieval-assets).

> **Modificado a partir de [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)**
> e redistribuído sob a **LFM Open License v1.0** (cópia em `LICENSE`). Os pesos foram alterados
> por fine-tuning LoRA (fundido) — ver "Treino". O uso comercial está sujeito ao limite de
> receita da Seção 5 da licença (US$ 10 milhões/ano).

## Quickstart (5 linhas)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr/resolve/main/requirements.txt
curl -LO https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr/resolve/main/inference.py
python inference.py "Quando trocar de turno, preciso trocar os bloqueios e as etiquetas?"
python inference.py --backend hf "..."   # via transformers bf16 (pip install -r requirements-hf.txt)
```

Saída: resposta com citação (norma e item), fontes usadas e os trechos exatos entregues ao
modelo. `--json` dá 1 objeto por pergunta; `--file perguntas.txt` roda em lote;
`--n-gpu-layers 99` põe o GGUF na GPU; `--retrieval-only` mostra só busca + compressão.
O `pip install` compila o `llama-cpp-python` (precisa de `cmake` e compilador C++): a versão
do PyPI (0.3.36) ainda não reconhece a arquitetura LFM3, então o requirements fixa um commit
que já reconhece.

## O que o `inference.py` executa (ponto de operação F6)

```
fala crua ─► classificador de NR (TF-IDF+LogReg; NR citada → filtro duro;
             p≥0,7 → filtro duro; senão boost 2× nas top-2)
          ─► BM25 FTS5 (36 NRs, 2202 trechos, expansão ASR-robusta)
           + denso EmbeddingGemma-300M (texto) + denso (expansão)
           + manuais (BM25 + denso, peso 1,75) só se o melhor manual ≥ melhor norma (margem 0)
          ─► RRF k=10, pesos (2,1,2[,1,75,1,75]) → top-5
          ─► compressão a 50% dos tokens do top-2 cru (janelas de 3 unidades,
             cos + 0,3·léxico + 0,05/(rank+1), mantém o código do item)
          ─► LFM3 f4-r32-ep2 (system com a ferramenta; turno tool injetado; thinking OFF;
             temp 0,1 · top_k 50 · rep 1,1 · seed 42) → resposta citada
```

Todos os parâmetros estão em `config.json` (cópia no dataset de assets). A busca e a
compressão são determinísticas e iguais nas duas vias de LM.

## Arquivos

| arquivo | o que é |
|---|---|
| `model.safetensors` + `config.json`, `modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py` | modelo fundido bf16 (`trust_remote_code`, transformers ≥ 5.20) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer e template do treino (tool-calling XML do LFM3) |
| `cemig-lfm3-1.7b-nr-Q4_0.gguf` | GGUF Q4_0 (1,05 GB; md5 `d26176d36ed225a9933ad39ad91ece59`), llama.cpp ≥ `19cce69` — padrão do `inference.py` |
| `cemig-lfm3-1.7b-nr-Q4_K_M.gguf` | GGUF Q4_K_M (1,10 GB; md5 `da25fdf5066d70f594fd6d752aee5f86`) — **opcional**: retém mais qualidade que o Q4_0 (−2,3 vs −4,9 p.p. vs bf16), mas o prefill no celular é 25% mais lento. O app Android usa o **Q4_0** por decisão de latência |
| `inference.py`, `requirements.txt`, `requirements-hf.txt`, `pipeline_config.json` | código de inferência autocontido e configuração |
| `LICENSE` | LFM Open License v1.0 (Liquid AI) |

## Treino (resumo)

Base `LiquidAI/LFM3-1.7B` (instruct). Duas etapas de LoRA r=32 só no MLP (`w1/w2/w3`, 72 módulos),
SFT chat com `assistant_only_loss`, **dados 100% sintéticos próprios** gerados a partir do texto
público das NRs (nenhum dado de cliente, nenhuma pergunta de avaliação):

1. **F3b `data2x-ep4`**: uso robusto de contexto recuperado (trechos certos, distratores e
   recusa quando o trecho não responde), 4 épocas, fundido.
2. **F4 `f4-r32-ep2`**: tool-calling multiturno no formato nativo XML do LFM3 (4.624 diálogos,
   6 famílias: decidir buscar/não buscar, reusar contexto, trocar de NR, multihop), 2 épocas,
   checkpoint-284, fundido (md5 do safetensors `a661ef377131b7fb222884d9f80afbe0`).

GGUF: `convert_hf_to_gguf.py` → bf16 → `llama-quantize Q4_0` / `Q4_K_M` (llama.cpp v0.9.0, sem imatrix).

## Métricas-chave (juiz Qwen 27B, régua honesta: cobre os fatos pedidos, sem alucinar, sem recusar)

| avaliação | resultado |
|---|---|
| benchmark mobile, recorte 344 (6 NRs × 4 vocabulários), pipeline completo | **68,0% aprovadas** (norma 75,9% · manual 53,3%) · alucinação 13,2% · recusa 15,4% |
| idem, vocabulário operacional (fala de campo) | 65,1% (o app anterior fazia 50,6%) |
| idem, quando a evidência entra no contexto | 81–83% aprovadas |
| 151 perguntas coloquiais longas da POC | **15,2%** (recusa ~53%) |
| tool-calling (F4) | decisão F1 97–98 · sintaxe XML 100% · reuso 100% · troca de NR 98–100% |
| multihop fim a fim (F4) | **7,1%** |
| GGUF Q4_0 × bf16, 20 perguntas operacionais | 15/20 = 15/20 (Q4 sem perda medida) |
| quantização no recorte 344 (vs bf16) | Q4_0 −4,9 p.p. (p=0,007) · **Q4_K_M −2,3 p.p.** (ruído) |
| GGUF Q4_K_M, mesmas 20 operacionais | 16/20 |
| Galaxy S24+ (CPU, 6 threads, pp450/tg300) Q4_K_M × Q4_0 | decode 27,7 × 29,5 t/s (−6%) · prefill 142 × 190 t/s (−25%) |

Contexto entregue: ~450 tokens; prompt ~1.100 tokens. No Galaxy S24+ o TTFT estimado é ~9,5 s.

## Limitações (leia antes de usar)

- **Não é parecer jurídico nem substitui o SESMT.** Respostas podem estar erradas ou
  incompletas; confira sempre o item citado na NR vigente.
- **Fala coloquial longa** (as 151 da POC) ainda vai mal: o modelo recusa muito quando a
  pergunta pede ações ou dados fora do trecho recuperado.
- **Multihop** (perguntas que exigem 2 buscas) não funciona bem (7%): a 2ª consulta formulada
  pelo 1.7B raramente acha o trecho certo. O `inference.py` faz **uma** busca por pergunta,
  com a fala crua, como no app.
- **Manuais** só entram quando o gate denso liga. Cerca de metade das perguntas cuja resposta
  está num manual ainda não recupera o trecho.
- **Tabelas** das NRs vêm de extração de PDF e algumas estão corrompidas (ex.: Anexo II da NR-10).
- **Corpus congelado** (NRs e manuais de 2026). Atualizações posteriores das normas não estão no índice.
- Amostragem com temperatura 0,1: a mesma pergunta pode variar um pouco entre execuções e entre
  CPU e GPU. A seed fixa a sequência em uma mesma máquina/build.

## Reprodutibilidade

O código-fonte canônico fica em `release/` do repositório da POC (porte fiel de
`bench/lfm3_f5`, `bench/lfm3_f6` e `bench/lfm3_f4`). A paridade foi provada: com os mesmos
vetores de query, busca + compressão + prompt saem **idênticos byte a byte** aos 344 prompts
congelados da F6 (`release/tools/parity_check.py`). O teste de replicação do zero está em
`release/RESULTS.md`.

## Atribuição

- Modelo base: Liquid AI, **LFM3-1.7B** (LFM Open License v1.0).
- Encoder denso (baixado em tempo de execução, não redistribuído aqui): Google
  **EmbeddingGemma-300M** QAT Q4_0 (`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`), sujeito aos
  Gemma Terms of Use.
- Corpus: Normas Regulamentadoras e manuais/guias do Ministério do Trabalho e Emprego
  (documentos públicos do governo brasileiro).
- POC CEMIG / CEIA-UFG.
