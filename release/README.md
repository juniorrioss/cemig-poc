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
| `apk/cemig-assistente-v1.0.0-f8.apk` | app Android (UI F8 + motor do ponto op) — ver "APK Android" |
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

## APK Android (app de campo, sideload)

`apk/cemig-assistente-v1.0.0-f8.apk` — **exatamente o APK release instalado e validado no Galaxy S24+**
em 05/11/2026 (SHA-256 do arquivo = SHA-256 do `base.apk` puxado do aparelho via `adb shell pm path`).

| | |
|---|---|
| versão | `versionName 1.0.0` (`versionCode 1`), build F8 — commit `31ecde7` da POC |
| tamanho | 236 MB (235.986.717 bytes) |
| SHA-256 | `91aaf20a6a7e4a9eb531233040c8f06d447ef69a15b3d79bc240dda3473a77c7` |
| pacote | `br.org.ceia.cemigpoc` (assinado com a chave de testes da POC, não é release de loja) |
| requisitos | Android 12+ (minSdk 31), **arm64-v8a**, ≥ 8 GB de RAM (pico medido ~3,8 GB PSS), ~2,2 GB livres p/ os pesos |
| validado em | Samsung Galaxy S24+ (SM-S926B): 8/8 perguntas-exemplo respondidas com a fonte esperada, 1º token ~3–4 s, total ~9–13 s |

**O que vem dentro do APK**: o app completo, com a **UI F8** ("Assistente CEMIG": chat, voz
segurar-para-falar com revisão da transcrição, perguntas-exemplo, fontes NR/item expansíveis,
indicação de busca do tool-calling, Modo Engenharia); runtimes nativos (llama.cpp v0.9.0 com
KleidiAI, sherpa-onnx/ONNX Runtime p/ o ASR, SQLite com FTS5); e todos os **assets de busca do
ponto de operação F6**: índice FTS5 das 36 NRs, manuais (FTS5 + denso), janelas fp16 do
**compressor de contexto**, classificador de NR e o system prompt de tool-calling.

**O que NÃO vem dentro** (grandes demais p/ o APK; o app os lê de
`/sdcard/Android/data/br.org.ceia.cemigpoc/files/`). São 3 downloads públicos, verificados byte a
byte contra os arquivos do aparelho validado:

| arquivo esperado no aparelho | origem | SHA-256 |
|---|---|---|
| `lfm3-1.7b-f4-r32-ep2-Q4_0.gguf` | `cemig-lfm3-1.7b-nr-Q4_0.gguf` **deste repo** (renomear) | `94847d5b…3e85050` |
| `embeddinggemma-300M-qat-Q4_0.gguf` | [`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF) | `50d28e22…dbccf26` |
| `nemotron-3.5-official-560ms-int8/` (`encoder/decoder/joiner.int8.onnx` + `tokens.txt`) | [`csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11) | encoder `012e9321…e896e68` |

**Sideload** (PC com `adb` e depuração USB/Wi-Fi ativa no celular):

```bash
pip install -U huggingface_hub
hf download juniorrios/automobil-ultra-semnexo apk/cemig-assistente-v1.0.0-f8.apk cemig-lfm3-1.7b-nr-Q4_0.gguf --local-dir cemig
hf download ggml-org/embeddinggemma-300M-qat-q4_0-GGUF embeddinggemma-300M-qat-Q4_0.gguf --local-dir cemig
hf download csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt --local-dir cemig/nemotron-3.5-official-560ms-int8

P=br.org.ceia.cemigpoc; EXT=/sdcard/Android/data/$P/files
adb install -r cemig/apk/cemig-assistente-v1.0.0-f8.apk
adb shell pm grant $P android.permission.RECORD_AUDIO
adb shell am start -n $P/.MainActivity; sleep 3; adb shell am force-stop $P   # cria o diretório do app
adb shell mkdir -p $EXT
adb push cemig/cemig-lfm3-1.7b-nr-Q4_0.gguf $EXT/lfm3-1.7b-f4-r32-ep2-Q4_0.gguf
adb push cemig/embeddinggemma-300M-qat-Q4_0.gguf $EXT/
adb push cemig/nemotron-3.5-official-560ms-int8 $EXT/
adb shell chmod -R 775 $EXT        # sem isto o app não lê o diretório criado pelo shell
adb shell am start -n $P/.MainActivity
```

Na 1ª abertura o app copia os assets para o armazenamento interno e carrega os modelos (~20–30 s;
a faixa sob o cabeçalho mostra "Carregando modelos no aparelho…" e depois "Pronto · 100% offline").
Depois disso funciona em **modo avião**. Sem o GGUF/encoder/Nemotron o app não fica pronto.

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
