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
`buscar_norma`.

> **Novidade — v1.1 (`v1.1-rlvr/`, modelo do APK v1.1):** o **`v3s400`** é o `f4-r32-ep2` + um
> mini-RLVR de groundedness (GRPO/DAPO, 400 steps, reward verificável por regra). Na bateria
> completa (juiz 27B, régua honesta): **alucinação no distrator ×1536 23,8 → 15,7%**, **fim a fim
> do app ×1536 64,1 → 71,8%**, **151 perguntas coloquiais da POC 12,6 → 21,2%**. É o modelo do
> **APK v1.2** (`apk/cemig-assistente-v1.2.apk`, que baixa os modelos sozinho na 1ª abertura;
> substitui o APK v1.1, que dependia de `adb push`). A raiz do repo continua com o `f4-r32-ep2`
> (APK v1.0, também na tag `v1.0-f4`). Detalhes em "v1.1 — v3s400" abaixo. Este repo traz os **pesos** (safetensors bf16 + GGUF Q4_0) e o **código de
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
CEMIG_MODEL_VARIANT=v1.1-rlvr python inference.py "..."   # v3s400 (modelo do APK v1.1)
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
| `v1.1-rlvr/model.safetensors` + config/código/tokenizer/template | **v3s400** fundido bf16 (sha256 `46c2b126…0716a`, md5 `4213d9ec…`) — `f4-r32-ep2` + LoRA RLVR fundido |
| `v1.1-rlvr/cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf` | GGUF Q4_0 do v3s400 (1,05 GB; md5 `b43caa0dc68f02fbe0eafd58bb2934b3`, sha256 `689735f3…92d6d8`) — **modelo do APK v1.1/v1.2** (baixado pela tela de setup do app) |
| `apk/cemig-assistente-v1.2.apk` | app Android v1.2 (UI F8 + motor do ponto op + v3s400; **baixa os modelos no 1º uso**) — ver "APK Android" |
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

3. **P2 `v3s400` (v1.1)**: mini-RLVR sobre o `f4-r32-ep2` fundido. GRPO com perda DAPO (TRL 1.17,
   clip-higher 0,2/0,28, máscara de truncadas, β = 0), LoRA novo r32 MLP, 8 prompts × 8 gerações,
   lr 5e-6, 400 steps. **Reward binário verificável por regra** (sem juiz no loop): com o trecho
   certo no contexto, cita o item, cobre ≥ 50% dos fatos e todo número existe no contexto; sem
   ele, recusa explícita no início. Pool de prompts filtrado por aprendibilidade (49% do universo
   descartado por ser fácil demais ou mal especificado) e cobrindo os dois formatos: turno de
   ferramenta do app e contexto estático.

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

### v1.1 — v3s400 (bateria completa, juiz 27B, régua honesta; base = `f4-r32-ep2` regerado no mesmo harness)

| avaliação | f4-r32-ep2 (v1.0) | **v3s400 (v1.1)** |
|---|--:|--:|
| distrator ×1536: alucinação (menor é melhor) | 23,8% | **15,7%** |
| distrator ×1536: recusa correta | 66,0% | 73,1% |
| fim a fim do app ×1536 (top-5 → 50%, formato de ferramenta) | 64,1% | **71,8%** (pareado 177/59) |
| fim a fim ×1536: cita o item-ouro | 64,3% | 69,2% |
| oráculo ×1536 | 88,5% | 90,0% |
| ranking ×1536 | 78,1% | 81,6% |
| 151 perguntas coloquiais da POC | 12,6% | **21,2%** (p = 0,011) |
| tokens por resposta (app ×1536) | 104 | 118 (+14%) |
| GGUF Q4_0 × bf16, 20 operacionais | — | 15/20 × 16/20 (pareado 0 ganho / 1 perda) |
| Galaxy S24+, 8 perguntas-exemplo (APK v1.1) | 8/8 | **8/8 texto (2 rodadas) e 8/8 voz** |
| S24+: 1º token visível / total / RAM pico | 8,1 s / 11,6 s / 3,71 GB | 8,1–8,3 s / 11,9–12,3 s / 3,69 GB |

Ressalvas: nas 151 coloquiais o v3s400 recusa menos (53 → 35%) e a alucinação sobe um pouco
(28,5 → 31,1%); no app ×1536 a alucinação fica estável (14,1 → 15,0%, diferença não significativa).
Responde ~14% mais longo. Os números de aparelho (linha S24+) vêm da mesma sessão, com o mesmo
código e só o GGUF trocado.

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

## APK Android (app de campo)

**Atual: `apk/cemig-assistente-v1.2.apk`** (`versionName 1.2.0`, `versionCode 3`). A v1.2 corrige
a distribuição: **instala limpo em qualquer aparelho**, sem PC nem `adb`. O modelo é o mesmo da
v1.1 (**v3s400** Q4_0).

> **Por que a v1.1 falhava em outro aparelho:** o APK v1.1 não trazia o modelo de linguagem, o
> encoder de busca nem o modelo de voz. No aparelho de desenvolvimento eles tinham sido copiados
> via `adb push`. Numa instalação limpa o motor não carregava ("Motor Llama não está carregado" /
> `ggml-base-q5_1.bin`). Além disso, os índices densos de busca (`dense_*.bin`) também não vinham
> no APK, então a busca caía só em BM25 sem avisar. **A v1.1 foi substituída pela v1.2.**

### Instalação (celular, sem PC)

1. Baixe `apk/cemig-assistente-v1.2.apk` deste repositório no celular e abra o arquivo. O Android
   vai pedir para permitir a instalação de apps desta fonte. Se houver uma versão anterior do app,
   desinstale-a antes (a chave de assinatura é a de testes da POC).
2. Abra o app **com internet (Wi-Fi recomendado)**. Aparece a tela **Configuração inicial**, que
   lista os modelos que faltam. Toque em **Baixar modelos (2,01 GB)**.
3. Aguarde a barra de progresso (no S24+ por Wi-Fi levou **~3,6 min**). Se a conexão cair ou o app
   for fechado, toque em **Baixar modelos** de novo: o download **continua de onde parou**. Cada
   arquivo é conferido por **tamanho e SHA-256** antes de ser usado.
4. Quando termina, o app carrega os modelos (~5 s) e mostra "Pronto · 100% offline". A partir daí
   funciona em **modo avião**. Permita o microfone para usar a voz.

| | |
|---|---|
| pacote | `br.org.ceia.cemigpoc` (assinado com a chave de testes da POC, não é release de loja) |
| APK | 250 MB (249.566.486 bytes), SHA-256 `19c4e12e929cbe60bb52ea5d54e3f728299076df3583c9064b75dbe48e91639b` |
| requisitos | Android 12+ (minSdk 31), **arm64-v8a**, ≥ 8 GB de RAM (pico medido ~3,7 GB PSS) |
| espaço | **~2,6 GB livres**: 250 MB do APK + 2,01 GB de modelos + ~110 MB de índices copiados na 1ª abertura. O app confere o espaço antes de baixar |
| download inicial | **2,01 GB, uma única vez** (internet só na configuração inicial; depois 100% offline) |
| validado em | Samsung Galaxy S24+ (SM-S926B), **instalação limpa** (app desinstalado, dados apagados), setup pela tela do app |

### O que o app baixa na configuração inicial

Os arquivos ficam no armazenamento interno do app e são apagados quando o app é desinstalado.

| arquivo | origem (HF, público) | tamanho | SHA-256 |
|---|---|---|---|
| `lfm3-1.7b-v3s400-Q4_0.gguf` | `v1.1-rlvr/cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf` **deste repo** | 1,05 GB | `689735f3…92d6d8` |
| `embeddinggemma-300M-qat-Q4_0.gguf` | [`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF) | 278 MB | `50d28e22…dbccf26` |
| `nemotron-3.5-official-560ms-int8/` (`encoder/decoder/joiner.int8.onnx` + `tokens.txt`) | [`csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11) | 682 MB | encoder `012e9321…e896e68` |

**Dentro do APK**: o app (UI F8: chat, voz segurar-para-falar com revisão da transcrição,
perguntas-exemplo, fontes NR/item, Modo Engenharia), os runtimes nativos (llama.cpp com KleidiAI,
sherpa-onnx/ONNX Runtime, SQLite com FTS5) e todos os assets de busca do ponto de operação F6:
índice FTS5 das 36 NRs, **índices densos `dense_text.bin`/`dense_exponly.bin`** (novos no APK na
v1.2), manuais (FTS5 + denso), janelas do compressor de contexto, classificador de NR e o system
prompt de tool-calling. O Whisper (`ggml-base-q5_1.bin`) só é usado em builds com
`-Pcemig.asrEngine=whisper` e não faz parte desta distribuição.

**Mensagens de erro**: se faltar algum modelo, o app volta para a Configuração inicial e diz qual
arquivo falta. Se o modelo de voz não carregar, o app funciona por texto e avisa que a voz está
indisponível.

### Alternativa com PC (`adb`)

Os arquivos colocados em `/sdcard/Android/data/br.org.ceia.cemigpoc/files/` têm prioridade e
pulam o download (útil sem internet no celular):

```bash
pip install -U huggingface_hub
hf download juniorrios/automobil-ultra-semnexo apk/cemig-assistente-v1.2.apk v1.1-rlvr/cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf --local-dir cemig
hf download ggml-org/embeddinggemma-300M-qat-q4_0-GGUF embeddinggemma-300M-qat-Q4_0.gguf --local-dir cemig
hf download csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt --local-dir cemig/nemotron-3.5-official-560ms-int8

P=br.org.ceia.cemigpoc; EXT=/sdcard/Android/data/$P/files
adb install cemig/apk/cemig-assistente-v1.2.apk
adb shell pm grant $P android.permission.RECORD_AUDIO
adb shell am start -n $P/.MainActivity; sleep 3; adb shell am force-stop $P   # cria o diretório do app
adb shell mkdir -p $EXT
adb push cemig/v1.1-rlvr/cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf $EXT/lfm3-1.7b-v3s400-Q4_0.gguf
adb push cemig/embeddinggemma-300M-qat-Q4_0.gguf $EXT/
adb push cemig/nemotron-3.5-official-560ms-int8 $EXT/
adb shell chmod -R 775 $EXT        # sem isto o app não lê o diretório criado pelo shell
adb shell am start -n $P/.MainActivity
```

### Versões anteriores

| | arquivo | observação |
|---|---|---|
| v1.0 | removida (`versionCode 1`, `f4-r32-ep2`; estado na tag `v1.0-f4`) | exigia os pesos por `adb push` (sem tela de setup) |
| v1.1 | removida (`versionCode 2`, v3s400) | mesmo problema; substituída pela v1.2 |

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
