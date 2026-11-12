# cemig-poc — Assistente CEMIG: perguntas por voz sobre NRs, 100% offline no celular

Assistente de segurança do trabalho para o **operário de campo**. Ele faz a pergunta por voz, em
português coloquial ("posso subir no poste com chuva?"), e o **próprio celular** transcreve a
fala, busca o trecho pertinente nas **36 Normas Regulamentadoras** e nos **9 manuais oficiais do
MTE** e responde em poucas frases, **citando a norma e o item**. Depois da configuração inicial,
tudo roda **em modo avião**: nenhuma chamada de rede e nenhuma nuvem.

É uma prova de conceito CEMIG/CEIA, validada no Samsung Galaxy S24+.

- **Instalar o app:** [APK v1.2 no Hugging Face](https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/apk/cemig-assistente-v1.2.apk)
  ([card com instruções e requisitos](https://huggingface.co/juniorrios/automobil-ultra-semnexo))
- **Como chegamos aqui, passo a passo:** [`bench/CAMINHO.md`](bench/CAMINHO.md)

## Arquitetura

```
 fala (PT-BR)
    │
    ▼  Nemotron 3.5 ASR streaming 0.6B INT8 (sherpa-onnx, CPU)
 [ ASR ] ──► texto da fala, revisável na tela
    │
    ▼  LFM3-1.7B v3s400 (GGUF Q4_0, llama.cpp + KleidiAI) decide: chama buscar_norma ou responde direto
 [ TOOL-CALLING ] (formato XML <tool_call> nativo do LFM3; reusa o contexto no multiturno)
    │ busca com a FALA CRUA
    ▼
 [ RETRIEVER ]  classificador de NR (gate 0,7 / boost 2×)
                + BM25 FTS5 com expansão ASR-robusta + 2 densos EmbeddingGemma-300M
                + manuais do MTE como aba própria (liga só quando o manual empata ou supera a norma)
                ─► RRF ─► top-5
    │
    ▼  COMPRESSÃO a 50% dos tokens do top-2 cru (mantém o código do item)
 [ SÍNTESE ]  o mesmo LFM3 responde com citação (norma + item), ou recusa se o trecho não responde
```

**O modelo.** `LiquidAI/LFM3-1.7B` com três etapas de treino, todas com LoRA r32 só no MLP:
1. SFT de uso robusto de contexto, com o retriever real e o dado em dobro (`data2x-ep4`);
2. SFT de tool-calling multiturno no formato nativo (`f4-r32-ep2`);
3. RLVR curto (GRPO/DAPO, 400 steps) com reward verificável por regra, sem juiz no loop, num pool
   filtrado por aprendibilidade. O resultado é o **`v3s400`**.

## Resultados

Medidos no benchmark `UFGCEMIGONA/benchmark_mobile`: 1536 perguntas, 384 × 4 vocabulários (limpo,
técnico, operacional e leigo). O juiz é o Qwen 27B com a régua honesta: aprovada é a resposta que
cobre ≥ 50% dos fatos pedidos, sem alucinar, sem recusar e sem truncar. As "151" são perguntas
coloquiais reais da POC.

| avaliação | `f4-r32-ep2` (antes do RLVR) | **`v3s400` (app)** |
|---|--:|--:|
| fim a fim do app ×1536 (aprovação) | 64,1% | **71,8%** |
| distrator ×1536: alucinação (menor é melhor) | 23,8% | **15,7%** |
| 151 perguntas coloquiais reais | 12,6% | **21,2%** (p = 0,011) |
| oráculo ×1536 / ranking ×1536 | 88,5 / 78,1 | 90,0 / 81,6 |

No **Galaxy S24+** com o APK: **8/8** perguntas-exemplo aprovadas em texto e em voz. O 1º token
visível leva **~8 s** (8,1–8,3 s) e o total ~12 s. O pico de RAM é **~3,7 GB** PSS (3,69 GB).
Fontes: [`bench/lfm3_p1p2/RESULTS.md`](bench/lfm3_p1p2/RESULTS.md) e
[`bench/lfm3_f9/RESULTS.md`](bench/lfm3_f9/RESULTS.md).

**Limites conhecidos:**
- fala coloquial longa (as 151 ficam em 21%);
- perguntas multihop, que pedem duas buscas;
- cerca de metade das perguntas de manual ainda não recupera o trecho certo.

A resposta não substitui o SESMT: confira sempre o item citado.

## Quickstart

### 1. Instalar o app (celular, sem PC)

1. Baixe o [`cemig-assistente-v1.2.apk`](https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/apk/cemig-assistente-v1.2.apk)
   no celular e instale.
   - Requisitos: Android 12+, arm64, ≥ 8 GB de RAM e ~2,6 GB livres.
   - SHA-256 `19c4e12e929cbe60bb52ea5d54e3f728299076df3583c9064b75dbe48e91639b`.
2. Abra o app com Wi-Fi e toque em **Baixar modelos (2,01 GB)**.
   - O download acontece uma única vez, continua de onde parou e confere o SHA-256 de cada arquivo.
3. Quando aparecer "Pronto · 100% offline", o app já funciona em modo avião.

Para instalar via `adb` sem internet no celular, siga o [card do HF](https://huggingface.co/juniorrios/automobil-ultra-semnexo).

### 2. Rodar a mesma pipeline no PC (`inference.py`)

Mesma busca, mesma compressão e mesmo prompt do app, com o GGUF via llama-cpp-python:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/requirements.txt
curl -LO https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/inference.py
CEMIG_MODEL_VARIANT=v1.1-rlvr python inference.py "Quando trocar de turno, preciso trocar os bloqueios e as etiquetas?"
```

- `CEMIG_MODEL_VARIANT=v1.1-rlvr` seleciona o **v3s400**, o modelo do app.
- `--json`, `--file` e `--retrieval-only` estão descritos no card.
- O espelho do código fica em [`release/`](release/) ([`release/README.md`](release/README.md) e
  [`release/RESULTS.md`](release/RESULTS.md)).

### 3. Build do app

```bash
./android/setup-sdk.sh                      # OpenJDK 17 + Android SDK 34 (idempotente)
export JAVA_HOME="$HOME/android-sdk/jdk-17" ANDROID_HOME="$HOME/android-sdk"
cd android
./gradlew :app:testDebugUnitTest            # testes JVM (paridade do template, retriever, compressor)
./gradlew :app:assembleRelease              # padrão: LFM3 v3s400 Q4_0 + Nemotron
```

Num clone novo, é preciso popular antes:
- os submódulos `android/llama.cpp` e `android/whisper.cpp`;
- os `.so` do sherpa-onnx em `android/sherpa/src/main/jniLibs/arm64-v8a/`;
- a keystore `android/app/poc-release.jks`.

Todos são gitignorados; detalhes em [`bench/lfm3_f10/RESULTS.md`](bench/lfm3_f10/RESULTS.md). Os
pesos não vão no APK: o app os baixa na configuração inicial. Ao trocar o GGUF ou o ASR, atualize o
manifesto em `ModelDownloader.kt`.

## Para integrar o ASR em outro app

O reconhecimento de voz offline (Nemotron 3.5 + sherpa-onnx) tem guia próprio, validado no S24+:

- [`docs/GUIA_ASR_NEMOTRON.md`](docs/GUIA_ASR_NEMOTRON.md): guia para pessoas;
- [`docs/AGENT_GUIDE_ASR.md`](docs/AGENT_GUIDE_ASR.md): o mesmo conteúdo em passos executáveis por
  um agente de código;
- [`docs/asr_teste_ptbr.wav`](docs/asr_teste_ptbr.wav): áudio de teste.

## Onde está cada peça

| pasta | o quê |
|---|---|
| [`android/`](android/) | app Kotlin/Compose: UI "Assistente CEMIG", motores nativos (llama.cpp, sherpa-onnx), retriever e compressor portados com paridade byte a byte |
| [`release/`](release/) | `inference.py` autocontido, card do HF e validação de replicação do zero |
| [`bench/CAMINHO.md`](bench/CAMINHO.md) | a linha vencedora, um passo por seção, com número e artefato |
| `bench/lfm3_f3`, `f3b`, `f4`, `p1p2` | treino do modelo: robustez de contexto, tool-calling e RLVR |
| `bench/lfm3_f5`, `f6` | ponto de operação da busca: gate, compressão e manuais |
| `bench/lfm3_f7`, `f8`, `f9`, `f10` | embarque no S24+, UI, promoção do v3s400 e distribuição do APK v1.2 |
| `retrieval4/`, `classifier/`, `corpus/` | retriever v4 (expansão ASR-robusta + densos), classificador de NR e índice FTS5 das 36 NRs |
| `asr/` | bancada de ASR offline PT-BR |

---

Histórico completo da exploração, com todas as tentativas e os desvios:
[`docs/historico/`](docs/historico/README.md). Notas de ambiente e sharp edges: `AGENTS.md`.
