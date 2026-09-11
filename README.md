# cemig-poc — Assistente por voz 100% offline para Normas Regulamentadoras

POC de um assistente de perguntas e respostas por **voz, 100% offline**, sobre Normas
Regulamentadoras (NRs) do domínio elétrico, rodando **inteiramente no dispositivo Android**
(Galaxy S21 e S24+), sem internet no momento da execução.

O usuário-alvo é o **operário de campo**: fala uma pergunta coloquial ("posso subir no poste
com chuva?"), o aparelho transcreve, recupera o trecho da norma pertinente e sintetiza uma
resposta curta **com citação obrigatória da fonte** — tudo em português do Brasil, sem rede.

Não é um produto; é uma prova de conceito para medir, com honestidade, o que é viável
dentro do orçamento de latência (≤10 s de voz) e de RAM (coexistência ASR + SLM em 6–8 GB).

## Arquitetura atual

```
  fala (PT-BR)
      │
      ▼
┌──────────────┐   Whisper Base Q5_1 (whisper.cpp, ~197 MB RAM, RTF 0.84)
│     ASR      │   transcrição on-device
└──────┬───────┘
       │ texto bruto da fala
       ▼
┌──────────────┐   LogisticRegression + TF-IDF (Kotlin puro, <1 ms)
│ Classificador│   prevê a NR; gate de confiança:
│   NR + gate  │   prob≥0.5 → filtro-duro · senão → boost 5× nas top-2
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐   Retrieval v3 (fusão RRF 3 sinais, k=30)
│  BM25-gated-exp (FTS5)        │   ·  BM25 sobre índice expandido (5 campos)
│  + denso EmbeddingGemma-300M  │   ·  denso texto+exp  (query↔passagem)
│  + denso EmbeddingGemma-300M  │   ·  denso exp-only    (query↔query coloquial)
└──────┬───────────────────────┘   → top-2 chunks
       │ 2 chunks + pergunta
       ▼
┌──────────────┐   LFM2.5-1.2B-Instruct QAD Q4_0 (llama.cpp, KleidiAI, thinking-OFF)
│   Síntese    │   prompt conciso (≤4 frases, voz), citação inline obrigatória
└──────┬───────┘
       │
       ▼
   resposta falada/textual com fonte (ex.: "NR-10, item 10.5.1")
```

Todos os componentes rodam nativamente no aparelho. O encoder denso reusa o **mesmo**
`libllama_engine.so` em modo embedding (dispensa ONNX Runtime); a busca densa é
brute-force exato em memória (KNN ~14 ms no S24+) sobre vetores `.bin` (formato DVEC1).

## Números-chave (honestos, com fonte)

| Métrica | Valor | Fonte |
|---|---|---|
| Retrieval R@2 (151 perguntas reais, fala bruta) | **13.9% → 52.3%** | `retrieval3/README.md` (baseline BM25 → fusão v3 3-sinais) |
| Retrieval R@5 v3 | **63.6%** | `retrieval3/README.md` |
| R@2 dos `.bin` embarcados (paridade GGUF) | 51.0% (R@5 63.6%) | `android/README_V3_INTEGRATION.md` |
| E2E S24+ (release, modo avião) | **10/10 ≤10 s, média 6.6 s** | `android/README_V3_INTEGRATION.md` |
| RAM S24+ com encoder residente (PSS) | **2.40 GB** | `android/README_V3_INTEGRATION.md` (on-demand: 1.92 GB) |
| Bancada sintetizador 2.6B thinking-OFF: gate-pass | **22.5%** | `bench/sintese2/README.md` (config `ordering`) |
| Bancada 2.6B: conversão chunk-certo → resposta aprovada | **36.7%** | `bench/sintese2/README.md` |
| ASR Whisper Base Q5_1: WER limpo / ruidoso (10 dB) | 9.5% / 11.8% | `asr/README.md` |

Observação: as latências medidas na GPU de bancada (RTX 5070) **não** valem para o aparelho;
os números de latência/RAM acima são medidos no próprio S24+ via `dumpsys`.

## Mapa de diretórios

| Dir | O quê |
|---|---|
| `corpus/` | Ingestão, chunking e indexação FTS5 das 36 NRs; avaliação de recall (harness honesto). |
| `classifier/` | Classificador leve de NR (LogReg+TF-IDF) + retrieval híbrido confiança-gated; deploy Kotlin puro. |
| `retrieval3/` | Retrieval v3: expansão de documento + fusão RRF BM25+denso (EmbeddingGemma-300M) mobile-first. |
| `bench/` | Benchmark de SLMs (≤1.2B GGUF), juízes automatizados (vLLM 27B), bancadas de síntese e on-device. |
| `finetune/` | LoRA do rewriter de query (enterrado — ver decididas-contra) e trilha limpa/descontaminada. |
| `finetune2/` | Trilha DPO do sintetizador 2.6B (**em curso** — ainda não versionada). |
| `android/` | App Kotlin/Compose: ASR + classificador + retrieval v3 + síntese; engines nativos `:llama`/`:whisper`. |
| `asr/` | Benchmark de ASR offline PT-BR (whisper.cpp, Nemotron/sherpa-onnx, nativo Android). |

## Estado das frentes

### Concluídas
- **Corpus v2** (36 NRs, índice FTS5 grosso `index_hf_36nr.db`); chunking grosso vence fino.
- **Classificador NR** LogReg+TF-IDF (top-2 69%), deploy Kotlin puro com paridade Δprob 2.3e-8.
- **Retrieval v3** mobile-only: R@2 52.3% / R@5 63.6% (**+22.5 p.p.** vs baseline BM25).
- **Integração v3 no app**: encoder denso no mesmo engine, brute-force `.bin`, E2E 10/10 ≤10 s.
- **ASR**: Whisper Base Q5_1 recomendado (197 MB RAM, cabe junto do SLM no S21).
- **Consolidação do pipeline**: Turno 1 de rewrite removido (piorava busca e custava ~1 s).

### Em curso
- **Treino DPO do sintetizador 2.6B** (`finetune2/`): a bancada mostra que o 2.6B thinking-OFF
  converte 3–7× mais que o 1.2B dado o chunk certo; DPO busca fechar a lacuna de qualidade.
- **Upgrade do engine nativo**: destravar `thinking-OFF` do 2.6B no path JNI (hoje o
  `libllama_engine.so` força thinking ON → 44–54 s no aparelho, inviável p/ voz). Ver
  `android/README_CONSOLIDACAO.md`.

### Decididas contra (com evidência)
- **LoRA rewriter** — enterrado: regrediu no holdout limpo (r=8 −8.6 p.p.; r=16 −4.6 p.p. vs
  zero-shot). O "92%" anterior era zero-shot em dataset sintético contaminado. `finetune/README_M3.md`.
- **Tool-calling nativo** — abandonado a favor do fluxo direto ASR→classificador→BM25+denso→síntese
  (mais barato e mais fiel). `android/README_CONSOLIDACAO.md`.
- **Chunking fino/médio** — piora o recall (grosso R@2 teto 77.5% vs fino 51%). `corpus/README_M3.md`.
- **Reranker na nuvem** — só o 27B listwise cruza a meta de 55%, mas é cloud; violaria o
  requisito 100% offline. Cross-encoders mobile e o 1.2B embarcado regridem. `retrieval3/README.md`.

## Como reproduzir

Cada módulo tem seu próprio README com comandos e achados:

- Corpus / índices: [`corpus/README.md`](corpus/README.md) · `make -C corpus v2-all`
- Classificador / híbrido: [`classifier/README.md`](classifier/README.md) · `make -C classifier all`
- Retrieval v3: [`retrieval3/README.md`](retrieval3/README.md) · `make -C retrieval3 all`
- Benchmark SLMs / juízes: [`bench/README.md`](bench/README.md) · `make -C bench all-v2`
- Bancada de síntese: [`bench/sintese2/README.md`](bench/sintese2/README.md) · `make -C bench/sintese2 all`
- Rewriter (histórico): [`finetune/README.md`](finetune/README.md)
- ASR: [`asr/README.md`](asr/README.md) · `make asr-bench`
- App Android: [`android/README.md`](android/README.md) e
  [`android/README_V3_INTEGRATION.md`](android/README_V3_INTEGRATION.md)

Build do app (WSL2 headless):

```bash
./android/setup-sdk.sh                       # OpenJDK 17 + Android SDK 34 (idempotente)
export JAVA_HOME="$HOME/android-sdk/jdk-17"
export ANDROID_HOME="$HOME/android-sdk"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
./gradlew test                               # testes unitários JVM
./gradlew assembleRelease                     # APK com todos os pesos embarcados
```

Notas de ambiente estão em `AGENTS.md` (venvs, storage externo de pesos, sharp edges).
Dados externos (PDFs das NRs, pesos GGUF) **nunca** são commitados.

## Hardware-alvo e requisitos

- **Aparelhos**: Galaxy S21 (6 GB, piso) e S24+ (Exynos 2400, 12 GB, alvo confortável).
- **Latência**: orçamento de voz ≤10 s ponta a ponta (ASR + retrieval + síntese).
- **RAM**: ASR (~197 MB) + SLM (~700 MB–2.4 GB conforme config) precisa caber sob o LMK.
  No S24+ o encoder denso residente usa 2.40 GB PSS (default ≥8 GB); modo on-demand
  (1.92 GB PSS) é o fallback para <8 GB.
- **Offline**: sem rede na execução; toda inferência (ASR, classificador, retrieval, síntese)
  roda no dispositivo. Fonte da resposta é **obrigatória**.
- **Build (desktop)**: llama.cpp/whisper.cpp ARM64 com KleidiAI (dotprod/i8mm/fp16);
  bancadas em GPU RTX 5070 (sm_120) apenas para avaliação, não para latência de campo.
