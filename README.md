# cemig-poc — Assistente por voz 100% offline para Normas Regulamentadoras

POC de um assistente de perguntas e respostas por **voz, 100% offline**, sobre Normas
Regulamentadoras (NRs) do domínio elétrico, rodando **inteiramente no dispositivo Android**
(Galaxy S21 e S24+), sem internet no momento da execução. O usuário-alvo é o **operário de
campo**: fala uma pergunta coloquial ("posso subir no poste com chuva?"), o aparelho transcreve,
recupera o trecho da norma pertinente e sintetiza uma resposta curta **com citação obrigatória da
fonte** — tudo em PT-BR, sem rede.

Não é um produto; é uma prova de conceito para medir, com honestidade, o que é viável dentro do
orçamento de latência (≤10 s de voz) e de RAM (coexistência ASR + SLM em 6–8 GB). O repositório
tem 18 pastas de experimento; **a maioria é histórico**. Comece pelos documentos abaixo antes de
abrir qualquer pasta.

## ▶ ONDE COMEÇAR (handoff)

| Documento | Para quê | Papel |
|---|---|---|
| **[`docs/AS_IS.md`](docs/AS_IS.md)** | O que é a **última entrega** e **por que** a arquitetura ficou assim (registro de decisões com número + impacto de mudar). | **Continuidade** — leia primeiro se vai continuar o projeto. |
| [`docs/HISTORICO.md`](docs/HISTORICO.md) | Mapa cronológico dos experimentos: pergunta / veredito / quem superou quem. | Contexto histórico (o que não repetir). |
| [`docs/relatorio_poc.html`](docs/relatorio_poc.html) | Storytelling autocontido da POC. | **Apresentação** (não é o doc de engenharia). |

> Em 2 minutos, o `AS_IS.md` responde: *por que 2 chunks e não 5? por que a busca não usa a
> consulta do modelo? por que 1.2B e não 2.6B?* — cada um com o número e o arquivo-fonte.

## Arquitetura vigente (a última entrega)

Pipeline **híbrido de tool-calling**: o modelo decide **quando** buscar/reusar; quando busca, a
busca externa usa a **fala bruta** (não a consulta reescrita pelo modelo). Detalhes e números em
[`docs/AS_IS.md`](docs/AS_IS.md) · integração em
[`android/README_TOOLS_HYBRID.md`](android/README_TOOLS_HYBRID.md).

```
  fala (PT-BR)
      │
      ▼  Nemotron 3.5 INT8 (sherpa-onnx, ~1,2 s, 91,9% termos técnicos)
   [ ASR ]
      │ fala BRUTA
      ▼  LFM2.5-1.2B tool-trained r128 Q4  →  DECIDE: buscar_norma(...) ou responder direto
   [ DECISÃO do modelo ]
      │ (chamou?)
      ▼  IGNORA a consulta do modelo; busca com FALA BRUTA + classificador NR (gate) + RRF v4
   [ BUSCA EXTERNA ]  → top-2 chunks
      │
      ▼  o mesmo 1.2B sintetiza (citação inline, n_ctx 2048, ≤1 busca/turno)
   [ SÍNTESE ]
      │
      ▼
   resposta com fonte (ex.: "NR-10, Anexo II")
```

**Números do vigente (régua honesta, n=151, `bench/regua`)**: oráculo aprovação **31,4% (Q4) /
34,9% (bf16)**; híbrido E2E **19,2%** (alucinação 34,2% Q4); recusa F1 **85,4% (Q4) / 90,5%
(bf16)**; decisão de chamar **F1 95%**. **No S24+**: RAM PSS **3,86 GB**, latência **6–8 s** sem
busca / **10–12 s** com busca. Opção por flag: **2.6B tool** (`-Pcemig.toolModel=2.6b`, oráculo
46,1% Q4, mas **18–29 s / 4,3–4,75 GB** — não é o padrão por latência de campo). Procedência de
cada número em [`docs/AS_IS.md`](docs/AS_IS.md).

## Como rodar o app

```bash
./android/setup-sdk.sh                       # OpenJDK 17 + Android SDK 34 (idempotente)
export JAVA_HOME="$HOME/android-sdk/jdk-17"
export ANDROID_HOME="$HOME/android-sdk"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"

cd android
./gradlew :app:testDebugUnitTest             # testes JVM (paridade do renderer, contrato ASR)
./gradlew assembleRelease -Pcemig.asrEngine=nemotron   # APK com Nemotron ASR + 1.2B tool

# instalar + servir o GGUF tool como override externo (pesos não vão no APK):
adb install -r app/build/outputs/apk/release/app-release.apk
adb push lfm2.5-1.2b-tools_1_2b_r128-Q4_0.gguf \
  /sdcard/Android/data/br.org.ceia.cemigpoc/files/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf
```

App e engines nativos: [`android/README.md`](android/README.md). Fluxo tool-calling e E2E:
[`android/README_TOOLS_HYBRID.md`](android/README_TOOLS_HYBRID.md). Pesos GGUF/ONNX e dados
externos (PDFs das NRs) **nunca** são commitados — ver `AGENTS.md` (venvs, storage externo,
sharp edges).

## Como reconstruir os modelos

Cada pasta vigente tem seu README com comandos. Treino/quantização rodam na **DGX Spark**; a
avaliação e a régua rodam local (`classifier/.venv`) com o juiz vLLM 27B.

```bash
make -C corpus v2-all         # índice FTS5 das 36 NRs (fundação)
make -C classifier all        # classificador NR + híbrido gated
make -C retrieval4 all        # índice v4 (expansão ASR-robusta) + fusão RRF k=10 (2,1,2)
# tool-calling (dados no sft_v3; treino na Spark):  ver tools_v1/README.md e tools_oraculo/README.md
# régua de qualidade:                               make -C bench/regua all
```

## Mapa das pastas — VIGENTE vs HISTÓRICO

Ver [`docs/HISTORICO.md`](docs/HISTORICO.md) para o veredito de cada uma.

| Pasta | O quê | Status |
|---|---|---|
| `corpus/` | Ingestão/chunking/índice FTS5 das 36 NRs. | **Fundação** |
| `classifier/` | Classificador NR leve + retrieval híbrido confiança-gated (Kotlin puro). | **Vigente** |
| `retrieval4/` | Índice v4 (expansão ASR-robusta) + fusão RRF; classificador+pares-de-graça descartado. | **Vigente** |
| `sft_v3/` | Dados/receita de treino (recusa ≤25%, 4 famílias) reusados pelo tool-calling. | **Vigente** |
| `tools_v1/` | Treino de tool-calling do 1.2B (o modelo decide quando buscar). | **Vigente** |
| `tools_oraculo/` | Desenho **híbrido** validado (oráculo/recusa/híbrido) — o que foi ao app. | **Vigente** |
| `tools_2_6b/` | Tool-calling no 2.6B — 2ª opção de embarque (por flag). | **Vigente (opção 2)** |
| `android/` | App Kotlin/Compose + engines nativos `:llama`/`:whisper`/`:sherpa`. | **Vigente** |
| `asr/` | Benchmark de ASR offline PT-BR (Whisper, Nemotron, nativo). | **Fundação** |
| `bench/regua/` | Régua honesta de qualidade (cobertura de fatos, citação fora do gate). | **Fundação** (métrica oficial) |
| `bench/prompt_teto/` | Teto do 27B + otimização de prompt (rejeitada). | **Fundação** |
| `bench/ctx_topk/` | Janela × nº de trechos — confirma topK=2. | **Fundação** |
| `docai/` | Reprocessamento de tabelas (Document AI) — reparo cirúrgico. | **Fundação** |
| `retrieval3/` | Retrieval v3 (expansão + fusão RRF). | Superado por `retrieval4/` |
| `slm_oraculo/` | Prova do teto em oráculo + SFT de destilação. | Superado por `sft_v2/`/`v3/` |
| `sft_v2/` | 4 famílias + varredura de rank + métrica de recusa. | Superado por `sft_v3/` |
| `sft_1_2b/` | SFT de síntese pura no 1.2B. | Superado por `tools_v1/` |
| `finetune/` | LoRA do rewriter de query. | **Descartado** |
| `finetune2/` | SFT+DPO do sintetizador 2.6B. | **Descartado** |
| `bench/` (raiz) | Benchmark de SLMs, juízes, `judge151`, `sintese2`, device. | Histórico/fundação |

## Hardware-alvo e requisitos

- **Aparelhos**: Galaxy S21 (6 GB, piso) e S24+ (Exynos 2400, 12 GB, alvo). O vigente (Nemotron +
  1.2B) mede **3,86 GB PSS no S24+**; no **S21 aperta o LMK** — para baixa RAM o `asr/README.md`
  recomenda Whisper Base como ASR. Não há E2E medido no S21 (ver lacunas em `docs/AS_IS.md`).
- **Latência**: orçamento de voz ≤10 s ponta a ponta.
- **Offline**: sem rede na execução; toda inferência roda no dispositivo. Fonte da resposta é
  **obrigatória**. Bancadas em GPU (RTX 5070 / DGX Spark) servem só à **qualidade** — a latência de
  campo é medida/projetada no próprio aparelho.

---

*Estado desta POC e o porquê de cada escolha: [`docs/AS_IS.md`](docs/AS_IS.md). História completa:
[`docs/HISTORICO.md`](docs/HISTORICO.md). Notas de ambiente e sharp edges: `AGENTS.md`.*
