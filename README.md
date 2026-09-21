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
# tool-calling (dados VERSIONADOS; treino na Spark):  ver "REPRODUÇÃO" abaixo e tools_v1/README.md
# régua de qualidade:                               make -C bench/regua all
```

## REPRODUÇÃO

*"O que eu preciso para replicar cada peça?"* Tudo aqui é **versionado no repo**, exceto os PDFs
das NRs, os pesos treinados e os modelos ONNX de ASR (todos externos, com caminho e comando de
reconstrução indicados). Caminhos são relativos à raiz; a Spark é `walcyrios@spark-b431`.

### 1. SLM (o modelo embarcado: LFM2.5-1.2B tool-trained)

Insumos (todos versionados salvo o base HF e os pesos):

| Insumo | Onde | Estado |
|---|---|---|
| Base | `LiquidAI/LFM2.5-1.2B-Instruct` (Hugging Face, bf16) | externo (público) |
| **Conjunto de treino** | `tools_v1/data/train_full.jsonl` (4.236 diálogos, 5 famílias, 21,1 MB) | **VERSIONADO** |
| Manifesto de procedência | `tools_v1/data/train_full.manifest.json` (contagem por família + **sha256 `be60946f04237e78c159a4caab8d4a9cf3ade2c6ed33ce880d126999a94ae145`**) | **VERSIONADO** |
| Receita de treino | `tools_v1/run_train_tools.sh` (curva de saturação r32 + varredura r16/32/64/128, 1 época, seed 42) | **VERSIONADO** |
| Índice de recuperação | `corpus/index_hf_36nr.db` (FTS5 das 36 NRs) + índice denso v4 (`retrieval4/`) | **VERSIONADO** |
| Régua de avaliação | `bench/regua/` (`data/gabarito_151.jsonl` + `ruler.py`) | **VERSIONADO** |

O `train_full.jsonl` **é o arquivo exato usado nos treinos** (sha256 idêntico ao da Spark em
`~/cemig-poc/tools_v1/data/train_full.jsonl`; conferido nesta publicação). Os splits de degrau
`train_step{750,1500,3000}.jsonl` são **subconjuntos aninhados** (mesma seed 42), reprodutíveis por
`tools_v1/prep_train.py` — por isso **não** são versionados.

```bash
# conferir a integridade do dataset publicado (bate com o manifesto):
sha256sum tools_v1/data/train_full.jsonl   # be60946f...a94ae145
wc -l     tools_v1/data/train_full.jsonl   # 4236

# TREINO (na Spark; ver cabeçalho do script para pré-requisitos de memória unificada):
ssh walcyrios@spark-b431 "cd ~/cemig-poc && WANDB_MODE=online \
  setsid bash tools_v1/run_train_tools.sh > logs/train_tools.log 2>&1 < /dev/null &"

# AVALIAÇÃO (régua honesta, n=151 — mede a QUALIDADE da resposta, citação fora do gate):
make -C bench/regua all         # gabarito → rejudge → analyze (juiz vLLM 27B)
```

> Se você **regerar** o dataset com `tools_v1/gen_dialogs.py` em vez de usar o `train_full.jsonl`
> versionado, o conjunto será **diferente** (depende do 27B gerador) e os números publicados deixam
> de ser comparáveis. Para reproduzir os resultados **exatos**, use o arquivo versionado.

### 2. Classificador de NR — 100% versionado

Nenhuma lacuna: dados rotulados, treinador, fontes dos rótulos e export Kotlin estão todos no repo.

| Insumo | Onde |
|---|---|
| Dados rotulados (1.970 falas) | `classifier/data/labels.jsonl` |
| Treinador | `classifier/train_classic.py` |
| Fontes dos rótulos | `retrieval3/data/expansions.jsonl` + `retrieval4/data/expansions_v4.jsonl` + `corpus/qa_pairs_v2.jsonl` (as 151) |
| Export para o app (Kotlin) | `classifier/data/kotlin_export.json` |

```bash
make -C classifier all          # classic + compare + hybrid + export (Kotlin puro, NrClassifier.kt)
```

### 3. Avaliação do tool-calling — suites derivadas (não armazenadas)

As 17 suites de avaliação do tool **não são versionadas de propósito**: são **derivadas** e
regeneráveis a partir de fontes versionadas, o que é seguro porque as fontes têm **filtro lexical
anti-contaminação** (Jaccard<0.4 vs holdout) e as NRs 33/16/26 ficam reservadas.

```bash
python3 tools_v1/build_eval_sets.py   # regenera as suites (VAL + NRs reservadas + holdout 151)
# medição de qualidade: bench/regua (item 1). Os RESULTADOS (data/*.json) já estão versionados.
```

### 4. Pesos treinados — FORA do repo (decisão pendente do capitão)

Os adapters LoRA e os GGUFs quantizados (~137 GB) **não estão no repo**; publicá-los é decisão
ainda pendente. Como reconstruí-los:

- **Onde estão**: DGX Spark, `~/cemig-poc/models/lfm2.5-1.2b-tools_1_2b_r{16,32,64,128}-{bf16,Q4_0}.gguf`
  (candidato embarcado: **r128 Q4_0**) + `tools_1_2b_step{750,1500,3000}_r32-*.gguf`.
- **Como reconstruir**: `tools_v1/run_train_tools.sh` (item 1) treina LoRA + requantiza para GGUF na
  própria Spark, a partir do base HF + `train_full.jsonl` versionado. Log em `~/cemig-poc/logs/train_tools.log`;
  wandb `cemig-tools-1.2b` (run `lh94m0ve`).
- **Como servir no app**: o GGUF vai como override externo (não embarcado no APK) — ver `## Como rodar o app`.

### 5. ASR — Nemotron 3.5 INT8 (modelos ONNX externos)

O ASR vigente é **Nemotron 3.5 Streaming 0.6B INT8** (sherpa-onnx). Os arquivos ONNX (encoder
657 MB etc.) **não são versionados** (peso); origem, variantes e trade-offs de RAM/WER estão em
[`asr/README.md`](asr/README.md) (INT8 oficial da NVIDIA via sherpa-onnx e o fine-tune PT-BR
`andrewmulya98/sherpa-onnx-ottema-nemotron-3.5-asr-ptbr-560ms-int8`). Bancada:

```bash
make asr-data          # gera os conjuntos de áudio (edge-tts)
make asr-bench         # roda o benchmark no aparelho conectado (ADB)
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
| `sft_v2/` | 4 famílias + varredura de rank + métrica de recusa. | Superado por `sft_v3/` |
| `sft_1_2b/` | SFT de síntese pura no 1.2B. | Superado por `tools_v1/` |
| `bench/` (raiz) | Benchmark de SLMs, juízes, device (harness/perf). | Histórico/fundação |

### Removido na limpeza (camada 1+2) — recuperável pela tag

Estes diretórios foram removidos do `HEAD` para enxugar o repo; o **histórico de commits foi
preservado** e o material vive na tag anotada **`poc-completa-pre-limpeza`** (commit `3bbd529`).
Recupere com `git show poc-completa-pre-limpeza:<caminho>` ou
`git checkout poc-completa-pre-limpeza -- <caminho>`. Inventário e verdicto de cada um em
[`docs/HISTORICO.md`](docs/HISTORICO.md) (seção "Artefatos removidos na limpeza").

| Pasta (na tag) | O quê | Status |
|---|---|---|
| `finetune/` | LoRA do rewriter de query (17 pesos versionados). | **Descartado** · removido |
| `finetune2/` | SFT+DPO do sintetizador 2.6B. | **Descartado** · removido |
| `slm_oraculo/` | Prova do teto em oráculo + SFT de destilação. | Superado por `sft_v2/`/`v3/` · removido¹ |
| `bench/results/`, `bench/judge151/`, `bench/sintese2/` | Saídas de bench/juiz + shootout de sintetizadores. | Superado por `bench/regua/` · removido |
| `retrieval3/results/vec0_*_rank.json` (2 arq.) | Rankings densos intermediários (~19 MB cada); os `bin_*` ficaram. | Intermediário · removido |

¹ O módulo `slm_oraculo/corpus_fix.py` (usado por `sft_v2/` e `tools_oraculo/`, vigentes) foi
**movido para `corpus/corpus_fix.py`** — segue vivo no `HEAD`.

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
