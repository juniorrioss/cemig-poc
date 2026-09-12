# Correção do fluxo de voz + qualidade da transcrição (task poc-asr-fix)

**Aparelho de referência:** Samsung Galaxy S24+ (`SM-S926B`, Exynos 2400, Android 16), `adb 192.168.0.6:41073`.
**Escopo:** (1) o defeito de reprocessamento do ASR relatado pelo capitão; (2) qualidade da transcrição (higiene de áudio + parâmetros do Whisper + comparação Base × Small).

---

## PARTE 1 — O DEFEITO

### Sintoma relatado
> "eu falo, e o assistente responde sem nem esperar a transcrição; depois que a resposta se
> iniciou aparece acima a minha transcrição" e o modelo "responde algo que não existe... ele
> vai responder referente à primeira pergunta". (Explicitamente NÃO é efeito de streaming.)

### Causa raiz (confirmada no código)
Três defeitos somados em `RealAsrEngine.kt` + `MainViewModel.kt`:

1. **Canal com replay (`MutableSharedFlow(replay = 1)`)** — ao reassinar o fluxo (o coletor
   do ViewModel roda dentro de `viewModelScope`), o **último `AsrEvent.Final` antigo era
   re-entregue**, disparando `processQuestion()` com a transcrição da gravação ANTERIOR.
   Este é o "responde referente à primeira pergunta".
2. **STATUS misturado com CONTEÚDO** — `AsrEvent.Partial` era usado tanto para textos de
   status ("Gravando áudio...", "Transcrevendo...") quanto para o texto transcrito, ambos
   escritos em `currentQuestionInput`. O status poluía o campo de pergunta.
3. **`Final` disparava `processQuestion()` automaticamente** — sem o operário poder revisar,
   e antes mesmo da transcrição estar visível ("responde sem esperar").

### Correção implementada
Contrato de eventos reescrito (`domain/engine/AsrEngine.kt`):
- **`AsrEvent.Status(stage, recordingId)`** — só STATUS (gravando/transcrevendo) → indicador
  de etapa. NUNCA toca em `currentQuestionInput`.
- **`AsrEvent.Partial/Final(text, recordingId)`** — só CONTEÚDO transcrito.
- **Todo evento carrega `recordingId`** (sequência monotônica por gravação).

Motor (`data/engine/RealAsrEngine.kt`, espelhado em `data/fake/FakeAsr.kt`):
- **`MutableSharedFlow(replay = 0)`** → nenhum evento antigo é re-entregue a novos coletores.
- `startListening(): Long` incrementa e retorna o `recordingId` da gravação corrente.

Consumidor (`ui/MainViewModel.kt`):
- Guarda `activeRecordingId`; **descarta qualquer evento cujo `recordingId` não seja o corrente**
  (trava anti-reprocessamento).
- `Status` → só `stage`. `Partial/Final` → só `currentQuestionInput`.
- **`Final` NÃO chama `processQuestion()`**: seta `awaitingConfirmation = true` e exibe a
  transcrição editável. O pipeline só roda quando o operário toca **Enviar** (`submitQuestion()`).

UI (`ui/MainScreen.kt`):
- Banner **"REVISE A TRANSCRIÇÃO E TOQUE EM ENVIAR"** enquanto aguarda confirmação.
- Botão Enviar habilitado sempre que há texto e o pipeline não está processando ativamente
  (corrige um wedge: após erro de voz, `stage==ERROR` bloqueava o reenvio digitado).

> **Decisão de UX de campo (com luva):** o brief autorizava confirmação com janela curta de
> cancelamento como alternativa. Adotou-se **confirmação manual explícita** (tocar Enviar) por
> ser o comportamento mais seguro contra o defeito relatado (nunca enviar sem exibir) e porque
> o botão Enviar já é grande e existe. A transcrição fica editável para corrigir erros de ASR
> antes do envio — ganho direto dado o relato de qualidade inconsistente.

### Prova ANTES × DEPOIS

**Reprodução automatizada (JVM, `app/src/test/.../AsrEventContractTest.kt`):**
- `replay1_reentregaFinalAntigo_defeito` — prova que `replay=1` re-entrega o `Final` antigo
  ("primeira pergunta") a um novo coletor. **É o bug.**
- `replay0_comFiltroDeId_naoReprocessa_correcao` — com `replay=0` + filtro por `recordingId`,
  cada `Final` é processado uma única vez e sempre o da gravação corrente. **É a correção.**
- `status_naoCarregaTextoTranscrito` — garante que `Status` não expõe texto.

**Prova no aparelho (S24+, APK debug, logcat):**

| Passo | Log observado | Interpretação |
| :-- | :-- | :-- |
| PTT #1 (segura ~3,5 s) | `RealAsrEngine: Transcrição final (rec=1): 'O que é isso?'` — e **nenhum** `AskPipeline`/`Busca BM25` a seguir | DEPOIS: `Final` **não** auto-dispara |
| PTT #2 (segura ~3 s) | `Transcrição final (rec=2): ...` — **sem** re-entrega de `rec=1` | DEPOIS: nada de reprocessar a gravação antiga |
| Toca **Enviar** | `AskPipeline: Busca BM25 (fala bruta) recuperou 2 chunks em 79ms` → resposta com fontes **NR-10 10.5 / NR-10 Anexo III.I** | Pipeline roda só na confirmação, com a fala CORRENTE |

Confirmado visualmente: o status de erro ("Nenhuma fala detectada") aparece **no banner de
etapa**, não no campo de pergunta (que mantém o placeholder) — STATUS separado de CONTEÚDO.

> **Roteiro de 3 perguntas por voz:** a trava por `recordingId` + `replay=0` + confirmação
> manual elimina, por construção, o reprocessamento entre perguntas consecutivas (provado nos
> testes e nas 2 gravações reais acima). A validação com 3 falas humanas seguidas de conteúdo
> distinto depende de amostras de voz do capitão (ver Parte 2) — a injeção de áudio por ADB
> grava apenas silêncio ambiente do bench.

---

## PARTE 2 — QUALIDADE DA TRANSCRIÇÃO

### 2.1 Higiene de áudio barata no app (`whisper/.../AudioPreprocessing.kt`)
Aplicada em `WhisperCppEngine.transcribe()` antes da inferência (O(n), custo desprezível):
1. **Recorte de silêncio nas pontas** — RMS por frame de 20 ms, limiar relativo a 8% do pico;
   preserva 100 ms de margem; devolve o original se nada for detectado.
2. **Normalização de ganho** — escala o pico para 0,95 (sobe fala fraca de microfone
   longe/luva); **não** amplifica quando o pico < 0,02 (evita estourar ruído de fundo).
3. **16 kHz mono** confirmado no `AudioRecordRecorder` (já era o formato de captura).

Cobertura: `whisper/src/test/.../AudioPreprocessingTest.kt` (4 testes JVM verdes).

### 2.2 Parâmetros do Whisper (`whisper/src/main/cpp/whisper_jni.cpp`)
O JNI **desabilitava** o temperature fallback (`temperature_inc = 0.0f`) e não tinha os gates
de qualidade. Reativados os defaults recomendados do whisper.cpp para fala espontânea:
`temperature_inc = 0.2`, `entropy_thold = 2.4`, `logprob_thold = -1.0`, `no_speech_thold = 0.6`,
`suppress_blank`, `suppress_nst` (suprime tokens non-speech). O **prompt inicial pt-BR** foi
enriquecido com jargão de segurança (talabarte, trava-quedas, cinto tipo paraquedista, luva
isolante, capacete com jugular) além das NRs e grandezas elétricas.

**Ablação no S24+ (áudio noisy 10 dB, mesmo modelo Base Q5_1) — `asr/results/prompt_ablation/`:**

| Frase | Config ANTIGA (sem fallback, sem prompt) | Config NOVA (fallback + prompt de domínio) |
| :-- | :-- | :-- |
| 06 | capacete **conjugular** | capacete **com jugular** ✓ |
| 19 | **NER-35** ... cinto tipo **para que dista** | **NR-35** ... cinto tipo **paraquedista** ✓ |

O prompt de domínio ancora códigos de NR e jargão; o fallback recupera decodes que os gates
reprovariam. Sem regressão nas demais frases da amostra.

### 2.3 Comparação Whisper Base × Small no S24+ (medição FRESCA)
Bancada: `whisper-cli` (build ARM64 `~/whisper.cpp/build-android`), `memtime` (RSS via
`getrusage`), 50 áudios (25 limpos + 25 noisy 10 dB), duração média ~5,65 s.
Reprodução: `python3 asr/scripts/benchmark.py --models "base q5,small q5"`.

| Modelo | Tamanho | Pico RAM | RTF (limpo/ruído) | Latência média | WER (limpo/ruído) | KW Acc (limpo/ruído) |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Whisper Base Q5_1** | 56,9 MB | **198 MB** | **0,53 / 0,58** | **~3,0 s** | 9,5% / 11,2% | 87,1% / 80,6% |
| Whisper Small Q5_1 | 181,3 MB | 466 MB | 2,37 / 2,48 | **~13,3 s** | 8,1% / 6,8% | 88,7% / 87,1% |

**Recomendação: manter Whisper Base Q5_1.**
- Small é **~4,5× mais lento** (RTF 2,37; ~13 s por enunciado de 5,6 s) — **inviável** para o
  teto de ~10 s de voz push-to-talk. RAM 2,4× maior (466 MB) aperta a coexistência com o SLM.
- O ganho de qualidade do Small é modesto em termos técnicos (KW +1,6 p.p. limpo, +6,5 p.p.
  ruído) e não paga a latência. A alavanca correta para qualidade é a **higiene de áudio +
  prompt de domínio + fallback** (acima), que atuam sobre o Base sem custo de latência/RAM.

> **RTF do Base subiu de 0,42→0,53 vs. o relatório histórico** por rodar com o temperature
> fallback reativado (re-decodes ocasionais) — troca deliberada de ~0,5 s por robustez em fala
> real. Continua bem abaixo de 1,0.

### 2.4 Limitação: voz humana real
O relato do capitão ("só acerta se falar bem lento e pausado") é sobre **fala humana natural**.
As medições de WER acima usam **áudio sintético TTS** (`edge-tts`), que tem prosódia estável e
**não** captura sotaque, hesitações e respiração ofegante de campo — serve como **ranking
relativo reprodutível**, não como WER de produção. A higiene de áudio e o prompt de domínio
foram desenhados exatamente para o caso de fala natural (ganho baixo, silêncio nas pontas,
jargão), mas **a validação de WER com voz real ainda não foi feita** — depende de amostras
gravadas pelo capitão (voz humana, ritmo natural, ruído ambiente). Ver `needs-decision`.

---

## Como reproduzir

```bash
# Testes JVM (contrato de eventos ASR + higiene de áudio)
cd android && ./gradlew :app:testDebugUnitTest :whisper:testDebugUnitTest

# Build + instalar no S24+
./gradlew :app:assembleDebug
adb -s 192.168.0.6:41073 install -r app/build/outputs/apk/debug/app-debug.apk

# Benchmark Base × Small no aparelho (após stage de binários/áudio — ver asr/README.md §6)
python3 asr/scripts/benchmark.py --models "base q5,small q5"
```

Artefatos: `asr/results/benchmark_summary.{json,csv}`, `asr/results/prompt_ablation/`.
