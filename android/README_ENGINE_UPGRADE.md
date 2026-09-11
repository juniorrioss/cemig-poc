# Engine upgrade — suporte a thinking-OFF no 2.6B (POC CEMIG)

Task `poc-engine-upgrade`. Entrega o **suporte** (flag de build/config) + evidência para
rodar o `LFM2.5-2.6B-Q4_0` como sintetizador **sem thinking** no engine nativo do app.
**NÃO** troca o modelo default (segue 1.2B QAD até o treino DPO fechar — ver `finetune2/`).

## TL;DR

- **Problema (herdado da consolidação):** o `chat_template` do LFM2.5-2.6B termina em
  `<|im_start|>assistant\n<think>` no `add_generation_prompt` e **não expõe** `enable_thinking`
  (verificado no GGUF). O 2.6B é modelo de raciocínio: no S24+ gerava 700 tok de `<think>`,
  **44–54 s/resposta** — inviável para voz. O `--reasoning-budget 0` do `llama-server` vive no
  path jinja do servidor, **ausente no JNI** (C-API `llama_chat_apply_template`).
- **Fix (auto-contido, sem bump de submódulo):** replica o `--reasoning-budget 0` no JNI —
  **anexa um bloco de raciocínio VAZIO `<think></think>\n` ao prompt** e **bane a reabertura do
  token `<think>` (id 124901) via logit-bias** na geração. Provado na RTX 5070 antes de embarcar.
- **Suporte por build, default intacto:** `-Pcemig.synthModel=2.6b` seleciona o 2.6B + thinking-OFF
  (`BuildConfig.LLM_MODEL_NAME`/`SUPPRESS_REASONING`); omitido = **1.2B QAD, thinking-OFF=false**.
- **Validação E2E no S24+ (release-equivalente, 10 perguntas, modo avião):**
  **0/10 respostas com `<think>`**, todas citam a NR correta, **162 tok médios** (150–250 alvo).
  **PORÉM** latência média **23,3 s** e RAM pico **4,32 GB** (ver Veredito).

## Onde está o código

| Camada | Arquivo | O que mudou |
| :-- | :-- | :-- |
| JNI C++ | `llama/src/main/cpp/llama_jni.cpp` | `nativeApplyChatTemplate`/`nativeGenerate` recebem `suppress_reasoning`; anexa `<think></think>\n` e bane token `<think>` |
| Bridge  | `llama/.../LlamaBridge.kt` | assinaturas JNI com `suppressReasoning` |
| Engine  | `llama/.../LlamaCppEngine.kt` | propriedade `suppressReasoning` (default false) |
| Build   | `app/build.gradle.kts` | `-Pcemig.synthModel={1.2b,2.6b}` → `BuildConfig.LLM_MODEL_NAME` + `SUPPRESS_REASONING` |
| App     | `data/model/ModelFileManager.kt` | `LLM_MODEL_NAME` vem de `BuildConfig` |
| App     | `ui/MainViewModel.kt`, `data/acceptance/AcceptanceRunner.kt` | setam `suppressReasoning = BuildConfig.SUPPRESS_REASONING` antes de carregar |

## Por que ESTE fix (evidência descartando as alternativas)

Testado na RTX 5070 (`~/llama.cpp/build-cuda`, `434ddbb`) e no S24+:

1. **Banir só o token `<think>`** (logit-bias) → **FALHA**: o modelo soletra `<think>` por
   sub-tokens e volta a raciocinar (S24+: 46–50 s).
2. **Prefill de `</think>` no texto do prompt** (`...assistant\n<think></think>`) na C-API →
   a C-API **não** prima `<think>` (termina em `assistant\n`, último token = `\n`, não 124901),
   então não havia o que fechar.
3. **Forçar `<think>`+`</think>` como tokens decodificados separados** antes de amostrar →
   thinking sumiu (21 s), **mas o modelo ECOAVA o prompt** ("Contexto normativo consultado:…").
   Causa provável: LFM2.5 é **híbrido (camadas shortconv/recorrentes)** — o estado de convolução
   difere entre decode em lote (prompt) e incremental (tokens forçados).
4. **Anexar `<think></think>\n` como TEXTO ao prompt + banir reabertura de `<think>`** →
   **VENCEDOR**: resposta limpa, correta, sem eco (idêntico ao `--reasoning-budget 0` do servidor,
   que valida na GPU com saída limpa de 52 tok).

O bloco é anexado de forma **idempotente** (só fecha se o jinja completo já tiver primado `<think>`),
então o mesmo código serve C-API e um eventual path jinja futuro. No **1.2B QAD** (não-reasoning) o
flag é `false` → **no-op** (default do app inalterado, revalidado: `thinking-OFF=false`, ~6 s/pergunta).

## Evidência E2E (S24+ SM-S926B, Exynos 2400, modo avião)

`android/acceptance_results_2.6b_thinkoff.json` (build `-Pcemig.synthModel=2.6b`, 2.6B via override
externo). Encoder denso RESIDENTE (default ≥8 GB). RAM medida por `/proc/<pid>/status VmHWM` e `dumpsys`.

| Métrica | 2.6B thinking-OFF | 1.2B QAD (default) |
| :-- | --: | --: |
| Respostas com `<think>` | **0/10** | — |
| NR correta citada | 10/10 | 10/10 |
| Tokens de resposta (média) | **162** (min 59, máx 425) | ~34–77 |
| TTFT (prefill ~1300 tok) | 10–15 s | ~1–2 s |
| Decode | 15–18 tok/s | ~48 tok/s |
| Total por pergunta (média) | **23,3 s** | ~5,97 s |
| ≤10 s de voz | **0/10** | 10/10 |
| **RAM pico (VmHWM)** | **4,32 GB** | ~1,9 GB |

**Overhead do path de template (critério ≤350 ms):** a supressão adiciona apenas uma concatenação
de string (`<think></think>\n`, ~10 chars) + um sampler logit-bias de 1 entrada — **efetivamente
0 ms** sobre o caminho do 1.2B. **Critério atendido.**

## Veredito de embarque

- **O engine agora SUPORTA o 2.6B thinking-OFF** (critério da task: 2.6B respondendo direto, sem
  thinking, com overhead de template desprezível). ✅
- **Mas o 2.6B NÃO cabe no orçamento de voz do aparelho HOJE**, por dois limites intrínsecos ao
  modelo (não ao engine):
  - **Latência 23,3 s** (TTFT 10–15 s + decode 15–18 tok/s) — muito acima do teto de 10 s.
  - **RAM pico 4,32 GB** com o encoder denso residente (2.6B ~1,6 GB + Whisper + EmbeddingGemma +
    KV/graph). Acima da projeção ~3,3 GB e **inviável no S21 (6 GB)**; aperta o S24+ com margem baixa.
- **Aplicada a regra do brief:** entregue o SUPORTE + evidência; **o app fica com o 1.2B QAD**
  (default `-Pcemig.synthModel=1.2b`). A troca definitiva para o 2.6B acontece **quando o treino DPO
  fechar** — e mesmo então exige mitigar latência (ex.: contexto lean topk2, decode menor) e RAM
  (encoder `lazyEncoder`, restringir a S24+/≥8 GB). Detalhes de RAM/latência do 2.6B: também em
  `README_CONSOLIDACAO.md` e `bench/judge151/README.md`.

## Como reproduzir

```bash
export JAVA_HOME="$HOME/android-sdk/jdk-17" ANDROID_HOME="$HOME/android-sdk"

# testes JVM
(cd android && ./gradlew :app:testDebugUnitTest)

# build 2.6B thinking-OFF (weights via override externo; não embarcar 1.6 GB no APK)
(cd android && ./gradlew :app:assembleDebug -Pcemig.synthModel=2.6b)
ADB=$ANDROID_HOME/platform-tools/adb
$ADB install -r -d android/app/build/outputs/apk/debug/app-debug.apk
$ADB push ~/models-poc/LFM2.5-2.6B-Q4_0.gguf \
  /sdcard/Android/data/br.org.ceia.cemigpoc.debug/files/LFM2.5-2.6B-Q4_0.gguf
$ADB shell pm grant br.org.ceia.cemigpoc.debug android.permission.RECORD_AUDIO
$ADB shell am start -n br.org.ceia.cemigpoc.debug/br.org.ceia.cemigpoc.MainActivity --ez run_acceptance true
# RAM pico: $ADB shell cat /proc/$($ADB shell pidof br.org.ceia.cemigpoc.debug)/status | grep VmHWM
$ADB shell run-as br.org.ceia.cemigpoc.debug cat files/telemetry.jsonl   # respostas por turno

# build default (1.2B, sem supressão) — NÃO altera nada
(cd android && ./gradlew :app:assembleDebug)     # BuildConfig.SUPPRESS_REASONING=false
```
