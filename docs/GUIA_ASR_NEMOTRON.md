# Guia de integração — ASR Nemotron 3.5 (sherpa-onnx) em app Android

Guia para quem vai colocar no próprio app o mesmo reconhecimento de voz offline que usamos no
assistente CEMIG: **Nemotron 3.5 ASR streaming 0.6B, INT8, chunk de 560 ms**, rodando em CPU via
**sherpa-onnx** (ONNX Runtime).

> Tem um agente de código (Claude, Cursor, Copilot…)? Entregue a ele o
> [`AGENT_GUIDE_ASR.md`](AGENT_GUIDE_ASR.md): é o mesmo conteúdo em forma de passos executáveis,
> com código completo e verificação depois de cada passo.

**Status de validação:** o código completo do guia de agente foi compilado contra o AAR oficial
`sherpa-onnx-1.13.7.aar` (e contra a variante `static-link-onnxruntime`) e executado num Galaxy S24+.
Resultado: `text='Posso realizar intervenção em circuito energizado durante chuva forte'` a partir de
um WAV PT-BR (seção 9).

Caminhos como `android/...` e `asr/...` referem-se ao repositório da POC CEMIG/CEIA.

Tudo o que está aqui foi conferido contra o código do app
(`android/sherpa/.../NemotronAsrEngine.kt`, `android/app/.../RealNemotronAsrEngine.kt`,
`android/app/.../ModelDownloader.kt`), contra o fonte do sherpa-onnx e contra o modelo rodando de
verdade (sherpa-onnx 1.13.7 e 2.0.1, 50 áudios PT-BR do `asr/`). Onde um número veio de
medição, a fonte está indicada.

---

## 1. Visão geral

```
microfone ─► AudioRecord 16 kHz mono PCM16 ─► FloatArray [-1,1] ─► OnlineStream (language=pt)
         ─► OnlineRecognizer (encoder/decoder/joiner INT8, CPU, 4 threads) ─► texto
```

No app o modo é **push-to-talk**: o usuário segura o botão, a gravação acumula o áudio e, ao
soltar, o enunciado inteiro é entregue ao modelo de uma vez. O modelo é streaming, mas aqui não
usamos resultado parcial. Ver a seção 6 para entender por que isso importa.

## 2. Arquivos do modelo (682 MB)

Repositório público no Hugging Face (sem token):
[`csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11)

| arquivo | bytes | SHA-256 |
|---|--:|---|
| `encoder.int8.onnx` | 657.601.403 | `012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68` |
| `decoder.int8.onnx` | 14.978.075 | `19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0` |
| `joiner.int8.onnx` | 9.504.438 | `4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce` |
| `tokens.txt` | 131.440 | `729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a` |
| **total** | **682.215.356** | |

URL de cada arquivo: `https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/resolve/main/<arquivo>`.

Os tamanhos e hashes são os mesmos do manifesto do nosso `ModelDownloader.kt` e foram recalculados
sobre os arquivos baixados (`sha256sum`). Também batem com o SHA-256 LFS publicado no repositório
do HF.

Para baixar no PC:

```bash
pip install -U huggingface_hub
hf download csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt \
  --local-dir nemotron-3.5-official-560ms-int8
cd nemotron-3.5-official-560ms-int8 && sha256sum *.onnx tokens.txt
```

**Os arquivos não vão no APK.** O encoder sozinho tem 657 MB. No nosso app ele é baixado na primeira
abertura (tela de setup, `ModelDownloader.kt`: retomada por HTTP Range, verificação de tamanho e
SHA-256, rename atômico para `filesDir/nemotron-3.5-official-560ms-int8/`). No desenvolvimento ele é
copiado por `adb push` para `getExternalFilesDir(null)/nemotron-3.5-official-560ms-int8/`, que tem
prioridade (ver `ModelFileManager.getNemotronModelDir()`).

Licença do modelo: NVIDIA OpenMDW 1.1 (card de
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)).

> **Não confunda com outras variantes.** O mesmo autor publica 80, 160, 320, 560 e 1120 ms, INT8 e
> FP32. Este guia (e os hashes acima) vale **só** para `560ms-int8-2026-06-11`. O
> `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` **não funciona** com sherpa-onnx: falta
> metadado e o erro é `Cannot find vocab_size` (ver `asr/README.md`).

## 3. Runtime: sherpa-onnx para Android

São necessários três componentes, que precisam ser **da mesma versão**:

1. `libsherpa-onnx-jni.so` (arm64-v8a)
2. `libonnxruntime.so` (arm64-v8a). O JNI depende dele (`NEEDED libonnxruntime.so`)
3. Os bindings Kotlin do pacote `com.k2fsa.sherpa.onnx` (`OnlineRecognizer`, `OnlineStream`,
   `OnlineRecognizerConfig`…). Eles chamam `System.loadLibrary("sherpa-onnx-jni")`

Suporte ao Nemotron 3.5 multilíngue entrou no sherpa-onnx **1.13.3** (CHANGELOG: *"Add multilingual
Nemotron-3.5 streaming ASR support (#3671)"*). Use **≥ 1.13.3**. Duas formas de obter:

### Opção A (recomendada): AAR oficial do sherpa-onnx

Baixe `sherpa-onnx-<versão>.aar` em
[github.com/k2-fsa/sherpa-onnx/releases](https://github.com/k2-fsa/sherpa-onnx/releases). Testado:
`sherpa-onnx-1.13.7.aar` (49.113.869 bytes, SHA-256
`c4ef49e309f24fcee5c106b8a279481aaecaabb078cd37b2cd6e9a62cc8a73c8`). Ele já traz os bindings
compilados e os `.so` de 4 ABIs (arm64-v8a, armeabi-v7a, x86, x86_64, com ONNX Runtime). Coloque
em `app/libs/` e:

```kotlin
dependencies {
    implementation(files("libs/sherpa-onnx-1.13.7.aar"))
}
```

O código da seção 4 compila contra esse AAR sem mudanças, e o APK resultante transcreveu no S24+
(testado).

Se o seu app já usa outro ONNX Runtime (ex.: `com.microsoft.onnxruntime`), use
`sherpa-onnx-static-link-onnxruntime-1.13.7.aar` (37.809.521 bytes, SHA-256
`220d7cb25ac6e57ec34082bc2551ebd7ec7d4d96cbfea520839e208f92347837`). Ele traz um único
`libsherpa-onnx-jni.so` de 23,6 MB, com o ORT embutido, e evita o conflito `More than one file was
found with OS independent path 'lib/arm64-v8a/libonnxruntime.so'`. Também transcreveu no S24+.

### Opção B: copiar o que usamos no app

O nosso módulo `android/sherpa/` tem:

- `src/main/java/com/k2fsa/sherpa/onnx/*.kt`: os 22 arquivos de `sherpa-onnx/kotlin-api/` do
  upstream, idênticos ao commit `b0899d9` (versão 1.13.7 + commits posteriores);
- `src/main/jniLibs/arm64-v8a/{libsherpa-onnx-jni.so,libonnxruntime.so}`: **gitignorados**, não
  estão no repositório. Foram compilados com `./build-android-arm64-v8a.sh` do sherpa-onnx (mesmo
  commit, ONNX Runtime 1.28.2, `ANDROID_PLATFORM=android-28`). No build da POC:
  `cp ~/sherpa-onnx/build-android-arm64-v8a/install/lib/{libsherpa-onnx-jni.so,libonnxruntime.so} android/sherpa/src/main/jniLibs/arm64-v8a/`.

Se misturar fontes (bindings de uma versão e `.so` de outra), espere `UnsatisfiedLinkError` ou
`NoSuchFieldError`/`Failed to get field ID` no JNI. Mantenha os três juntos.

## 4. Configuração exata e fluxo de transcrição

Esta é a configuração que roda no app (`NemotronAsrEngine.initModel`):

```kotlin
val config = OnlineRecognizerConfig(
    featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
    modelConfig = OnlineModelConfig(
        transducer = OnlineTransducerModelConfig(
            encoder = "$dir/encoder.int8.onnx",
            decoder = "$dir/decoder.int8.onnx",
            joiner  = "$dir/joiner.int8.onnx",
        ),
        tokens = "$dir/tokens.txt",
        numThreads = 4,
        provider = "cpu",
        modelType = "nemotron",
    ),
    enableEndpoint = false,          // push-to-talk: o app decide início e fim
    decodingMethod = "greedy_search",
)
val recognizer = OnlineRecognizer(assetManager = null, config = config)  // caminhos absolutos
```

E a transcrição de um enunciado (`NemotronAsrEngine.transcribe`):

```kotlin
val stream = recognizer.createStream()
try {
    stream.setOption("language", "pt")              // ver pegadinha 2: a chave é "language"
    stream.acceptWaveform(samples, 16000)            // FloatArray em [-1, 1], 16 kHz, mono
    stream.acceptWaveform(FloatArray(16000 / 2), 16000)  // 0,5 s de silêncio de cauda
    stream.inputFinished()
    while (recognizer.isReady(stream)) recognizer.decode(stream)
    val text = recognizer.getResult(stream).text.trim()
} finally {
    stream.release()
}
```

Notas sobre cada campo, verificadas no fonte do sherpa-onnx:

| campo | valor | o que faz de fato |
|---|---|---|
| `sampleRate` | 16000 | taxa esperada pelo extrator de features. Áudio com outra taxa é reamostrado, **desde que** você passe a taxa verdadeira em `acceptWaveform` |
| `featureDim` | 80 | **sobrescrito** pelo modelo: o encoder declara `feat_dim=128` e o sherpa usa esse valor (`config_.feat_config.feature_dim = model_->FeatureDim()`). 80 ou 128 dão o mesmo texto |
| `modelType` | `"nemotron"` | não é usado para escolher o modelo na CPU: o sherpa detecta o transducer NeMo pelo número de saídas do decoder. Com `""`, `"nemotron"` ou até `"zipformer2"` a saída é idêntica (testado). Mantenha `"nemotron"` por documentação e paridade com o app |
| `provider` | `"cpu"` | o AAR/`.so` padrão só tem CPUExecutionProvider. Outro valor cai em CPU com aviso |
| `numThreads` | 4 | threads do ONNX Runtime. O app usa 4 (`MainViewModel`: `initModel(..., numThreads = 4)`) |
| `enableEndpoint` | false | desliga o endpointing automático. Ver pegadinha 4 |
| `decodingMethod` | `"greedy_search"` | **único suportado** para transducer NeMo. Qualquer outro: `Unsupported decoding method: ...` e o processo **morre** (`_Exit(-1)` no nativo) |

Outros detalhes do app que vale copiar:

- **Cauda de silêncio de 0,5 s.** O app adiciona para "drenar" o último bloco de 560 ms. Nos 50
  áudios de teste ela não mudou nenhum resultado (o `inputFinished()` já completa o bloco), mas
  também não custa nada. É barata e protege contra corte do fim da frase.
- **Áudio mínimo de 300 ms** (`RealNemotronAsrEngine.MIN_SAMPLES = 4800`). Abaixo disso o app
  nem transcreve e avisa "áudio muito curto".
- **Uma stream por enunciado**, liberada com `release()` no `finally`. O recognizer é carregado uma
  vez e reutilizado.
- **Tudo fora da main thread.** O app usa `Dispatchers.IO` (carregamento leva segundos e a
  transcrição leva de 1 a 3 s).
- **Gravação** (`whisper/.../AudioRecordRecorder.kt`): `AudioRecord(MIC, 16000, CHANNEL_IN_MONO,
  ENCODING_PCM_16BIT)`, PCM little-endian convertido com `short / 32768.0f`.

## 5. As 5 pegadinhas

1. **`modelType`: declare, mas não é a causa de falha.** O esqueleto inicial deste guia dizia que
   `modelType` era obrigatório. Medimos: na CPU, com sherpa-onnx 1.13.7 e 2.0.1, o recognizer é
   criado e transcreve igual com `modelType` vazio ou errado. O que dá "falha no load" é **arquivo
   faltando, caminho relativo ou `.so` de versão diferente** (seção 8). Mesmo assim, use
   `modelType = "nemotron"`: é o que o app usa e documenta a intenção.

2. **Idioma por stream: a chave é `"language"`, não `"lang"`.** O Nemotron 3.5 é multilíngue (121
   códigos no `prompt_dictionary` do encoder, incluindo `pt`, `pt-BR` e `pt-PT`). O idioma é passado
   **por stream**, com `stream.setOption("language", "pt")` antes do `acceptWaveform`. O código C++
   lê só `GetOption("language")`. `setOption("lang", "pt")` é aceito sem erro e **ignorado**, e o
   modelo roda em auto-detecção.
   > ⚠️ O nosso `NemotronAsrEngine.kt` usa hoje `setOption("lang", "pt")`, ou seja, **o app roda em
   > auto-detecção**. Nos 50 áudios PT-BR do `asr/` (25 limpos + 25 com ruído de 10 dB), auto × `pt`
   > forçado: o texto muda em 9/50, WER bruta limpo 8,2% × 8,2% e ruído 8,5% × 7,3%. Os erros típicos
   > do auto são perder o artigo inicial ("Luva isolante…" em vez de "A luva isolante…") e escrever
   > números por extenso. No seu app use `"language"`. A correção no nosso é trocar uma palavra e
   > validar no aparelho.

3. **Formato de áudio: float [-1, 1], mono, e a taxa VERDADEIRA.** O teste de pior caso é um texto
   **vazio**, sem exceção:
   - PCM16 passado como float sem dividir por 32768 → saída vazia;
   - taxa errada em `acceptWaveform` (ex.: dizer 8000 ou 44100 para um áudio de 16 kHz) → o sherpa
     reamostra o que não devia → saída vazia;
   - áudio de 8 kHz declarado como 16 kHz (toca "rápido") → lixo parcial ("Squedimentos
     obrigatórios para desganização…").

   Estéreo intercalado passado como mono é o mesmo erro de taxa. Grave mono.

4. **Endpoint desligado em push-to-talk.** Com `enableEndpoint = true` o sherpa usa regras de
   silêncio (padrão: 2,4 s sem fala, ou 1,4 s depois de fala) para dizer "acabou o enunciado". Num
   loop de streaming ao vivo, quem segue o exemplo oficial chama `reset()` quando `isEndpoint()` é
   verdadeiro, e tudo o que foi dito antes da pausa some do resultado. No push-to-talk quem decide o
   fim é o botão. Com o fluxo da seção 4 (áudio inteiro + `inputFinished`), ligado ou desligado dá o
   mesmo texto. Deixe **desligado** para não ter um segundo critério de fim competindo com o botão.

5. **Só arm64-v8a.** O app restringe `abiFilters` a `arm64-v8a` e só testamos aí (Galaxy S24+). O AAR
   oficial traz `.so` de outras ABIs, mas não testamos nenhuma delas, e um encoder INT8 de 657 MB com
   pico de ~800 MB de RAM não é alvo para aparelho 32-bit. Emulador não serve para validar latência.
   Valide em aparelho físico arm64. Se o seu app tiver `.so` só de arm64, um aparelho de outra ABI dá
   `UnsatisfiedLinkError` no primeiro uso de `com.k2fsa.sherpa.onnx`.

## 6. Por que 560 ms

O Nemotron 3.5 é *cache-aware streaming*: processa o áudio em blocos sem sobreposição, e o tamanho do
bloco é escolhido na exportação (80, 160, 320, 560 ou 1120 ms). O trade-off:

- **Bloco maior → mais contexto à direita → mais acurácia, e menos chamadas ao encoder por segundo de
  áudio** (menos overhead por bloco, melhor para CPU e bateria). WER no FLEURS em português, segundo
  o card da NVIDIA, com idioma informado: 80 ms 6,29% · 160 ms 6,10% · 320 ms 5,81% · **560 ms
  5,65%** · 1120 ms 5,48%.
- **Bloco menor → texto parcial aparece mais cedo enquanto a pessoa fala** (latência intra-fala).

**No push-to-talk essa latência não existe:** não mostramos texto parcial, então o usuário só espera
o tempo de processar o enunciado depois que solta o botão. Por isso fica o bloco grande. 560 ms
estava disponível e validado desde a trilha ASR (`asr/README.md`, Exp C). O ganho de 1120 ms em
português é pequeno (0,17 p.p. no FLEURS) e não foi medido no aparelho.

**Medido no Galaxy S24+ (Exynos 2400, release, modo avião, 4 threads, Nemotron 560 ms INT8):**

| fonte | áudio | tempo de transcrição |
|---|---|---|
| `android/acceptance_results_v3_nemotron_2p6b.json` (10 perguntas) | frases curtas | **0,83–1,36 s** (média 1,17 s) |
| `bench/lfm3_f7/data/device/f7_e2e_voice.json` (8 perguntas) | 8,3–13,0 s | **1,35–3,00 s** (média 2,34 s; RTF 0,16–0,24) |
| `bench/lfm3_f10/.../f7_e2e_results_v12_clean_voice.json` (instalação limpa) | 12,7 s | 1,87 s (RTF 0,15) |

Ou seja: cerca de 0,15 a 0,27 s de processamento por segundo de fala, e **~1–2 s para uma pergunta
típica**. Esse tempo começa quando o usuário solta o botão. Na trilha ASR de bancada (CLI no S24+,
`asr/README.md`): WER 6,8% limpo e 8,2% com ruído de 10 dB, 91,9% dos termos técnicos corretos,
**pico de RSS 792 MB**.

RAM: contando com o resto do app (LLM 1,2B + encoder de busca), o PSS medido foi 3,86 GB
(`android/tool_e2e_results.json`). Em aparelhos com menos de 8 GB, a coexistência do Nemotron com
um LLM residente é arriscada (ver `asr/README.md`, seção "Por que NÃO usar Nemotron no celular
intermediário").

## 7. Referência: como está no nosso app

| arquivo | papel |
|---|---|
| `android/sherpa/build.gradle.kts` | módulo `:sherpa` (minSdk 31, `abiFilters arm64-v8a`, `.so` pré-compilados em `jniLibs/`) |
| `android/sherpa/src/main/java/br/org/ceia/cemigpoc/sherpa/NemotronAsrEngine.kt` | config + `transcribe` (seção 4) |
| `android/app/src/main/java/br/org/ceia/cemigpoc/data/engine/RealNemotronAsrEngine.kt` | push-to-talk: `recordingId` monotônico, `SharedFlow(replay=0)`, mínimo de 300 ms, `Final` não dispara o pipeline sozinho |
| `android/whisper/src/main/java/br/org/ceia/cemigpoc/whisper/AudioRecordRecorder.kt` | gravação 16 kHz mono PCM16 → float |
| `android/app/src/main/java/br/org/ceia/cemigpoc/data/model/ModelDownloader.kt` | download dos 4 arquivos com retomada + SHA-256 |
| `android/app/src/main/java/br/org/ceia/cemigpoc/data/model/ModelFileManager.kt` | `NEMOTRON_DIR_NAME`, override externo × `filesDir` |
| `android/app/build.gradle.kts` | `-Pcemig.asrEngine=nemotron` (default) → `BuildConfig.USE_NEMOTRON_ASR` |

Uma armadilha de push-to-talk que já nos custou um bug no S24+: se o canal de eventos de ASR tiver
*replay* (`MutableSharedFlow(replay = 1)`), o resultado da gravação anterior é reentregue quando a
tela volta a coletar, e o app responde à pergunta errada. Use `replay = 0` e um id por gravação
(`android/README_ASR_FIX.md`).

## 8. Troubleshooting

| sintoma | causa provável | o que fazer |
|---|---|---|
| `IllegalArgumentException: Invalid OnlineRecognizerConfig: failed to create native OnlineRecognizer` | arquivo faltando ou caminho errado. O logcat (tag `sherpa-onnx`) mostra antes `transducer encoder: '…' does not exist`, `joiner: '…' does not exist`, `tokens: '…' does not exist…` ou `Errors found in config!` | confira os 4 arquivos (tamanho exato da seção 2) e passe **caminhos absolutos** com `assetManager = null`. Caminho relativo só funciona com `AssetManager`, e os arquivos não cabem em assets |
| "falha silenciosa no load" (nosso app cai no Whisper ou avisa voz indisponível) | o nosso `initModel` captura a exceção acima e devolve `false`. Procure `Falha ao carregar Nemotron` / `Arquivos do Nemotron ausentes` no logcat | idem. Se o diretório foi criado por `adb push`, rode `adb shell chmod -R 775 <dir>`: sem isso o app (uid `u0_a*`) não lê o diretório |
| `UnsatisfiedLinkError: dlopen failed: library "libsherpa-onnx-jni.so" not found` (ou `libonnxruntime.so`) | `.so` fora de `jniLibs/arm64-v8a/`, ABI errada (aparelho/emulador não arm64) ou AAR não incluído | coloque os dois `.so` juntos na ABI certa e confira com `unzip -l app.apk \| grep sherpa` |
| `UnsatisfiedLinkError: No implementation found for ...` / `Failed to get field ID for ...` | bindings Kotlin e `.so` de versões diferentes | use bindings e `.so` do mesmo AAR ou do mesmo commit |
| app fecha sem exceção Java ao carregar | `decodingMethod` diferente de `greedy_search` (`Unsupported decoding method` + `_Exit(-1)` no nativo) | use `"greedy_search"` |
| transcrição **vazia** com áudio audível | escala errada (PCM16 sem `/32768`) ou taxa declarada ≠ taxa real | converta para float [-1, 1] e passe a taxa verdadeira em `acceptWaveform` |
| lixo / palavras truncadas | taxa real menor que a declarada (8 kHz declarado como 16 kHz), estéreo tratado como mono | grave 16 kHz mono ou declare a taxa real |
| sai em inglês (ou em outro idioma), some o artigo inicial, números por extenso | idioma não fixado: o modelo está em auto-detecção. **Inclui quem usou `setOption("lang", …)`**, que é ignorado | `stream.setOption("language", "pt")` em **cada** stream, antes do `acceptWaveform`. Nos nossos 50 áudios PT-BR o auto nunca produziu inglês, mas trocou palavras e omitiu artigos (seção 5, item 2) |
| fala antes de uma pausa some do resultado | endpoint ligado + `reset()` no `isEndpoint()` (padrão dos exemplos de streaming) | `enableEndpoint = false` em push-to-talk |
| só o começo da frase sai | áudio cortado antes de chegar ao modelo (gravação parada cedo, buffer não drenado) | junte a thread de gravação antes de ler o buffer; mantenha a cauda de 0,5 s |
| ANR / tela congela | load ou transcrição na main thread | rode em thread de fundo (`Dispatchers.IO` ou executor) |
| `Cannot find vocab_size` | ONNX de outra exportação (ex.: `onnx-community` INT4) | use o repositório `csukuangfj2/...-560ms-int8-2026-06-11` |
| app é morto em segundo plano / pouca RAM | 792 MB de pico do ASR + LLM residente | aparelho ≥ 8 GB, ou carregue o ASR sob demanda e libere com `release()` |

## 9. Teste rápido de aceitação

1. Grave ~3 s dizendo, por exemplo, "Posso realizar intervenção em circuito energizado durante chuva
   forte?".
2. O texto sai **não vazio, em português**, em ~1 s no S24+.
3. Para um teste sem microfone use um WAV mono 16 kHz com
   `WaveReader.readWave(path)` → `transcribe(samples, sampleRate)`. Há um de exemplo em
   `docs/asr_teste_ptbr.wav` no repositório HF
   [`juniorrios/automobil-ultra-semnexo`](https://huggingface.co/juniorrios/automobil-ultra-semnexo/tree/main/docs)
   (voz sintética edge-tts pt-BR do conjunto `asr/data/clean/phrase_08.wav`, mono 16 kHz, 5,1 s,
   162.894 bytes, SHA-256 `a6b4fa9b7353422e6d671fd4407fdfd29a8b374d57607a1f983076d43eafedb4`).
   Saída esperada, obtida no S24+: "Posso realizar intervenção em circuito energizado durante chuva
   forte" (a pontuação pode variar).
