# AGENT_GUIDE_ASR — Integrar ASR Nemotron 3.5 (sherpa-onnx) num app Android

<!-- Este arquivo é para um AGENTE DE CÓDIGO executar. Versão humana: GUIA_ASR_NEMOTRON.md -->

## 0. Contrato

- **Objetivo:** o app Android do usuário transcreve fala em português, offline, com o modelo
  `Nemotron 3.5 ASR streaming 0.6B INT8 chunk 560 ms` via sherpa-onnx, em modo push-to-talk
  (grava → solta → texto).
- **Entrada:** um projeto Android Gradle (Kotlin ou Java) com um módulo de aplicativo.
- **Saída:** 6 arquivos Kotlin novos, 1 dependência AAR, 2 permissões no manifest e um autoteste
  que loga `ASR_SELFTEST wav=OK ... text='<português não vazio>'`.
- **Fonte de verdade:** o código abaixo foi compilado contra `sherpa-onnx-1.13.7.aar` e executado num
  Samsung Galaxy S24+ (arm64). Resultado:
  `ASR_SELFTEST wav=OK sr=16000 n=81408 ... text='Posso realizar intervenção em circuito energizado durante chuva forte'`.
- **Regras:**
  - Copie o código **literalmente**. Só troque o nome do pacote (passo 4).
  - Não troque `"language"` por `"lang"`, nem `"greedy_search"` por outro método.
  - Não rode load nem transcrição na main thread.
  - Depois de cada passo, rode a verificação. Se ela falhar, vá à seção 9 antes de seguir.

## 1. Constantes (não altere)

```text
MODEL_REPO   = csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
MODEL_BASE   = https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/resolve/main
MODEL_DIR    = nemotron-3.5-official-560ms-int8
encoder.int8.onnx  657601403  012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68
decoder.int8.onnx   14978075  19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0
joiner.int8.onnx     9504438  4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce
tokens.txt            131440  729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a
TOTAL              682215356
AAR_URL      = https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-1.13.7.aar
AAR_SIZE     = 49113869
AAR_SHA256   = c4ef49e309f24fcee5c106b8a279481aaecaabb078cd37b2cd6e9a62cc8a73c8
TEST_WAV_URL = https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/docs/asr_teste_ptbr.wav
TEST_WAV_SHA256 = a6b4fa9b7353422e6d671fd4407fdfd29a8b374d57607a1f983076d43eafedb4
TEST_WAV_TEXT   = Posso realizar intervenção em circuito energizado durante chuva forte
```

Parâmetros do recognizer (os mesmos do app de referência `NemotronAsrEngine.kt`):
`sampleRate=16000`, `featureDim=80`, `numThreads=4`, `provider="cpu"`, `modelType="nemotron"`,
`enableEndpoint=false`, `decodingMethod="greedy_search"`. Por stream: `setOption("language","pt")`,
áudio `FloatArray` em [-1, 1], mono, com a taxa real passada a `acceptWaveform`, mais uma cauda de
0,5 s de zeros e depois `inputFinished()`.

## 2. Pré-requisitos (verifique todos antes de editar)

Rode na raiz do projeto do usuário:

```bash
# P1. Projeto Gradle Android com wrapper
test -x ./gradlew && echo P1_OK
# P2. Identificar o módulo de aplicativo (anote o caminho; abaixo chamado APP_MODULE, ex.: app)
grep -rl --include=build.gradle* --include=build.gradle.kts -E 'com\.android\.application' . | grep -v '/build/'
# P3. Kotlin disponível no módulo (plugin kotlin-android / org.jetbrains.kotlin.android)
grep -E 'kotlin' APP_MODULE/build.gradle* || echo "P3_FALHOU: adicione o plugin Kotlin Android"
# P4. minSdk >= 21 (AAR declara minSdk 21; o app de referência usa 31)
grep -E 'minSdk' APP_MODULE/build.gradle*
# P5. namespace do módulo (usado no passo 4)
grep -E 'namespace|applicationId' APP_MODULE/build.gradle*
# P6. ferramentas
command -v curl && command -v sha256sum && echo P6_OK
```

Asserts:
- `P1_OK` impresso.
- Exatamente um módulo de aplicativo identificado (se houver mais de um, use o que contém a
  `MainActivity`/launcher).
- O plugin Kotlin está presente. Se o projeto for só Java, adicione
  `id("org.jetbrains.kotlin.android")` ao módulo antes de seguir; não reescreva o código em Java.
- `minSdk` ≥ 21.
- Aparelho de teste: **físico arm64-v8a** (`adb shell getprop ro.product.cpu.abi` → `arm64-v8a`).
  Emulador x86/x86_64 não serve para aceitação.
- O aparelho precisa de ~700 MB livres para o modelo e, de preferência, ≥ 8 GB de RAM se o app também
  carregar um LLM.

## 3. Adicionar o runtime sherpa-onnx (AAR)

```bash
mkdir -p APP_MODULE/libs
curl -L --fail -o APP_MODULE/libs/sherpa-onnx-1.13.7.aar \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-1.13.7.aar
echo "c4ef49e309f24fcee5c106b8a279481aaecaabb078cd37b2cd6e9a62cc8a73c8  APP_MODULE/libs/sherpa-onnx-1.13.7.aar" | sha256sum -c -
```

Edite `APP_MODULE/build.gradle.kts` (ou `build.gradle`). No bloco `dependencies` acrescente:

```kotlin
// build.gradle.kts
implementation(files("libs/sherpa-onnx-1.13.7.aar"))
```

```groovy
// build.gradle (Groovy)
implementation files('libs/sherpa-onnx-1.13.7.aar')
```

No bloco `android { defaultConfig { ... } }` restrinja a ABI (o modelo só é suportado em arm64):

```kotlin
// build.gradle.kts
ndk { abiFilters += listOf("arm64-v8a") }
```

```groovy
// build.gradle (Groovy)
ndk { abiFilters 'arm64-v8a' }
```

Se o módulo já tiver `abiFilters`, garanta que `arm64-v8a` esteja incluído; não remova as outras ABIs
sem perguntar ao usuário.

Variante sem `libonnxruntime.so` separado (um único `.so` de 23,6 MB): use
`sherpa-onnx-static-link-onnxruntime-1.13.7.aar` (SHA-256
`220d7cb25ac6e57ec34082bc2551ebd7ec7d4d96cbfea520839e208f92347837`). Também foi testada no S24+.
Use-a **só** se o projeto já trouxer outro `libonnxruntime.so` (ex.: `com.microsoft.onnxruntime`).
Duas cópias com o mesmo nome dão `More than one file was found with OS independent path
'lib/arm64-v8a/libonnxruntime.so'`.

**Verificação 3:**

```bash
sha256sum APP_MODULE/libs/sherpa-onnx-1.13.7.aar   # == c4ef49e3...
./gradlew :APP_MODULE:assembleDebug
unzip -l APP_MODULE/build/outputs/apk/debug/*.apk | grep -E 'lib/arm64-v8a/lib(sherpa-onnx-jni|onnxruntime)\.so'
```

Assert: o build passa e as **duas** linhas aparecem (`libsherpa-onnx-jni.so` e `libonnxruntime.so`).
Sem `unzip`: `python3 -c "import zipfile,glob;print([n for n in zipfile.ZipFile(glob.glob('APP_MODULE/build/outputs/apk/debug/*.apk')[0]).namelist() if n.startswith('lib/arm64')])"`.

## 4. Criar os 6 arquivos Kotlin

Defina o pacote: `ASR_PKG = <namespace do passo P5>.asr` (ex.: namespace `com.acme.campo` →
`com.acme.campo.asr`). Crie o diretório
`APP_MODULE/src/main/java/<ASR_PKG com / no lugar de .>/` e grave os 6 arquivos abaixo
**exatamente** como estão. Depois troque o pacote em todos:

```bash
ASR_PKG=com.acme.campo.asr          # <-- ajuste
DIR=APP_MODULE/src/main/java/$(echo $ASR_PKG | tr . /)
sed -i "s/^package com\.example\.asr$/package $ASR_PKG/" $DIR/*.kt
grep -L "^package $ASR_PKG$" $DIR/*.kt   # deve imprimir NADA
```

Se o projeto usa `src/main/kotlin/`, use esse diretório no lugar de `src/main/java/`.

### 4.1 `NemotronAsr.kt` (motor: config + transcrição)

```kotlin
package com.example.asr

import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import java.io.Closeable
import java.io.File

/**
 * ASR Nemotron 3.5 streaming 0.6B INT8 (chunk 560 ms) via sherpa-onnx, modo push-to-talk:
 * grava o enunciado inteiro e transcreve no fim. Não é thread-safe: chame de UMA thread de fundo.
 */
class NemotronAsr : Closeable {

    companion object {
        private const val TAG = "NemotronAsr"
        const val SAMPLE_RATE = 16000
        val REQUIRED_FILES = listOf("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    }

    private var recognizer: OnlineRecognizer? = null

    val isLoaded: Boolean get() = recognizer != null

    /** Carrega o modelo de [modelDir]. Retorna false (e loga o motivo) se falhar. Leva segundos: não chame na main thread. */
    fun load(modelDir: File, numThreads: Int = 4): Boolean {
        close()
        val missing = REQUIRED_FILES.filterNot { File(modelDir, it).isFile }
        if (missing.isNotEmpty()) {
            Log.e(TAG, "Arquivos ausentes em ${modelDir.absolutePath}: $missing")
            return false
        }
        val config = OnlineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
            modelConfig = OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = File(modelDir, "encoder.int8.onnx").absolutePath,
                    decoder = File(modelDir, "decoder.int8.onnx").absolutePath,
                    joiner = File(modelDir, "joiner.int8.onnx").absolutePath,
                ),
                tokens = File(modelDir, "tokens.txt").absolutePath,
                numThreads = numThreads,
                provider = "cpu",
                modelType = "nemotron",
            ),
            enableEndpoint = false,
            decodingMethod = "greedy_search",
        )
        return try {
            recognizer = OnlineRecognizer(assetManager = null, config = config)
            Log.i(TAG, "Nemotron carregado de ${modelDir.absolutePath} (threads=$numThreads)")
            true
        } catch (e: Throwable) {
            Log.e(TAG, "Falha ao criar OnlineRecognizer", e)
            recognizer = null
            false
        }
    }

    /**
     * Transcreve um enunciado completo.
     * @param samples PCM mono float em [-1, 1].
     * @param sampleRate taxa REAL do áudio (16000 se veio do PcmRecorder; o sherpa reamostra se for outra).
     * @param language código de idioma do Nemotron ("pt", "pt-BR", "en", "auto"...). Default "pt".
     */
    fun transcribe(samples: FloatArray, sampleRate: Int = SAMPLE_RATE, language: String = "pt"): String {
        val rec = recognizer ?: error("NemotronAsr.load() não foi chamado ou falhou")
        if (samples.isEmpty()) return ""
        val stream = rec.createStream()
        try {
            // A chave lida pelo sherpa-onnx é "language" (NÃO "lang": é ignorada sem erro).
            stream.setOption("language", language)
            stream.acceptWaveform(samples, sampleRate)
            // Cauda de 0,5 s de silêncio para drenar o último bloco de 560 ms.
            stream.acceptWaveform(FloatArray(sampleRate / 2), sampleRate)
            stream.inputFinished()
            while (rec.isReady(stream)) {
                rec.decode(stream)
            }
            return rec.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override fun close() {
        recognizer?.release()
        recognizer = null
    }
}
```

### 4.2 `PcmRecorder.kt` (gravação 16 kHz mono → float [-1,1])

```kotlin
package com.example.asr

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Grava 16 kHz, mono, PCM 16-bit e devolve FloatArray em [-1, 1] — o formato que o NemotronAsr espera. */
class PcmRecorder {

    private var record: AudioRecord? = null
    private var thread: Thread? = null
    private val buffer = ByteArrayOutputStream()

    @Volatile
    var isRecording = false
        private set

    /** Exige RECORD_AUDIO concedida ANTES de chamar. */
    @SuppressLint("MissingPermission")
    fun start() {
        if (isRecording) return
        val minBuf = AudioRecord.getMinBufferSize(
            NemotronAsr.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            NemotronAsr.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, NemotronAsr.SAMPLE_RATE * 2)
        )
        check(rec.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord não inicializou (permissão RECORD_AUDIO?)" }
        synchronized(buffer) { buffer.reset() }
        record = rec
        isRecording = true
        rec.startRecording()
        thread = Thread {
            val tmp = ByteArray(4096)
            while (isRecording) {
                val n = rec.read(tmp, 0, tmp.size)
                if (n > 0) synchronized(buffer) { buffer.write(tmp, 0, n) }
            }
        }.also { it.start() }
    }

    /** Para a gravação e devolve as amostras float [-1, 1] a 16 kHz. */
    fun stop(): FloatArray {
        if (!isRecording) return FloatArray(0)
        isRecording = false
        thread?.join(1000)
        thread = null
        record?.run { stop(); release() }
        record = null
        val bytes = synchronized(buffer) { buffer.toByteArray().also { buffer.reset() } }
        val shorts = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(shorts.remaining()) { shorts.get() / 32768.0f }
    }
}
```

### 4.3 `NemotronModelInstaller.kt` (download do modelo com tamanho + SHA-256)

```kotlin
package com.example.asr

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** Baixa os 4 arquivos do Nemotron 560ms INT8 do Hugging Face (público) e confere tamanho + SHA-256. */
object NemotronModelInstaller {

    const val DIR_NAME = "nemotron-3.5-official-560ms-int8"
    private const val BASE =
        "https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/resolve/main"

    data class ModelFile(val name: String, val size: Long, val sha256: String)

    val FILES = listOf(
        ModelFile("encoder.int8.onnx", 657_601_403L, "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68"),
        ModelFile("decoder.int8.onnx", 14_978_075L, "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0"),
        ModelFile("joiner.int8.onnx", 9_504_438L, "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce"),
        ModelFile("tokens.txt", 131_440L, "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a"),
    )

    /** Diretório final: override de `adb push` em getExternalFilesDir se completo; senão filesDir. */
    fun modelDir(context: Context): File {
        val ext = context.getExternalFilesDir(null)?.resolve(DIR_NAME)
        if (ext != null && FILES.all { File(ext, it.name).length() == it.size }) return ext
        return File(context.filesDir, DIR_NAME)
    }

    fun isInstalled(context: Context): Boolean {
        val dir = modelDir(context)
        return FILES.all { File(dir, it.name).length() == it.size }
    }

    /** BLOQUEANTE (rede + ~682 MB). Chame numa thread de fundo. [onProgress] recebe (bytes baixados, total). */
    @Throws(IOException::class)
    fun install(context: Context, onProgress: (Long, Long) -> Unit = { _, _ -> }) {
        val dir = File(context.filesDir, DIR_NAME).apply { mkdirs() }
        val total = FILES.sumOf { it.size }
        var done = 0L
        for (f in FILES) {
            val dest = File(dir, f.name)
            if (dest.length() == f.size) { done += f.size; onProgress(done, total); continue }
            val part = File(dir, f.name + ".part")
            val conn = URL("$BASE/${f.name}").openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = true
            conn.connectTimeout = 20_000
            conn.readTimeout = 60_000
            try {
                if (conn.responseCode != HttpURLConnection.HTTP_OK) throw IOException("HTTP ${conn.responseCode} em ${f.name}")
                val md = MessageDigest.getInstance("SHA-256")
                conn.inputStream.use { input ->
                    FileOutputStream(part).use { out ->
                        val buf = ByteArray(256 * 1024)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            md.update(buf, 0, n)
                            done += n
                            onProgress(done, total)
                        }
                    }
                }
                val hash = md.digest().joinToString("") { "%02x".format(it) }
                if (part.length() != f.size || hash != f.sha256) {
                    part.delete()
                    throw IOException("${f.name} corrompido: ${part.length()} bytes, sha256 $hash")
                }
                if (!part.renameTo(dest)) throw IOException("Não foi possível gravar ${dest.absolutePath}")
            } finally {
                conn.disconnect()
            }
        }
    }
}
```

### 4.4 `AsrSelfTest.kt` (autoteste com WAV, linha ASR_SELFTEST)

```kotlin
package com.example.asr

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.WaveReader
import java.util.Locale

/**
 * Autoteste de aceitação. Loga UMA linha com prefixo "ASR_SELFTEST" (filtre no logcat):
 *   ASR_SELFTEST wav=OK text='...'      (ou wav=SKIP se o WAV não existir)
 * O WAV deve ser mono; a taxa é lida do arquivo.
 */
object AsrSelfTest {
    private const val TAG = "ASR_SELFTEST"

    fun runWav(context: Context, asr: NemotronAsr) {
        val wav = context.getExternalFilesDir(null)?.resolve("asr_teste_ptbr.wav")
        if (wav == null || !wav.isFile) {
            Log.i(TAG, "ASR_SELFTEST wav=SKIP (coloque asr_teste_ptbr.wav em ${context.getExternalFilesDir(null)})")
            return
        }
        val t0 = System.currentTimeMillis()
        val data = WaveReader.readWave(wav.absolutePath)
        val text = asr.transcribe(data.samples, data.sampleRate, "pt")
        val ms = System.currentTimeMillis() - t0
        Log.i(TAG, "ASR_SELFTEST wav=OK sr=${data.sampleRate} n=${data.samples.size} ms=$ms text='$text'")
    }

    fun logMic(samples: FloatArray, text: String, ms: Long) {
        val seconds = String.format(Locale.US, "%.2f", samples.size / NemotronAsr.SAMPLE_RATE.toFloat())
        Log.i(TAG, "ASR_SELFTEST mic=OK seconds=$seconds ms=$ms text='$text'")
    }
}
```

### 4.5 `AsrController.kt` (orquestração push-to-talk em thread de fundo)

```kotlin
package com.example.asr

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executors

/**
 * Orquestra push-to-talk: load (uma vez) → startRecording → stopAndTranscribe → callback na main thread.
 * Todo trabalho pesado roda num único executor de fundo (o recognizer não é thread-safe).
 */
class AsrController(private val context: Context) {

    private val asr = NemotronAsr()
    private val recorder = PcmRecorder()
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    /** Baixa (se preciso) e carrega o modelo. onReady(true) quando pronto. */
    fun init(onProgress: (Long, Long) -> Unit = { _, _ -> }, onReady: (Boolean, String?) -> Unit) {
        worker.execute {
            try {
                if (!NemotronModelInstaller.isInstalled(context)) {
                    NemotronModelInstaller.install(context) { d, t -> main.post { onProgress(d, t) } }
                }
                val ok = asr.load(NemotronModelInstaller.modelDir(context))
                if (ok) AsrSelfTest.runWav(context, asr)
                main.post { onReady(ok, if (ok) null else "Falha ao carregar o Nemotron (ver logcat NemotronAsr)") }
            } catch (e: Exception) {
                Log.e("AsrController", "init falhou", e)
                main.post { onReady(false, e.message) }
            }
        }
    }

    /** Exige RECORD_AUDIO concedida ANTES de chamar. */
    fun startRecording() = recorder.start()

    /** Para a gravação e transcreve. onResult recebe o texto (vazio = nada reconhecido). */
    fun stopAndTranscribe(onResult: (String) -> Unit) {
        val samples = recorder.stop()
        worker.execute {
            val t0 = System.currentTimeMillis()
            val text = if (samples.size < NemotronAsr.SAMPLE_RATE * 3 / 10) "" else asr.transcribe(samples)
            AsrSelfTest.logMic(samples, text, System.currentTimeMillis() - t0)
            main.post { onResult(text) }
        }
    }

    fun release() {
        worker.execute { asr.close() }
        worker.shutdown()
    }
}
```

### 4.6 `AsrAcceptanceActivity.kt` (tela de aceitação: carrega, testa o WAV e grava 3 s (temporária))

```kotlin
package com.example.asr

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.TextView

/**
 * Tela de aceitação (SÓ PARA TESTE; remova do manifest antes do release).
 * Ao abrir: baixa/carrega o modelo, roda o autoteste com WAV (se existir) e grava 3 s do microfone.
 * Resultado no logcat, tag ASR_SELFTEST, e na própria tela.
 */
class AsrAcceptanceActivity : Activity() {

    private lateinit var controller: AsrController
    private lateinit var view: TextView
    private val main = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        view = TextView(this).apply { textSize = 18f; setPadding(32, 32, 32, 32) }
        setContentView(view)
        controller = AsrController(applicationContext)
        show("Carregando Nemotron...")
        controller.init(
            onProgress = { d, t -> show("Baixando modelo: ${d * 100 / t}%") },
            onReady = { ok, err ->
                if (!ok) {
                    show("ERRO: $err")
                    Log.e("ASR_SELFTEST", "ASR_SELFTEST load=FAIL $err")
                } else {
                    Log.i("ASR_SELFTEST", "ASR_SELFTEST load=OK")
                    recordThreeSeconds()
                }
            }
        )
    }

    private fun recordThreeSeconds() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
            return
        }
        show("FALE AGORA (3 s)...")
        controller.startRecording()
        main.postDelayed({
            show("Transcrevendo...")
            controller.stopAndTranscribe { text -> show("Transcrição: '$text'") }
        }, 3000)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) recordThreeSeconds()
        else show("ERRO: permissão de microfone negada")
    }

    private fun show(msg: String) = main.post { view.text = msg }

    override fun onDestroy() {
        controller.release()
        super.onDestroy()
    }
}
```

**Verificação 4:**

```bash
ls $DIR | sort   # AsrAcceptanceActivity.kt AsrController.kt AsrSelfTest.kt NemotronAsr.kt NemotronModelInstaller.kt PcmRecorder.kt
grep -n 'setOption("language", language)' $DIR/NemotronAsr.kt   # 1 linha
grep -n 'setOption("lang"' $DIR/*.kt                              # NADA
grep -n 'modelType = "nemotron"' $DIR/NemotronAsr.kt              # 1 linha
grep -n 'decodingMethod = "greedy_search"' $DIR/NemotronAsr.kt    # 1 linha
grep -n 'enableEndpoint = false' $DIR/NemotronAsr.kt              # 1 linha
./gradlew :APP_MODULE:compileDebugKotlin
```

Assert: as 6 entradas existem, cada grep tem o resultado indicado e a compilação passa.

## 5. Manifest

Em `APP_MODULE/src/main/AndroidManifest.xml`:

1. Dentro de `<manifest>`, se ainda não existirem, acrescente:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
```

`INTERNET` só é necessária para o download do modelo (passo 6, opção A). Se o modelo for sempre
instalado por `adb push` (opção B), ela pode ficar de fora.

2. Dentro de `<application>`, acrescente a tela de aceitação (temporária; ver passo 8):

```xml
<activity
    android:name="ASR_PKG.AsrAcceptanceActivity"
    android:exported="true" />
```

(troque `ASR_PKG` pelo pacote real, ex.: `com.acme.campo.asr.AsrAcceptanceActivity`).

**Verificação 5:**

```bash
grep -c 'android.permission.RECORD_AUDIO' APP_MODULE/src/main/AndroidManifest.xml   # >= 1
grep -n 'AsrAcceptanceActivity' APP_MODULE/src/main/AndroidManifest.xml             # 1 linha
./gradlew :APP_MODULE:assembleDebug
```

## 6. Colocar o modelo no aparelho

O código procura primeiro em `getExternalFilesDir(null)/nemotron-3.5-official-560ms-int8/` (se os 4
arquivos estiverem com o tamanho exato) e depois em `filesDir/nemotron-3.5-official-560ms-int8/`.

**Opção A, automática (padrão):** com internet no aparelho, o `AsrController.init()` chama o
`NemotronModelInstaller.install()`, que baixa os 682 MB e confere o SHA-256. Não é preciso fazer nada.

**Opção B, por adb (sem internet no aparelho; recomendada para o 1º teste):**

```bash
APP_ID=<applicationId do passo P5>
pip install -U huggingface_hub
hf download csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt --local-dir nemotron-3.5-official-560ms-int8
(cd nemotron-3.5-official-560ms-int8 && sha256sum encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt)
curl -L --fail -o asr_teste_ptbr.wav https://huggingface.co/juniorrios/automobil-ultra-semnexo/resolve/main/docs/asr_teste_ptbr.wav
echo "a6b4fa9b7353422e6d671fd4407fdfd29a8b374d57607a1f983076d43eafedb4  asr_teste_ptbr.wav" | sha256sum -c -

./gradlew :APP_MODULE:installDebug
EXT=/sdcard/Android/data/$APP_ID/files
adb shell am start -n $APP_ID/ASR_PKG.AsrAcceptanceActivity; sleep 3; adb shell am force-stop $APP_ID   # getExternalFilesDir() cria $EXT (troque ASR_PKG)
adb shell mkdir -p $EXT/nemotron-3.5-official-560ms-int8
adb push nemotron-3.5-official-560ms-int8/encoder.int8.onnx nemotron-3.5-official-560ms-int8/decoder.int8.onnx \
         nemotron-3.5-official-560ms-int8/joiner.int8.onnx nemotron-3.5-official-560ms-int8/tokens.txt \
         $EXT/nemotron-3.5-official-560ms-int8/
adb push asr_teste_ptbr.wav $EXT/
adb shell chmod -R 775 $EXT/nemotron-3.5-official-560ms-int8
```

Em build debug, se o `applicationId` tiver sufixo (ex.: `.debug`), use o id com sufixo em `APP_ID`.

**Verificação 6:**

```bash
adb shell ls -l $EXT/nemotron-3.5-official-560ms-int8
```

Assert: 4 arquivos com exatamente `657601403`, `14978075`, `9504438` e `131440` bytes (os saídos de
`sha256sum` no PC batem com a seção 1), e `asr_teste_ptbr.wav` presente em `$EXT`.

## 7. Aceitação (obrigatória)

```bash
adb shell pm grant $APP_ID android.permission.RECORD_AUDIO
adb shell input keyevent KEYCODE_WAKEUP          # tela ligada e DESBLOQUEADA: com a tela bloqueada o processo é congelado
adb logcat -c
adb shell am start -n $APP_ID/ASR_PKG.AsrAcceptanceActivity
# Aguarde até 120 s (mais se a opção A estiver baixando 682 MB). Assim que aparecer "FALE AGORA",
# uma PESSOA fala uma frase em português por 3 s perto do microfone.
adb logcat -d -s ASR_SELFTEST:* NemotronAsr:* AsrController:* sherpa-onnx:* AndroidRuntime:E
```

Saída esperada (formato):

```text
I NemotronAsr: Nemotron carregado de /storage/emulated/0/Android/data/<APP_ID>/files/nemotron-3.5-official-560ms-int8 (threads=4)
I ASR_SELFTEST: ASR_SELFTEST wav=OK sr=16000 n=81408 ms=<n> text='Posso realizar intervenção em circuito energizado durante chuva forte'
I ASR_SELFTEST: ASR_SELFTEST load=OK
I ASR_SELFTEST: ASR_SELFTEST mic=OK seconds=<~2.9> ms=<n> text='<o que a pessoa falou>'
```

Checklist (todos obrigatórios):

- [ ] **A1** `ASR_SELFTEST load=OK` presente; nenhuma linha `load=FAIL`, `FATAL EXCEPTION` ou
      `UnsatisfiedLinkError`.
- [ ] **A2** Linha `wav=OK` com `text=` **não vazio** e igual (ignorando pontuação e maiúsculas) a
      `Posso realizar intervenção em circuito energizado durante chuva forte`.
- [ ] **A3** O texto está em **português**: contém palavras como `intervenção`, `circuito` e
      `energizado`, e não é tradução em inglês.
- [ ] **A4** Linha `mic=OK` com `seconds` entre 2,5 e 3,5 (houve gravação de ~3 s).
- [ ] **A5** Com uma pessoa falando português durante a gravação, `mic=OK ... text=` é **não vazio** e
      em português. Se nenhum humano estiver disponível, marque A5 como **pendente-humano** e informe
      ao usuário. Não invente o resultado: `text=''` com o ambiente em silêncio é o comportamento
      correto, não falha.
- [ ] **A6** O app não congela durante o load (sem ANR no logcat: `grep -i "ANR in"` não acha nada).

Asserts automatizáveis de A1–A3:

```bash
L=$(adb logcat -d -s ASR_SELFTEST:*)
echo "$L" | grep -q 'ASR_SELFTEST load=OK' && echo A1_OK
echo "$L" | grep -q "wav=OK.*text='Posso realizar intervenção em circuito energizado durante chuva forte" && echo A2_A3_OK
echo "$L" | grep -E "mic=OK seconds=(2\.[5-9]|3\.[0-4])" >/dev/null && echo A4_OK
```

Tempo: **não** use este teste para medir latência (build debug, primeira execução do processo). No
app de referência (release, primeiro plano, S24+) a transcrição de uma pergunta leva ~1–3 s.

## 8. Integrar no fluxo real do app e limpar

1. Na tela que tem o botão de falar, use o `AsrController`:
   - criação da tela: `controller = AsrController(applicationContext)` e
     `controller.init(onProgress = {...}, onReady = { ok, err -> ... })` uma vez (habilite o botão só
     quando `ok == true`);
   - ao **pressionar** o botão: se a permissão `RECORD_AUDIO` estiver concedida,
     `controller.startRecording()`; senão, peça a permissão com `requestPermissions`;
   - ao **soltar**: `controller.stopAndTranscribe { text -> /* texto vazio = nada reconhecido */ }`;
   - na destruição da tela: `controller.release()`.
2. Se o app tiver mais de uma gravação em voo (ex.: o usuário aperta de novo antes do resultado),
   descarte resultados de gravações antigas com um id por gravação. **Não** use
   `MutableSharedFlow(replay = 1)` para entregar o texto: o replay reentrega a transcrição anterior ao
   recoletar.
3. Remova a tela de aceitação antes do release: apague `AsrAcceptanceActivity.kt` e a
   `<activity ... AsrAcceptanceActivity>` do manifest. Mantenha o `AsrSelfTest.kt` (inofensivo: só
   roda se o WAV existir) ou remova também a chamada `AsrSelfTest.runWav` do `AsrController`.
4. Não adicione os `.onnx` aos `assets/`. O encoder tem 657 MB, e o `OnlineRecognizer` deste guia
   recebe caminhos absolutos com `assetManager = null`.

**Verificação 8:** `./gradlew :APP_MODULE:assembleRelease` (ou `assembleDebug`) passa; refaça A1 e A5
pelo fluxo real (botão).

## 9. Erros conhecidos: assinatura → causa → correção

| assinatura exata (logcat / build) | causa | correção |
|---|---|---|
| `java.lang.IllegalArgumentException: Invalid OnlineRecognizerConfig: failed to create native OnlineRecognizer` | arquivo do modelo ausente ou caminho errado | ver as linhas `sherpa-onnx` logo antes: `transducer encoder: '<path>' does not exist` / `transducer decoder: ...` / `joiner: '<path>' does not exist` / `tokens: '<path>' does not exist, you should provide either a tokens buffer or a tokens file` / `Errors found in config!`. Refaça o passo 6 e a verificação 6 |
| `NemotronAsr: Arquivos ausentes em <dir>: [...]` | idem (checagem do próprio `NemotronAsr`) | refaça o passo 6 |
| `ASR_SELFTEST load=FAIL Unable to resolve host "huggingface.co"` | opção A sem internet | use a opção B (adb) ou dê internet ao aparelho |
| `ASR_SELFTEST load=FAIL HTTP <código> em <arquivo>` | download bloqueado (proxy, 403/404) | confira `MODEL_BASE`; use a opção B |
| `ASR_SELFTEST load=FAIL <arquivo> corrompido: <n> bytes, sha256 <h>` | download truncado/alterado | rode de novo (o `.part` é refeito); persistindo, opção B |
| `java.lang.UnsatisfiedLinkError: dlopen failed: library "libsherpa-onnx-jni.so" not found` | AAR não entrou no APK ou ABI do aparelho não é arm64 | verificação 3; `adb shell getprop ro.product.cpu.abi` deve ser `arm64-v8a` |
| `java.lang.UnsatisfiedLinkError: dlopen failed: library "libonnxruntime.so" not found` | `libsherpa-onnx-jni.so` copiado sem o `libonnxruntime.so` | use o AAR inteiro ou a variante `static-link-onnxruntime` |
| `More than one file was found with OS independent path 'lib/arm64-v8a/libonnxruntime.so'` | o projeto já tem outro ONNX Runtime | use `sherpa-onnx-static-link-onnxruntime-1.13.7.aar` (seção 3) |
| `No implementation found for ... com.k2fsa.sherpa.onnx...` ou `Failed to get field ID for ...` | bindings Kotlin de uma versão e `.so` de outra | use bindings e `.so` do mesmo AAR; não copie `.kt` do sherpa de outra versão |
| `Unresolved reference: OnlineRecognizer` / `k2fsa` | AAR não adicionado às dependências | passo 3 |
| `Unsupported decoding method: <x>` e o processo some sem stack Java | `decodingMethod` ≠ `greedy_search` (o nativo chama `_Exit(-1)`) | `decodingMethod = "greedy_search"` |
| `Cannot find vocab_size` | ONNX de outra exportação (ex.: `onnx-community/...-int4`) | use os arquivos de `MODEL_REPO` |
| `wav=OK ... text=''` | escala errada (PCM16 sem `/32768`) ou taxa declarada ≠ taxa real | não altere `PcmRecorder`/`WaveReader`; passe a taxa verdadeira em `transcribe(samples, sampleRate)` |
| texto truncado / palavras deformadas ("Squedimentos…") | áudio de 8 kHz declarado como 16 kHz, ou estéreo tratado como mono | grave 16 kHz mono (o `PcmRecorder` já faz) ou passe a taxa real |
| texto em outro idioma, sem o artigo inicial, números por extenso | idioma não fixado: auto-detecção. Inclui `setOption("lang", ...)`, ignorado pelo sherpa | `stream.setOption("language", "pt")` (já no `NemotronAsr.transcribe`) |
| `AudioRecord não inicializou (permissão RECORD_AUDIO?)` | permissão não concedida ou microfone em uso por outro app | conceda a permissão antes de `startRecording()`; feche outros apps de áudio |
| `ANR in <app>` durante o load | `NemotronAsr.load` chamado na main thread | chame só via `AsrController.init` (executor de fundo) |
| nenhuma linha `ASR_SELFTEST` e `FreecessController ... <app>` no logcat | tela desligada ou bloqueada: o processo foi congelado | ligue e desbloqueie a tela e repita o passo 7 |
| `ls: .../files/nemotron-...: Permission denied` ou o app não acha os arquivos que o adb gravou | diretório criado pelo shell sem permissão de leitura para o app | `adb shell chmod -R 775 $EXT/nemotron-3.5-official-560ms-int8` |

## 10. Não faça

- Não use `setOption("lang", "pt")`: a chave lida pelo sherpa-onnx é `"language"`.
- Não ligue `enableEndpoint` em push-to-talk.
- Não mude `numThreads` acima de 4 sem medir (o app de referência usa 4).
- Não carregue o recognizer mais de uma vez por processo, e não use a mesma instância de duas threads.
- Não use o 1º resultado de aceitação como benchmark de latência.
- Não declare a integração concluída sem A1–A4 marcados e A5 marcado ou explicitamente
  **pendente-humano**.
