package br.org.ceia.cemigpoc.whisper

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * Motor Whisper ASR offline executando nativamente em ARM64 com suporte a prompt de domínio CEMIG.
 *
 * Configuração:
 * - Idioma: Português ("pt")
 * - Threads: 6 (núcleos de alta performance do Exynos 2400)
 * - Prompt de domínio: Vocabulário técnico de NRs, grandezas elétricas e manobras
 */
class WhisperCppEngine(
    private val bridge: WhisperBridge = WhisperBridge(),
    val domainPrompt: String = DEFAULT_DOMAIN_PROMPT
) : Closeable {

    companion object {
        private const val TAG = "WhisperCppEngine"

        // Vocabulário de domínio calibrado na trilha ASR (eleva acurácia de termos técnicos para ~90%)
        const val DEFAULT_DOMAIN_PROMPT =
            "NR-10, NR-35, NR-06, NR-12, NR-18, 13,8 kV, 1000 V, desenergização, religador, " +
            "chave seccionadora, chave fusível, LOTO, bloqueio e etiquetagem, aterramento temporário, " +
            "linha viva, bastão de manobra, EPI, EPC, tensão de segurança."

        const val DEFAULT_THREADS = 4
    }

    private var nativeHandle: Long = 0L
    private val recorder = AudioRecordRecorder()

    val isLoaded: Boolean
        get() = nativeHandle != 0L

    val isRecording: Boolean
        get() = recorder.recording

    /**
     * Inicializa o modelo Whisper a partir do caminho do arquivo binário (ex: ggml-base-q5_1.bin).
     */
    suspend fun initModel(modelPath: String): Boolean = withContext(Dispatchers.IO) {
        if (isLoaded) {
            close()
        }

        val handle = bridge.nativeInit(modelPath)
        if (handle == 0L) {
            Log.e(TAG, "Falha ao carregar modelo Whisper em $modelPath")
            return@withContext false
        }

        nativeHandle = handle
        Log.i(TAG, "Whisper carregado com sucesso (handle=$nativeHandle)")
        true
    }

    /**
     * Inicia a gravação de áudio do microfone do dispositivo (16 kHz mono).
     */
    fun startRecording() {
        recorder.start()
    }

    /**
     * Encerra a gravação e retorna os dados de áudio prontos para transcrição.
     */
    fun stopRecording(): FloatArray {
        return recorder.stop()
    }

    /**
     * Transcreve um array de amostras de áudio float normalizadas [-1.0f, 1.0f].
     */
    suspend fun transcribe(
        audioData: FloatArray,
        language: String = "pt",
        threads: Int = DEFAULT_THREADS
    ): String = withContext(Dispatchers.IO) {
        check(isLoaded) { "Whisper não foi inicializado" }

        if (audioData.isEmpty()) {
            return@withContext ""
        }

        val rawText = bridge.nativeTranscribe(
            handle = nativeHandle,
            audioData = audioData,
            numThreads = threads,
            language = language,
            initialPrompt = domainPrompt
        )

        cleanTranscription(rawText)
    }

    /**
     * Grava áudio e executa a transcrição completa.
     */
    suspend fun transcribeRecordedAudio(
        language: String = "pt",
        threads: Int = DEFAULT_THREADS
    ): String {
        val samples = stopRecording()
        if (samples.size < 4800) { // menos de 300 ms de áudio
            Log.w(TAG, "Áudio muito curto para transcrição (${samples.size} amostras)")
            return ""
        }
        return transcribe(samples, language, threads)
    }

    /**
     * Remove marcadores especiais e espaçamentos excedentes do Whisper.
     */
    private fun cleanTranscription(rawText: String): String {
        return rawText
            .replace(Regex("\\[_TT_\\d+\\]"), "")
            .replace(Regex("<\\|.*?\\|>"), "")
            .trim()
    }

    override fun close() {
        if (isLoaded) {
            val h = nativeHandle
            nativeHandle = 0L
            bridge.nativeFree(h)
            Log.i(TAG, "WhisperCppEngine liberado com sucesso")
        }
    }
}
