package br.org.ceia.cemigpoc.sherpa

import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File

/**
 * Motor ASR Nemotron 3.5 streaming (0.6B INT8) via sherpa-onnx (ONNX Runtime), como alternativa
 * ao Whisper Base. Motivo (ordem do capitão): o Whisper não captura termo técnico ("disjuntor"
 * não sai); o Nemotron acertou 91,9% dos termos na trilha ASR (asr/README.md) com WER 6,8/8,2%.
 *
 * É um transducer STREAMING de chunk 560 ms: alimentamos as amostras acumuladas do push-to-talk
 * de uma vez (o app grava e transcreve no fim, igual ao Whisper), decodificando em blocos até o
 * fim do áudio. O modelo é multilíngue com idioma por-stream — aqui fixamos "pt".
 *
 * Pesos carregados do sistema de arquivos (encoder/decoder/joiner.int8.onnx + tokens.txt),
 * copiados pelo ModelFileManager. Comentários PT-BR; código em inglês.
 */
class NemotronAsrEngine : Closeable {

    companion object {
        private const val TAG = "NemotronAsrEngine"
        private const val SAMPLE_RATE = 16000
        // Idioma por-stream do Nemotron 3.5 (README do modelo: "en", "ja", "auto", ...).
        private const val LANGUAGE = "pt"
    }

    private var recognizer: OnlineRecognizer? = null

    val isLoaded: Boolean
        get() = recognizer != null

    /**
     * Inicializa o recognizer a partir do diretório com encoder/decoder/joiner + tokens.
     * @param modelDir diretório contendo encoder.int8.onnx, decoder.int8.onnx,
     *                 joiner.int8.onnx e tokens.txt.
     */
    suspend fun initModel(modelDir: String, numThreads: Int = 4): Boolean =
        withContext(Dispatchers.IO) {
            if (isLoaded) close()
            val dir = File(modelDir)
            val encoder = File(dir, "encoder.int8.onnx")
            val decoder = File(dir, "decoder.int8.onnx")
            val joiner = File(dir, "joiner.int8.onnx")
            val tokens = File(dir, "tokens.txt")
            if (!encoder.exists() || !decoder.exists() || !joiner.exists() || !tokens.exists()) {
                Log.e(TAG, "Arquivos do Nemotron ausentes em $modelDir")
                return@withContext false
            }
            try {
                val config = OnlineRecognizerConfig(
                    featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
                    modelConfig = OnlineModelConfig(
                        transducer = OnlineTransducerModelConfig(
                            encoder = encoder.absolutePath,
                            decoder = decoder.absolutePath,
                            joiner = joiner.absolutePath,
                        ),
                        tokens = tokens.absolutePath,
                        numThreads = numThreads,
                        provider = "cpu",
                        modelType = "nemotron",
                    ),
                    // Push-to-talk: transcrição de um enunciado completo, não streaming contínuo.
                    // Desabilita endpointing automático (o app controla início/fim).
                    enableEndpoint = false,
                    decodingMethod = "greedy_search",
                )
                recognizer = OnlineRecognizer(assetManager = null, config = config)
                Log.i(TAG, "Nemotron 3.5 carregado (dir=$modelDir, threads=$numThreads)")
                true
            } catch (e: Throwable) {
                Log.e(TAG, "Falha ao carregar Nemotron", e)
                recognizer = null
                false
            }
        }

    /**
     * Transcreve um enunciado completo (amostras float [-1,1], 16 kHz mono).
     */
    suspend fun transcribe(audioData: FloatArray): String = withContext(Dispatchers.IO) {
        val rec = recognizer ?: error("Nemotron não foi inicializado")
        if (audioData.isEmpty()) return@withContext ""
        val stream = rec.createStream()
        try {
            // Fixa o idioma por-stream (modelo multilíngue).
            try {
                stream.setOption("lang", LANGUAGE)
            } catch (e: Throwable) {
                Log.d(TAG, "setOption(lang) ignorado: ${e.message}")
            }
            stream.acceptWaveform(audioData, SAMPLE_RATE)
            // Cauda de zeros para drenar o buffer do chunk de 560 ms.
            stream.acceptWaveform(FloatArray(SAMPLE_RATE / 2), SAMPLE_RATE)
            stream.inputFinished()
            while (rec.isReady(stream)) {
                rec.decode(stream)
            }
            val result = rec.getResult(stream)
            result.text.trim()
        } finally {
            stream.release()
        }
    }

    override fun close() {
        recognizer?.release()
        recognizer = null
        Log.i(TAG, "NemotronAsrEngine liberado")
    }
}
