package br.org.ceia.cemigpoc.asr

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.io.File
import java.util.Locale

/**
 * Exp 0 — ASR Nativo Android Offline (M2 Reference Implementation).
 *
 * Avaliação do SpeechRecognizer nativo do Android com preferência explícita
 * por execução offline (EXTRA_PREFER_OFFLINE).
 *
 * Comportamento verificado no Galaxy S24+ (Android 16):
 * 1. Provedor padrão: com.google.android.tts/com.google.android.apps.speech.tts.googletts.service.GoogleTTSRecognitionService
 * 2. Provedor OEM Samsung: com.samsung.android.intellivoiceservice/.asr.SpeechRecognizerService
 * 3. Dependência crítica: exige que o pacote de idioma off-line "Português (Brasil)"
 *    já esteja baixado no aparelho pelo usuário/MDM. Sem o pacote, o sistema
 *    retorna SpeechRecognizer.ERROR_SERVER_DISCONNECTED (code 2) ou falha silenciosamente.
 * 4. Não permite customização de vocabulário técnico (NR-10, kV, religador, LOTO).
 */
class NativeSpeechRecognizerTest(
    private val context: Context,
    private val onResult: (String) -> Unit,
    private val onError: (Int, String) -> Unit
) {
    companion object {
        private const val TAG = "NativeSpeechRecognizer"
    }

    private var speechRecognizer: SpeechRecognizer? = null

    fun initialize() {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            onError(-1, "Speech recognition is not available on this device")
            return
        }

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    Log.d(TAG, "onReadyForSpeech")
                }

                override fun onBeginningOfSpeech() {
                    Log.d(TAG, "onBeginningOfSpeech")
                }

                override fun onRmsChanged(rmsdB: Float) {}

                override fun onBufferReceived(buffer: ByteArray?) {}

                override fun onEndOfSpeech() {
                    Log.d(TAG, "onEndOfSpeech")
                }

                override fun onError(error: Int) {
                    val msg = when (error) {
                        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                        SpeechRecognizer.ERROR_CLIENT -> "Client side error"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
                        SpeechRecognizer.ERROR_NETWORK -> "Network error"
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
                        SpeechRecognizer.ERROR_NO_MATCH -> "No recognition result matched"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "RecognitionService busy"
                        SpeechRecognizer.ERROR_SERVER -> "Server error"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input"
                        SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> "Server disconnected / offline model not downloaded"
                        else -> "Unknown error ($error)"
                    }
                    Log.e(TAG, "SpeechRecognizer error: $msg ($error)")
                    onError(error, msg)
                }

                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = matches?.firstOrNull() ?: ""
                    Log.i(TAG, "SpeechRecognizer result: $text")
                    onResult(text)
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    Log.d(TAG, "Partial: ${matches?.firstOrNull()}")
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    /**
     * Inicia a escuta com forçamento de modo estritamente offline.
     */
    fun startListeningOffline() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "pt-BR")
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "pt-BR")
            putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, "pt-BR")
            // Android 6.0+ (API 23+) flag para forçar reconhecimento 100% no dispositivo
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }

        speechRecognizer?.startListening(intent)
    }

    fun destroy() {
        speechRecognizer?.destroy()
        speechRecognizer = null
    }
}
