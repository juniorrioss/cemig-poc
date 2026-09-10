package br.org.ceia.cemigpoc.whisper

import android.util.Log

/**
 * Ponte JNI direta com a biblioteca nativa `libwhisper_engine.so`.
 */
class WhisperBridge {

    companion object {
        private const val TAG = "WhisperBridge"

        init {
            try {
                System.loadLibrary("whisper_engine")
                Log.i(TAG, "libwhisper_engine carregada com sucesso")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Falha ao carregar libwhisper_engine", e)
            }
        }
    }

    external fun nativeInit(modelPath: String): Long

    external fun nativeTranscribe(
        handle: Long,
        audioData: FloatArray,
        numThreads: Int,
        language: String,
        initialPrompt: String
    ): String

    external fun nativeFree(handle: Long)
}
