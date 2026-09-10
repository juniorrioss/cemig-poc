package br.org.ceia.cemigpoc.llama

import android.util.Log

/**
 * Interface de callback chamada a cada token produzido pelo motor de inferência C++.
 */
fun interface LlamaCallback {
    /**
     * Invocado com o fragmento textual de cada token emitido.
     * @return `true` para prosseguir na geração; `false` para abortar imediatamente.
     */
    fun onToken(token: String): Boolean
}

/**
 * Ponte JNI direta com a biblioteca nativa `libllama_engine.so`.
 */
class LlamaBridge {

    companion object {
        private const val TAG = "LlamaBridge"

        init {
            try {
                System.loadLibrary("llama_engine")
                Log.i(TAG, "libllama_engine carregada com sucesso")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Falha ao carregar libllama_engine", e)
            }
        }
    }

    external fun nativeLoad(modelPath: String, nCtx: Int, nThreads: Int): Long

    external fun nativeApplyChatTemplate(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        addAssistant: Boolean
    ): String

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repPenalty: Float,
        callback: LlamaCallback
    ): Int

    external fun nativeClearKv(handle: Long)

    external fun nativeFree(handle: Long)
}
