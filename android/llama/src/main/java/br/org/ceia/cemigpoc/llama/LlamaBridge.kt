package br.org.ceia.cemigpoc.llama

import android.util.Log

/**
 * Interface de callback chamada a cada token produzido pelo motor de inferência C++.
 */
fun interface LlamaCallback {
    /**
     * Invocado com os BYTES UTF-8 de cada token emitido (podem ser um pedaço de caractere
     * multibyte; o consumidor junta antes de decodificar — ver Utf8Accumulator).
     * @return `true` para prosseguir na geração; `false` para abortar imediatamente.
     */
    fun onToken(token: ByteArray): Boolean
}

/**
 * Junta bytes UTF-8 de tokens e devolve só o texto COMPLETO (segura o sufixo de um caractere
 * multibyte partido entre tokens). Evita '�' em acentos/símbolos no streaming.
 */
class Utf8Accumulator {
    private var pending = ByteArray(0)

    fun push(bytes: ByteArray): String {
        val buf = pending + bytes
        // Recuo até o último início de caractere completo.
        var cut = buf.size
        var i = buf.size - 1
        var back = 0
        while (i >= 0 && back < 4) {
            val b = buf[i].toInt() and 0xFF
            if (b and 0xC0 != 0x80) { // byte líder (ou ASCII)
                val need = when {
                    b < 0x80 -> 1
                    b >= 0xF0 -> 4
                    b >= 0xE0 -> 3
                    b >= 0xC0 -> 2
                    else -> 1
                }
                if (buf.size - i < need) cut = i
                break
            }
            i--; back++
        }
        pending = buf.copyOfRange(cut, buf.size)
        return String(buf, 0, cut, Charsets.UTF_8)
    }

    fun flush(): String {
        val s = String(pending, Charsets.UTF_8)
        pending = ByteArray(0)
        return s
    }
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
        addAssistant: Boolean,
        suppressReasoning: Boolean
    ): String

    // prompt em bytes UTF-8 PADRÃO (o JNI GetStringUTFChars usa modified-UTF-8 e quebraria a
    // paridade de tokenização em chars fora do BMP). keepPrefix: reusa o estado se ele for
    // prefixo exato do novo prompt (decisão -> síntese, task lfm3-f7).
    external fun nativeGenerate(
        handle: Long,
        prompt: ByteArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repPenalty: Float,
        minP: Float,
        suppressReasoning: Boolean,
        keepPrefix: Boolean,
        callback: LlamaCallback
    ): Int

    /** Nº de tokens do texto no tokenizador do GGUF carregado (sem BOS; especiais reconhecidos). */
    external fun nativeCountTokens(handle: Long, text: ByteArray): Int

    external fun nativeClearKv(handle: Long)

    external fun nativeFree(handle: Long)

    // --- Modo embedding (Retrieval v3): encoder EmbeddingGemma-300M GGUF ---
    external fun nativeLoadEmbedder(modelPath: String, nCtx: Int, nThreads: Int): Long

    external fun nativeEmbed(handle: Long, text: String): FloatArray?

    external fun nativeFreeEmbedder(handle: Long)
}
