package br.org.ceia.cemigpoc.llama

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * Encoder de embeddings on-device (Retrieval v3) usando o MESMO runtime llama.cpp já
 * linkado (libllama_engine.so). Carrega o EmbeddingGemma-300M QAT-Q4_0 em modo embedding
 * (pooling mean, embeddings=true) e codifica UMA query por pergunta.
 *
 * Prompt oficial do EmbeddingGemma para consulta de retrieval:
 *   "task: search result | query: {texto}"
 * (os DOCUMENTOS foram indexados offline com "title: none | text: {texto}"; ambos no
 * MESMO espaço vetorial 768-d — ver retrieval3/build_dense_gguf_index.py.)
 *
 * O vetor retornado já vem L2-normalizado do lado nativo (cosseno = produto interno).
 */
class LlamaEmbedder(
    private val bridge: LlamaBridge = LlamaBridge()
) : Closeable {

    companion object {
        private const val TAG = "LlamaEmbedder"
        const val QUERY_PROMPT = "task: search result | query: "
        // Contexto pequeno: a query do operário é curta; economiza RAM do encoder.
        const val EMBED_CTX = 1024
        const val EMBED_THREADS = 6
    }

    private var handle: Long = 0L

    val isLoaded: Boolean
        get() = handle != 0L

    suspend fun load(modelPath: String): Boolean = withContext(Dispatchers.IO) {
        if (isLoaded) close()
        val h = bridge.nativeLoadEmbedder(modelPath, EMBED_CTX, EMBED_THREADS)
        if (h == 0L) {
            Log.e(TAG, "Falha ao carregar encoder de embedding em $modelPath")
            return@withContext false
        }
        handle = h
        Log.i(TAG, "Encoder EmbeddingGemma carregado (handle=$handle)")
        true
    }

    /** Codifica a query (aplica o prompt oficial). Retorna vetor 768-d L2-normalizado. */
    suspend fun embedQuery(text: String): FloatArray? = withContext(Dispatchers.IO) {
        if (!isLoaded) {
            Log.w(TAG, "embedQuery chamado sem encoder carregado")
            return@withContext null
        }
        bridge.nativeEmbed(handle, QUERY_PROMPT + text)
    }

    override fun close() {
        if (isLoaded) {
            val h = handle
            handle = 0L
            bridge.nativeFreeEmbedder(h)
            Log.i(TAG, "LlamaEmbedder liberado")
        }
    }
}
