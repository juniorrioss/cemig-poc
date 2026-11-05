package br.org.ceia.cemigpoc.llama

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * Motor de inferência LLM / SLM em execução nativa local (ARM64) via llama.cpp.
 *
 * Parâmetros de amostragem padrão (Model Card Liquid AI LFM2.5 Instruct):
 * - Temperature: 0.1 (foco em exatidão normativa e rigor factual)
 * - Top-P: 1.0 (amostragem guiada por temperatura baixa)
 * - Min-P: 0.05 (filtro de corte de cauda estatística)
 * - Repetition Penalty: 1.05 (prevenção de repetição de termos)
 * - Threads: 6 (núcleos de performance Cortex-X4 / A720 do Galaxy S24+)
 */
class LlamaCppEngine(
    private val bridge: LlamaBridge = LlamaBridge()
) : Closeable {

    companion object {
        private const val TAG = "LlamaCppEngine"

        // Defaults Liquid LFM2.5 Instruct certificados M1
        const val DEFAULT_TEMPERATURE = 0.1f
        const val DEFAULT_TOP_K = 50
        const val DEFAULT_TOP_P = 1.0f
        const val DEFAULT_REP_PENALTY = 1.05f
        const val DEFAULT_MIN_P = 0.05f
        const val DEFAULT_CONTEXT_SIZE = 2048
        const val DEFAULT_THREADS = 6
    }

    private var nativeHandle: Long = 0L

    /**
     * Suprime o bloco de raciocínio (thinking-OFF) para modelos cujo chat_template força
     * `<think>` no add_generation_prompt (ex.: LFM2.5-2.6B). Quando true, o template limpa
     * o `<think>` primado e a geração bane o token especial `<think>` (ver llama_jni.cpp).
     * Para o 1.2B QAD (não-reasoning) o efeito é no-op seguro. Configurável por build
     * (BuildConfig.SUPPRESS_REASONING) sem trocar o modelo default do app.
     */
    var suppressReasoning: Boolean = false

    /**
     * Perfil de amostragem. LFM2.5 (legado): temp 0.1 / top_k 50 / rep 1.05 / min_p 0.05.
     * LFM3 f4-r32-ep2 (task lfm3-f7): temp 0.1 / top_k 50 / rep 1.1 / min_p 0 — o MESMO do
     * treino/avaliação (gen_e2e.py da F5/F6) e do smoke bf16 x Q4_0 da F7.
     */
    var repPenalty: Float = DEFAULT_REP_PENALTY
    var minP: Float = DEFAULT_MIN_P

    /** Reusa o estado do prompt anterior quando ele é prefixo exato do novo (decisão -> síntese). */
    var keepPrefix: Boolean = false

    /** Nº de tokens do texto no tokenizador do GGUF carregado (-1 se não carregado). */
    fun countTokens(text: String): Int =
        if (isLoaded) bridge.nativeCountTokens(nativeHandle, text.toByteArray(Charsets.UTF_8)) else -1

    val isLoaded: Boolean
        get() = nativeHandle != 0L

    /**
     * Carrega os pesos quantizados GGUF na memória e inicializa o contexto.
     */
    suspend fun load(
        modelPath: String,
        ctxSize: Int = DEFAULT_CONTEXT_SIZE,
        nThreads: Int = DEFAULT_THREADS
    ): Boolean = withContext(Dispatchers.IO) {
        if (isLoaded) {
            Log.w(TAG, "Modelo já carregado. Liberando instância anterior...")
            close()
        }

        val handle = bridge.nativeLoad(modelPath, ctxSize, nThreads)
        if (handle == 0L) {
            Log.e(TAG, "Falha ao carregar modelo em $modelPath")
            return@withContext false
        }

        nativeHandle = handle
        Log.i(TAG, "LFM2.5 carregado com sucesso (handle=$nativeHandle)")
        true
    }

    /**
     * Aplica o Chat Template oficial embutido no arquivo GGUF (sem ChatML manual).
     */
    fun applyChatTemplate(
        messages: List<Pair<String, String>>,
        addAssistant: Boolean = true
    ): String {
        check(isLoaded) { "Motor Llama não está carregado" }

        val roles = messages.map { it.first }.toTypedArray()
        val contents = messages.map { it.second }.toTypedArray()

        return bridge.nativeApplyChatTemplate(
            nativeHandle, roles, contents, addAssistant, suppressReasoning
        )
    }

    /**
     * Gera resposta com emissão token a token em streaming contínuo.
     */
    fun generateStream(
        prompt: String,
        maxTokens: Int = 700,
        temperature: Float = DEFAULT_TEMPERATURE,
        topK: Int = DEFAULT_TOP_K,
        topP: Float = DEFAULT_TOP_P,
        repPenalty: Float = this.repPenalty
    ): Flow<String> = callbackFlow {
        check(isLoaded) { "Motor Llama não está carregado" }

        withContext(Dispatchers.IO) {
            val acc = Utf8Accumulator()
            bridge.nativeGenerate(
                handle = nativeHandle,
                prompt = prompt.toByteArray(Charsets.UTF_8),
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                repPenalty = repPenalty,
                minP = minP,
                suppressReasoning = suppressReasoning,
                keepPrefix = keepPrefix
            ) { bytes ->
                val text = acc.push(bytes)
                if (text.isEmpty()) true else trySend(text).isSuccess
            }
            val tail = acc.flush()
            if (tail.isNotEmpty()) trySend(tail)
            channel.close()
        }

        awaitClose { }
    }

    /**
     * Gera resposta síncrona/completa (usado no Turno 1 para geração curta de palavras-chave).
     */
    suspend fun generateComplete(
        prompt: String,
        maxTokens: Int = 50,
        temperature: Float = DEFAULT_TEMPERATURE,
        topK: Int = DEFAULT_TOP_K,
        topP: Float = DEFAULT_TOP_P,
        repPenalty: Float = this.repPenalty
    ): String = withContext(Dispatchers.IO) {
        check(isLoaded) { "Motor Llama não está carregado" }

        val acc = Utf8Accumulator()
        val sb = StringBuilder()
        bridge.nativeGenerate(
            handle = nativeHandle,
            prompt = prompt.toByteArray(Charsets.UTF_8),
            maxTokens = maxTokens,
            temperature = temperature,
            topK = topK,
            topP = topP,
            repPenalty = repPenalty,
            minP = minP,
            suppressReasoning = suppressReasoning,
            keepPrefix = false
        ) { bytes ->
            sb.append(acc.push(bytes))
            true
        }
        sb.append(acc.flush())
        sb.toString().trim()
    }

    /**
     * Limpa o cache KV e os tokens em cache entre conversas independentes.
     */
    fun clearKvCache() {
        if (isLoaded) {
            bridge.nativeClearKv(nativeHandle)
        }
    }

    override fun close() {
        if (isLoaded) {
            val h = nativeHandle
            nativeHandle = 0L
            bridge.nativeFree(h)
            Log.i(TAG, "LlamaCppEngine liberado com sucesso")
        }
    }
}
