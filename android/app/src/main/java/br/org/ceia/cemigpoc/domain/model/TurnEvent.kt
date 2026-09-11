package br.org.ceia.cemigpoc.domain.model

/**
 * Métricas detalhadas de tempo e consumo por etapa de um turno conversacional.
 *
 * Pipeline consolidado (sem Turno 1 de rewrite): ASR -> classificador+gate -> BM25 (fala
 * bruta) -> síntese. Os campos nr* registram a decisão do estágio 1 (HybridRetriever) para
 * o Modo Engenharia; chunksReused indica reuso por Jaccard entre falas brutas consecutivas.
 */
data class TurnMetrics(
    val asrMs: Long = 0L,
    val searchMs: Long = 0L,
    val ttftMs: Long = 0L,
    val decodeMs: Long = 0L,
    val totalMs: Long = 0L,
    val contextTokens: Int = 0,
    val completionTokens: Int = 0,
    val tokPerSec: Double = 0.0,
    // Estágio 1 (classificador híbrido gated): NR top-1/top-2, probabilidade e gate acionado.
    val nrTop1: String = "",
    val nrTop1Prob: Float = 0f,
    val nrTop2: String = "",
    val gateMode: String = "", // "hard" | "soft" | "none" | "explicit" | "-"
    val boostNrs: String = "", // NRs efetivamente boostadas/filtradas (join por vírgula)
    val chunksReused: Boolean = false,
    // Retrieval v3: estratégia usada ("rrf3" | "bm25" | "rrf3-fallback-bm25") e o custo do
    // encode denso on-device (EmbeddingGemma) dentro do tempo de busca.
    val retrievalMode: String = "bm25",
    val denseEncodeMs: Long = 0L
)

/**
 * Estados do ciclo de vida da execução do pipeline.
 */
enum class PipelineStage(val label: String) {
    IDLE("PRONTO · OFFLINE"),
    LISTENING("OUVINDO VOZ..."),
    TRANSCRIBING("TRANSCREVENDO (WHISPER BASE)..."),
    CLASSIFYING("CLASSIFICANDO NR + BUSCANDO (BM25 TOP-2)..."),
    RESPONDING("SINTETIZANDO RESPOSTA (LFM2.5)..."),
    DONE("RESPOSTA CONCLUÍDA"),
    ERROR("FALHA OPERACIONAL")
}

/**
 * Registro de uma troca conversacional {pergunta, resposta} no histórico.
 */
data class ConversationTurn(
    val id: String = java.util.UUID.randomUUID().toString(),
    val question: String,
    val answer: String,
    val chunks: List<Chunk> = emptyList(),
    val metrics: TurnMetrics = TurnMetrics(),
    val isStreaming: Boolean = false
)

/**
 * Eventos emitidos durante o processamento do pipeline multiturno (AskPipeline).
 */
sealed interface TurnEvent {
    data class StageChanged(val stage: PipelineStage) : TurnEvent
    data class AsrFinal(val text: String, val durationMs: Long) : TurnEvent
    /** Decisão do estágio 1 (classificador + gate) tornada visível para o Modo Engenharia. */
    data class Classified(
        val nrTop1: String,
        val nrTop1Prob: Float,
        val nrTop2: String,
        val gateMode: String,
        val boostNrs: List<String>
    ) : TurnEvent
    data class ChunksRetrieved(val chunks: List<Chunk>, val reused: Boolean, val durationMs: Long) : TurnEvent
    data class TextDelta(val text: String) : TurnEvent
    data class Done(
        val finalAnswer: String,
        val chunksUsed: List<Chunk>,
        val metrics: TurnMetrics
    ) : TurnEvent
    data class Error(val message: String, val cause: Throwable? = null) : TurnEvent
}
