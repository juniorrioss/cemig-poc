package br.org.ceia.cemigpoc.domain.model

/**
 * Métricas detalhadas de tempo e consumo por etapa de um turno conversacional.
 */
data class TurnMetrics(
    val asrMs: Long = 0L,
    val rewriteMs: Long = 0L,
    val searchMs: Long = 0L,
    val ttftMs: Long = 0L,
    val decodeMs: Long = 0L,
    val totalMs: Long = 0L,
    val contextTokens: Int = 0,
    val completionTokens: Int = 0,
    val tokPerSec: Double = 0.0,
    val keywords: String = "",
    val keywordsReused: Boolean = false
)

/**
 * Estados do ciclo de vida da execução do pipeline.
 */
enum class PipelineStage(val label: String) {
    IDLE("PRONTO · OFFLINE"),
    LISTENING("OUVINDO VOZ..."),
    TRANSCRIBING("TRANSCREVENDO (WHISPER BASE)..."),
    REWRITING("REESCREVENDO TERMOS (TURNO 1)..."),
    SEARCHING("CONSULTANDO NORMAS (BM25 TOP-2)..."),
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
    val keywords: String = "",
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
    data class KeywordsExtracted(val keywords: String, val reused: Boolean, val durationMs: Long) : TurnEvent
    data class ChunksRetrieved(val chunks: List<Chunk>, val reused: Boolean, val durationMs: Long) : TurnEvent
    data class TextDelta(val text: String) : TurnEvent
    data class Done(
        val finalAnswer: String,
        val chunksUsed: List<Chunk>,
        val metrics: TurnMetrics
    ) : TurnEvent
    data class Error(val message: String, val cause: Throwable? = null) : TurnEvent
}
