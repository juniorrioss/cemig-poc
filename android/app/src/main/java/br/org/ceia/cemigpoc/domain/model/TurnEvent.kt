package br.org.ceia.cemigpoc.domain.model

/**
 * Eventos selados emitidos durante a execução do pipeline de perguntas (AskPipeline).
 * Representa cada turno do loop de raciocínio, tool-calling e streaming de resposta.
 */
sealed interface TurnEvent {
    /**
     * Emissão incremental de texto (streaming de resposta).
     */
    data class TextDelta(val text: String) : TurnEvent

    /**
     * Decisão do modelo de invocar uma ferramenta externa (ex: retriever BM25).
     */
    data class ToolCallEvent(val call: ToolCall) : TurnEvent

    /**
     * Resultado da execução da ferramenta contendo os chunks recuperados.
     */
    data class ToolResultEvent(val chunks: List<Chunk>) : TurnEvent

    /**
     * Conclusão do processamento, contendo a resposta final e métricas de execução.
     */
    data class Done(
        val finalAnswer: String,
        val chunksUsed: List<Chunk>,
        val totalDurationMs: Long,
        val toolCallsMade: List<ToolCall> = emptyList()
    ) : TurnEvent

    /**
     * Evento de falha durante a execução do pipeline.
     */
    data class Error(val message: String, val cause: Throwable? = null) : TurnEvent
}
