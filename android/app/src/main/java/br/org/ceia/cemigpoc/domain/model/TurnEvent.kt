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
    val denseEncodeMs: Long = 0L,
    // Tool-calling híbrido (task poc-app-tools): decisão do PRÓPRIO modelo neste turno.
    // - calledTool: o modelo emitiu buscar_norma() (houve busca externa)?
    // - toolConsulta/toolNr: argumentos que o modelo gerou (a busca IGNORA a consulta e usa a
    //   fala bruta + classificador + RRF v4; guardados só para depurar o hibrido).
    // - reasonMs: tempo do 1º turno de geração (decisão) até a tool_call/1º token.
    val calledTool: Boolean = false,
    val toolConsulta: String = "",
    val toolNr: String = "",
    val reasonMs: Long = 0L
)

/**
 * Estados do ciclo de vida da execução do pipeline.
 */
enum class PipelineStage(val label: String) {
    IDLE("PRONTO · OFFLINE"),
    LISTENING("OUVINDO VOZ..."),
    TRANSCRIBING("TRANSCREVENDO (NEMOTRON 3.5 INT8)..."),
    CLASSIFYING("MODELO DECIDINDO: BUSCAR OU REUSAR..."),
    RESPONDING("SINTETIZANDO RESPOSTA (LFM2.5 1.2B)..."),
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
    val isStreaming: Boolean = false,
    // Tool-calling híbrido: metadados da decisão do modelo neste turno, para reconstruir o
    // histórico no formato nativo (a chamada que o modelo EMITIU é replicada byte-a-byte no
    // prompt do turno seguinte, mantendo a conversa coerente com o que ele viu). `toolConsulta`
    // é o argumento REESCRITO pelo modelo (só reconstrução; a busca real usa a fala bruta).
    val calledTool: Boolean = false,
    val toolConsulta: String? = null,
    val toolNr: String? = null
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
    /**
     * Decisão de tool-calling do modelo (task poc-app-tools) tornada visível ao Modo Engenharia.
     * @param called o modelo emitiu buscar_norma()?
     * @param consulta argumento reescrito pelo modelo (IGNORADO na busca; só depuração)
     * @param nr norma que o modelo citou no argumento (ou vazio)
     * @param reasonMs tempo do turno de decisão
     */
    data class ToolDecided(
        val called: Boolean,
        val consulta: String,
        val nr: String,
        val reasonMs: Long
    ) : TurnEvent
    data class TextDelta(val text: String) : TurnEvent
    data class Done(
        val finalAnswer: String,
        val chunksUsed: List<Chunk>,
        val metrics: TurnMetrics
    ) : TurnEvent
    data class Error(val message: String, val cause: Throwable? = null) : TurnEvent
}
