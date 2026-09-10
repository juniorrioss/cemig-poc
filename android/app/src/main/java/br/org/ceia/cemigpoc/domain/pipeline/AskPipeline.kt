package br.org.ceia.cemigpoc.domain.pipeline

import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.domain.model.ToolCall
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Orquestrador central do fluxo de resposta com RAG disparado estritamente via tool-calling.
 *
 * Princípio arquitetural (Capitão):
 * O pipeline NÃO injeta chunks de forma estática no prompt. O LLM recebe a pergunta do
 * usuário e a especificação da ferramenta `retriever(query: String)`. O modelo avalia a
 * necessidade e formula os termos técnicos de busca (BM25). Quando a ferramenta é acionada,
 * o pipeline busca os trechos no índice FTS5, devolve o resultado como mensagem do tipo TOOL,
 * e o modelo produz a resposta final fundamentada nos trechos recuperados com citação obrigatória.
 */
class AskPipeline(
    private val systemPrompt: String,
    private val retriever: Retriever,
    private val llmEngine: LlmEngine,
    private val telemetryLogger: TelemetryLogger? = null
) {
    /**
     * Executa o loop de raciocínio, tool-calling e streaming de resposta.
     *
     * @param userQuestion Pergunta do operário (transcrita por voz ou digitada)
     * @return Fluxo de eventos sequenciais de turnos (TextDelta, ToolCall, ToolResult, Done)
     */
    fun execute(userQuestion: String): Flow<TurnEvent> = flow {
        val startTime = System.currentTimeMillis()
        val messages = mutableListOf<Message>()
        messages.add(Message(role = Message.Role.USER, content = userQuestion))

        val toolCallsMade = mutableListOf<ToolCall>()
        val chunksRetrieved = mutableListOf<Chunk>()
        var finalAnswerBuilder = StringBuilder()
        var activeToolCall: ToolCall? = null

        // ---------------------------------------------------------------------
        // TURNO 1: Avaliação inicial do LLM (resposta direta vs decisão de tool)
        // ---------------------------------------------------------------------
        try {
            llmEngine.streamChat(messages, systemPrompt).collect { chunk ->
                when (chunk) {
                    is LlmResponseChunk.Text -> {
                        finalAnswerBuilder.append(chunk.delta)
                        emit(TurnEvent.TextDelta(chunk.delta))
                    }
                    is LlmResponseChunk.Call -> {
                        activeToolCall = chunk.toolCall
                        toolCallsMade.add(chunk.toolCall)
                        emit(TurnEvent.ToolCallEvent(chunk.toolCall))
                    }
                }
            }
        } catch (e: Throwable) {
            emit(TurnEvent.Error("Falha na geração inicial do modelo: ${e.message}", e))
            return@flow
        }

        // ---------------------------------------------------------------------
        // TURNO 2: Execução da ferramenta de busca (se solicitada pelo modelo)
        // ---------------------------------------------------------------------
        if (activeToolCall != null) {
            val toolCall = activeToolCall!!
            val chunks = try {
                retriever.search(query = toolCall.query, topK = 3)
            } catch (e: Throwable) {
                emit(TurnEvent.Error("Falha ao consultar índice FTS5: ${e.message}", e))
                emptyList()
            }

            chunksRetrieved.addAll(chunks)
            emit(TurnEvent.ToolResultEvent(chunks))

            // Registra a intenção de chamada do assistente no histórico
            messages.add(
                Message(
                    role = Message.Role.ASSISTANT,
                    content = "",
                    toolCall = toolCall
                )
            )

            // Formata o retorno da ferramenta como mensagem para o modelo
            val toolResultContent = if (chunks.isEmpty()) {
                "Nenhum trecho relevante foi encontrado no índice de normas para a busca: '${toolCall.query}'."
            } else {
                buildString {
                    appendLine("Trechos encontrados nas normas regulamentadoras:")
                    chunks.forEachIndexed { index, chunk ->
                        appendLine("[Trecho ${index + 1}] (${chunk.doc} - ${chunk.section}):")
                        appendLine(chunk.content)
                    }
                }
            }
            messages.add(
                Message(
                    role = Message.Role.TOOL,
                    content = toolResultContent
                )
            )

            // Reinicia buffer para a resposta final após o resultado da ferramenta
            finalAnswerBuilder = StringBuilder()

            try {
                llmEngine.streamChat(messages, systemPrompt).collect { chunk ->
                    when (chunk) {
                        is LlmResponseChunk.Text -> {
                            finalAnswerBuilder.append(chunk.delta)
                            emit(TurnEvent.TextDelta(chunk.delta))
                        }
                        is LlmResponseChunk.Call -> {
                            // Turno único de tool-calling para esta fase da POC
                        }
                    }
                }
            } catch (e: Throwable) {
                emit(TurnEvent.Error("Falha na síntese final do modelo: ${e.message}", e))
                return@flow
            }
        }

        val totalDurationMs = System.currentTimeMillis() - startTime
        val finalAnswer = finalAnswerBuilder.toString().trim()

        val doneEvent = TurnEvent.Done(
            finalAnswer = finalAnswer,
            chunksUsed = chunksRetrieved,
            totalDurationMs = totalDurationMs,
            toolCallsMade = toolCallsMade
        )
        emit(doneEvent)

        // Registro de telemetria local JSONL
        try {
            telemetryLogger?.log(
                userQuestion = userQuestion,
                transcription = userQuestion,
                toolCallsMade = toolCallsMade,
                chunksUsed = chunksRetrieved,
                finalAnswer = finalAnswer,
                totalDurationMs = totalDurationMs
            )
        } catch (_: Throwable) {
            // Falha de telemetria nunca bloqueia o usuário
        }
    }
}
