package br.org.ceia.cemigpoc.domain.pipeline

import android.util.Log
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Pipeline conversacional multiturno consolidado (sem Turno 1 de rewrite).
 *
 * Decisão do Capitão (POC consolidação): o Turno 1 de rewrite via LLM foi REMOVIDO do caminho.
 * A bancada do classifier provou que keywords reescritas PIORAM a busca vs fala bruta
 * (22.5% vs 28.5% R@2 — ver classifier/README.md) e o rewrite custava ~1-1.5 s por pergunta.
 *
 * Fluxo atual:
 * - ASR -> transcrição da fala do trabalhador (medida fora, `asrMs`).
 * - Estágio 1+2 (HybridRetriever): classificador leve de NR dá boost/filtro gated ao BM25;
 *   a busca BM25 recebe a FALA BRUTA (não keywords). A decisão do gate é exposta para o
 *   Modo Engenharia via TurnEvent.Classified.
 * - Reuso por Jaccard: se a fala bruta atual for quase idêntica (> jaccardThreshold) à do
 *   turno anterior e houver chunks, reutiliza os chunks sem nova consulta BM25. Mantido
 *   porque em multiturno perguntas de continuação repetem a fala anterior; agora compara
 *   falas BRUTAS consecutivas (antes comparava keywords do rewrite, que não existe mais).
 * - Turno 2 (Síntese): system prompt + histórico enxuto (últimas maxTurnsT2 trocas brutas)
 *   + chunks recuperados + pergunta atual -> síntese com citação obrigatória de norma.
 *   - Orçamento rígido: T2 <= t2MaxBudgetTokens; se estourar, poda a troca mais antiga.
 */
class AskPipeline(
    private val retriever: Retriever,
    private val llmEngine: LlmEngine,
    private val telemetryLogger: TelemetryLogger? = null,
    val maxTurnsT2: Int = 3,
    val t2MaxBudgetTokens: Int = 1000,
    val jaccardThreshold: Double = 0.7,
    val topK: Int = 2
) {
    companion object {
        private const val TAG = "AskPipeline"

        // Prompt de síntese CONCISO (variante 'v1_rigido' vencedora do experimento judge151):
        // dev nas 20 smoke (fora do holdout) -> 144 tok médios, 100% sem markdown, 75% citação
        // inline. Escolhido no experimento do 2.6B thinking-OFF (gate 9.9% na GPU), mas mantido
        // no 1.2B embarcado (fallback do brief; ver ModelFileManager e README_CONSOLIDACAO):
        // domar a verbosidade para o canal de VOZ vale para ambos os modelos. Limites duros.
        const val SYNTHESIS_SYSTEM_PROMPT =
            "Você é o assistente técnico de campo da CEMIG. O eletricista OUVE sua resposta por voz, então seja curto e direto.\n" +
            "REGRAS OBRIGATÓRIAS (nunca viole):\n" +
            "- Responda em NO MÁXIMO 4 frases curtas.\n" +
            "- PROIBIDO usar markdown, títulos, negrito, listas, bullets ou numeração. Escreva em prosa corrida.\n" +
            "- Cite a norma e o item DENTRO da frase (ex.: 'conforme a NR-10, item 10.5.1, ...').\n" +
            "- Baseie-se EXCLUSIVAMENTE no contexto normativo fornecido; não invente procedimentos.\n" +
            "- Se o contexto não responder, diga apenas: 'Não sei com base nas normas consultadas.'"
    }

    /**
     * Executa o processamento multiturno completo de uma pergunta.
     */
    fun execute(
        userQuestion: String,
        history: List<ConversationTurn> = emptyList(),
        asrMs: Long = 0L
    ): Flow<TurnEvent> = flow {
        val totalStart = System.currentTimeMillis()

        // ---------------------------------------------------------------------
        // ETAPA 1: Classificação (estágio 1) + Busca BM25 (estágio 2) com fala bruta
        // ---------------------------------------------------------------------
        emit(TurnEvent.StageChanged(PipelineStage.CLASSIFYING))

        // Reuso por Jaccard: compara a fala BRUTA atual com a do turno anterior.
        val previousTurn = history.lastOrNull()
        val jaccard = calculateJaccardSimilarity(userQuestion, previousTurn?.question.orEmpty())
        val canReuse = previousTurn != null &&
                previousTurn.chunks.isNotEmpty() &&
                jaccard >= jaccardThreshold

        val chunks: List<Chunk>
        val searchMs: Long
        var decision: HybridRetriever.Decision? = null

        if (canReuse) {
            chunks = previousTurn!!.chunks
            searchMs = 0L
            Log.i(TAG, "Reuso de chunks anteriores (Jaccard=%.2f) para fala '%s'".format(jaccard, userQuestion))
            emit(TurnEvent.ChunksRetrieved(chunks, reused = true, durationMs = 0L))
        } else {
            val searchStart = System.currentTimeMillis()
            chunks = try {
                // Estágio 1 híbrido: classifica a FALA BRUTA e aplica boost/filtro gated;
                // o BM25 também recebe a fala bruta (melhor que keywords — ver AGENTS.md).
                if (retriever is HybridRetriever) {
                    // Retrieval v3: se a busca densa estiver disponível, usa fusão RRF
                    // 3-sinais (BM25-gated-exp + 2 densos EmbeddingGemma); senão BM25-gated.
                    val r = if (retriever.hasDense) {
                        retriever.searchV3(rawQuestion = userQuestion, topK = topK)
                    } else {
                        retriever.searchWithRaw(bm25Query = userQuestion, rawQuestion = userQuestion, topK = topK)
                    }
                    decision = retriever.lastDecision
                    r
                } else {
                    retriever.search(query = userQuestion, topK = topK)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Erro na busca BM25", e)
                emptyList()
            }
            searchMs = System.currentTimeMillis() - searchStart

            // Expõe a decisão do gate para o Modo Engenharia (o capitão quer ver a escolha).
            decision?.let { d ->
                emit(
                    TurnEvent.Classified(
                        nrTop1 = d.top1,
                        nrTop1Prob = d.top1Prob,
                        nrTop2 = d.top2,
                        gateMode = d.mode,
                        boostNrs = d.boostNrs
                    )
                )
            }

            Log.i(TAG, "Busca BM25 (fala bruta) recuperou ${chunks.size} chunks em ${searchMs}ms")
            emit(TurnEvent.ChunksRetrieved(chunks, reused = false, durationMs = searchMs))
        }

        // ---------------------------------------------------------------------
        // ETAPA 2: Turno 2 - Síntese Final com Orçamento de Contexto
        // ---------------------------------------------------------------------
        emit(TurnEvent.StageChanged(PipelineStage.RESPONDING))
        val t2Start = System.currentTimeMillis()

        val contextStr = if (chunks.isNotEmpty()) {
            chunks.mapIndexed { idx, c ->
                "[${idx + 1}] (${c.doc} - ${c.section} - ${c.title}):\n${c.content}"
            }.joinToString("\n\n")
        } else {
            "Nenhum contexto normativo recuperado para a consulta."
        }

        val userContentT2 = "Contexto normativo consultado:\n$contextStr\n\nPergunta do eletricista:\n$userQuestion"

        // Orçamento de Turno 2: poda trocas mais antigas se ultrapassar limite de tokens
        val rawT2History = history.takeLast(maxTurnsT2)
        val prunedT2History = pruneHistoryForBudget(rawT2History, SYNTHESIS_SYSTEM_PROMPT, userContentT2, t2MaxBudgetTokens)

        val messagesT2 = mutableListOf<Message>()
        for (turn in prunedT2History) {
            messagesT2.add(Message(role = Message.Role.USER, content = turn.question))
            messagesT2.add(Message(role = Message.Role.ASSISTANT, content = turn.answer))
        }
        messagesT2.add(Message(role = Message.Role.USER, content = userContentT2))

        val answerBuilder = StringBuilder()
        var ttftMs = 0L
        var firstTokenReceived = false
        var completionTokens = 0

        try {
            llmEngine.streamChat(messagesT2, SYNTHESIS_SYSTEM_PROMPT).collect { chunk ->
                if (chunk is LlmResponseChunk.Text) {
                    if (!firstTokenReceived) {
                        firstTokenReceived = true
                        ttftMs = System.currentTimeMillis() - t2Start
                    }
                    completionTokens++
                    answerBuilder.append(chunk.delta)
                    emit(TurnEvent.TextDelta(chunk.delta))
                }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Erro na síntese do Turno 2", e)
            emit(TurnEvent.Error("Falha na síntese de resposta: ${e.message}", e))
            return@flow
        }

        val t2End = System.currentTimeMillis()
        val totalMs = t2End - totalStart
        val decodeMs = maxOf(0L, (t2End - t2Start) - ttftMs)
        val tokPerSec = if (decodeMs > 0) (completionTokens.toDouble() / (decodeMs / 1000.0)) else 0.0

        val estimatedContextTokens = estimateTokens(SYNTHESIS_SYSTEM_PROMPT) +
                prunedT2History.sumOf { estimateTokens(it.question) + estimateTokens(it.answer) } +
                estimateTokens(userContentT2)

        val finalAnswer = answerBuilder.toString().trim()

        val metrics = TurnMetrics(
            asrMs = asrMs,
            searchMs = searchMs,
            ttftMs = ttftMs,
            decodeMs = decodeMs,
            totalMs = totalMs,
            contextTokens = estimatedContextTokens,
            completionTokens = completionTokens,
            tokPerSec = Math.round(tokPerSec * 100.0) / 100.0,
            nrTop1 = decision?.top1 ?: (if (canReuse) "-" else ""),
            nrTop1Prob = decision?.top1Prob ?: 0f,
            nrTop2 = decision?.top2 ?: "",
            gateMode = decision?.mode ?: (if (canReuse) "reuso" else "-"),
            boostNrs = decision?.boostNrs?.joinToString(",").orEmpty(),
            chunksReused = canReuse,
            retrievalMode = decision?.retrieval ?: (if (canReuse) "reuso" else "bm25"),
            // O encode denso está embutido no searchMs; expomos separadamente quando possível.
            denseEncodeMs = if (decision?.retrieval == "rrf3") searchMs else 0L
        )

        emit(TurnEvent.Done(finalAnswer = finalAnswer, chunksUsed = chunks, metrics = metrics))

        // Telemetria local JSONL
        try {
            telemetryLogger?.logTurn(
                turnIndex = history.size + 1,
                question = userQuestion,
                transcription = userQuestion,
                finalAnswer = finalAnswer,
                chunksUsed = chunks,
                metrics = metrics
            )
        } catch (e: Throwable) {
            Log.w(TAG, "Falha ao gravar telemetria", e)
        }
    }

    /**
     * Calcula a similaridade de Jaccard entre duas falas brutas (reuso de chunks em multiturno).
     */
    fun calculateJaccardSimilarity(text1: String, text2: String): Double {
        val s1 = tokenizeWords(text1)
        val s2 = tokenizeWords(text2)
        if (s1.isEmpty() && s2.isEmpty()) return 1.0
        if (s1.isEmpty() || s2.isEmpty()) return 0.0

        val intersection = s1.intersect(s2).size
        val union = s1.union(s2).size
        return if (union > 0) intersection.toDouble() / union.toDouble() else 0.0
    }

    private fun tokenizeWords(text: String): Set<String> {
        return text.lowercase()
            .replace(Regex("[^a-zA-Z0-9À-ÿ]"), " ")
            .split(Regex("\\s+"))
            .filter { it.length >= 3 }
            .toSet()
    }

    /**
     * Poda o histórico mais antigo para manter o contexto do Turno 2 rigorosamente abaixo do orçamento.
     */
    fun pruneHistoryForBudget(
        history: List<ConversationTurn>,
        systemPrompt: String,
        userContent: String,
        maxBudgetTokens: Int
    ): List<ConversationTurn> {
        val baseTokens = estimateTokens(systemPrompt) + estimateTokens(userContent)
        val result = history.toMutableList()

        while (result.isNotEmpty()) {
            val historyTokens = result.sumOf { estimateTokens(it.question) + estimateTokens(it.answer) }
            if (baseTokens + historyTokens <= maxBudgetTokens) {
                break
            }
            // Remove a troca mais antiga
            result.removeAt(0)
        }

        return result
    }

    /**
     * Estimador rápido de contagem de tokens para controle de orçamento (1 token ~ 3.8 caracteres).
     */
    fun estimateTokens(text: String): Int {
        if (text.isBlank()) return 0
        return maxOf(1, (text.length / 3.8).toInt())
    }
}
