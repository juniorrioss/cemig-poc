package br.org.ceia.cemigpoc.domain.pipeline

import android.util.Log
import br.org.ceia.cemigpoc.data.engine.RealLlamaEngine
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
 * Pipeline conversacional multiturno em Modo C Top-2 (Rewrite -> BM25 -> Injeção).
 *
 * Arquitetura de turnos decidida pelo Capitão:
 * - Turno 1 (Rewrite): Recebe histórico bruto (até MAX_TURNS_T1=6 trocas) + pergunta atual -> extrai 3 a 6 keywords técnicas.
 *   - Verificação Jaccard (> 0.7): se as palavras-chave forem quase idênticas às do turno anterior,
 *     reutiliza os chunks do turno anterior sem nova consulta BM25.
 * - Busca BM25 Top-2 no acervo consolidado de 36 NRs (`index_hf_36nr.db`).
 * - Turno 2 (Síntese): System prompt + histórico enxuto (últimas MAX_TURNS_T2=3 trocas brutas, sem chunks antigos)
 *   + 2 chunks recuperados + pergunta atual -> síntese com citação obrigatória de norma.
 *   - Orçamento rígido: T2 <= ~1000 tokens. Se estourar, poda a troca mais antiga do histórico.
 *   - Prefix cache estável: prefixo mantido na memória KV do llama.cpp para acelerar TTFT.
 */
class AskPipeline(
    private val retriever: Retriever,
    private val llmEngine: LlmEngine,
    private val telemetryLogger: TelemetryLogger? = null,
    val maxTurnsT1: Int = 6,
    val maxTurnsT2: Int = 3,
    val t2MaxBudgetTokens: Int = 1000,
    val jaccardThreshold: Double = 0.7,
    val topK: Int = 2
) {
    companion object {
        private const val TAG = "AskPipeline"

        const val REWRITE_SYSTEM_PROMPT =
            "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n" +
            "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n" +
            "Dado o histórico da conversa e a nova fala do trabalhador, gere de 3 a 6 palavras-chave técnicas para busca.\n" +
            "Diretrizes mandatórias:\n" +
            "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n" +
            "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n" +
            "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n" +
            "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."

        const val SYNTHESIS_SYSTEM_PROMPT =
            "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras (NR-10, NR-06, NR-35, NR-12, NR-18 e demais NRs aplicáveis).\n" +
            "Suas diretrizes mandatórias:\n" +
            "1. Responda em português brasileiro com precisão técnica e objetividade.\n" +
            "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. Não adicione procedimentos não contidos nas normas.\n" +
            "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n" +
            "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: " +
            "'Não sei com base nas normas consultadas.' Não tente adivinhar."
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
        // ETAPA 1: Turno 1 - Query Rewrite (Extração de palavras-chave técnicas)
        // ---------------------------------------------------------------------
        emit(TurnEvent.StageChanged(PipelineStage.REWRITING))
        val t1Start = System.currentTimeMillis()

        val rewriteHistory = history.takeLast(maxTurnsT1)
        val userPromptT1 = formatRewriteUserPrompt(rewriteHistory, userQuestion)

        var rawRewrite = ""
        try {
            if (llmEngine is RealLlamaEngine) {
                // Execução rápida síncrona via chat template do GGUF
                val pairs = listOf(
                    "system" to REWRITE_SYSTEM_PROMPT,
                    "user" to userPromptT1
                )
                val promptFormatted = llmEngine.applyChatTemplate(pairs, addAssistant = true)
                rawRewrite = llmEngine.generateComplete(promptFormatted, maxTokens = 40)
            } else {
                // Fallback genérico para motores de teste / fakes
                val messages = listOf(
                    Message(role = Message.Role.SYSTEM, content = REWRITE_SYSTEM_PROMPT),
                    Message(role = Message.Role.USER, content = userPromptT1)
                )
                val sb = StringBuilder()
                llmEngine.streamChat(messages, REWRITE_SYSTEM_PROMPT).collect { chunk ->
                    if (chunk is LlmResponseChunk.Text) {
                        sb.append(chunk.delta)
                    }
                }
                rawRewrite = sb.toString()
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Erro na reescrita do Turno 1; usando pergunta bruta como fallback", e)
            rawRewrite = userQuestion
        }

        val rewriteMs = System.currentTimeMillis() - t1Start
        val keywords = cleanKeywords(rawRewrite, fallback = userQuestion)

        // ---------------------------------------------------------------------
        // ETAPA 2: Similaridade Jaccard e Decisão de Reuso de Chunks
        // ---------------------------------------------------------------------
        val previousTurn = history.lastOrNull()
        val previousKeywords = previousTurn?.keywords.orEmpty()
        val jaccard = calculateJaccardSimilarity(keywords, previousKeywords)
        val canReuse = previousTurn != null &&
                previousTurn.chunks.isNotEmpty() &&
                jaccard >= jaccardThreshold

        val chunks: List<Chunk>
        val searchMs: Long

        if (canReuse) {
            chunks = previousTurn!!.chunks
            searchMs = 0L
            Log.i(TAG, "Turno 1: Keywords '$keywords' reutilizam chunks anteriores (Jaccard=%.2f)".format(jaccard))
            emit(TurnEvent.KeywordsExtracted(keywords, reused = true, durationMs = rewriteMs))
            emit(TurnEvent.ChunksRetrieved(chunks, reused = true, durationMs = 0L))
        } else {
            emit(TurnEvent.KeywordsExtracted(keywords, reused = false, durationMs = rewriteMs))
            emit(TurnEvent.StageChanged(PipelineStage.SEARCHING))

            val searchStart = System.currentTimeMillis()
            chunks = try {
                retriever.search(query = keywords, topK = topK)
            } catch (e: Throwable) {
                Log.e(TAG, "Erro na busca BM25", e)
                emptyList()
            }
            searchMs = System.currentTimeMillis() - searchStart
            Log.i(TAG, "Turno 1: Busca BM25 '$keywords' recuperou ${chunks.size} chunks em ${searchMs}ms")
            emit(TurnEvent.ChunksRetrieved(chunks, reused = false, durationMs = searchMs))
        }

        // ---------------------------------------------------------------------
        // ETAPA 3: Turno 2 - Síntese Final com Orçamento de Contexto
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
            rewriteMs = rewriteMs,
            searchMs = searchMs,
            ttftMs = ttftMs,
            decodeMs = decodeMs,
            totalMs = totalMs,
            contextTokens = estimatedContextTokens,
            completionTokens = completionTokens,
            tokPerSec = Math.round(tokPerSec * 100.0) / 100.0,
            keywords = keywords,
            keywordsReused = canReuse
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
     * Formata o prompt do Turno 1 integrando o histórico conversacional recente.
     */
    fun formatRewriteUserPrompt(history: List<ConversationTurn>, currentQuestion: String): String {
        if (history.isEmpty()) {
            return "Dúvida do trabalhador: $currentQuestion\nTermos técnicos para busca:"
        }

        return buildString {
            appendLine("Histórico recente da conversa:")
            history.forEach { turn ->
                appendLine("Usuário: ${turn.question}")
                appendLine("Assistente: ${turn.answer.take(160)}")
            }
            appendLine()
            appendLine("Nova dúvida do trabalhador: $currentQuestion")
            append("Termos técnicos para busca:")
        }
    }

    /**
     * Limpa e padroniza a resposta de palavras-chave geradas pelo Turno 1.
     */
    fun cleanKeywords(raw: String, fallback: String): String {
        var text = raw.lines().firstOrNull { it.isNotBlank() } ?: ""
        text = text.replace(Regex("^(Termos técnicos|Palavras-chave|Busca|Keywords):", RegexOption.IGNORE_CASE), "")
        text = text.replace(Regex("[*\"`']"), " ").trim()

        val words = text.split(Regex("[,;\\s]+")).filter { it.length >= 2 }
        if (words.isEmpty()) {
            return fallback.trim()
        }
        return words.take(6).joinToString(" ")
    }

    /**
     * Calcula a similaridade de Jaccard entre dois conjuntos de palavras-chave.
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
