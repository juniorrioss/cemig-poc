package br.org.ceia.cemigpoc.domain.pipeline

import android.util.Log
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Pipeline conversacional HÍBRIDO de tool-calling (task poc-app-tools).
 *
 * Arquitetura provada (tools_oraculo/README.md), implementada EXATAMENTE:
 * - O PRÓPRIO MODELO (1.2B treinado em tool-calling, r128) decide QUANDO buscar e quando
 *   reusar o contexto já em tela (decisão F1 95%, reuso-correto 84,8%, novo-tópico 100%).
 * - Quando o modelo chama buscar_norma, o APP IGNORA a `consulta` reescrita pelo modelo e
 *   executa a busca externa já existente com a FALA BRUTA + classificador de NR + RRF v4.
 *   Motivo medido: a consulta reescrita pelo 1.2B PIORA a busca (got_gold 19-25% vs 51% da
 *   fala crua; R@2 22,3 vs 30,5) e não melhora com mais dado nem rank.
 * - A heurística Jaccard de reuso SAIU: quem decide reuso agora é o modelo (não chamar a
 *   ferramenta = reusar o contexto/histórico já presente).
 *
 * Laço por turno do usuário:
 *   1. DECISÃO: renderiza o prompt nativo (system BAKED com `List of tools:` + histórico +
 *      fala) e gera. O modelo emite `<|tool_call_start|>[buscar_norma(...)]` OU responde direto.
 *   2a. Se HÁ tool_call: parseia (nome+args), IGNORA a `consulta`, roda a busca externa (fala
 *       bruta + classificador + RRF v4), injeta o turno `tool` com os chunks, e gera a resposta
 *       final (streaming). LIMITE: 1 busca por turno do usuário (ver maxSearchesPerTurn).
 *   2b. Se NÃO há tool_call: a saída do turno de decisão É a resposta (reuso/saudação/fora de
 *       escopo) — nenhuma busca é feita.
 *
 * Falhas tratadas: chamada malformada (intenção de buscar preservada -> busca com fala bruta);
 * tool desconhecida (idem, cai no default seguro de fundamentar com as normas); segunda
 * tool_call na síntese é ignorada (limite de 1 busca/turno).
 *
 * Multiturno / orçamento (n_ctx 2048): o histórico é reconstruído no formato nativo (turnos
 * user + [tool_call + tool] + assistant). Se o prompt estimado passar de maxPromptTokens, a
 * troca MAIS ANTIGA é podada primeiro (o contexto normativo recente e a fala atual ficam).
 */
class AskPipeline(
    private val retriever: Retriever,
    private val llmEngine: LlmEngine,
    private val telemetryLogger: TelemetryLogger? = null,
    // System prompt de tool-calling IDÊNTICO ao do treino (asset tools_system_prompt.txt).
    // Default = SYNTHESIS_SYSTEM_PROMPT (fallback dos testes JVM); em produção o ViewModel
    // injeta a string BAKED lida do asset (paridade byte-a-byte, SHA256 363185b7…).
    private val toolSystemPrompt: String = SYNTHESIS_SYSTEM_PROMPT,
    val maxTurnsT2: Int = 3,
    // Orçamento do prompt de síntese (system + histórico + contexto + fala). O n_ctx é 2048;
    // deixamos folga para os tokens gerados. Poda a troca mais antiga do histórico se estourar.
    val maxPromptTokens: Int = 1700,
    val topK: Int = 2,
    // Teto de tokens do turno de DECISÃO: uma tool_call tem ~40 tok; uma resposta direta
    // (saudação/reuso curto) cabe folgada. Se o modelo não chamar, isto limita a resposta.
    val decisionMaxTokens: Int = 220,
    // Teto de tokens da SÍNTESE final (resposta de voz, <=4 frases).
    val synthesisMaxTokens: Int = 384
) {
    companion object {
        private const val TAG = "AskPipeline"

        // Limite RÍGIDO de buscas por turno do usuário (anti-loop de chamadas repetidas).
        private const val MAX_SEARCHES_PER_TURN = 1

        // Marcadores de tool-call reemitidos (ou não) pelo engine nativo.
        private val TOOL_MARKERS = listOf("<|tool_call_start|>", "buscar_norma(")

        // Prompt de síntese CONCISO herdado (fallback dos testes e do caminho não-tool). O
        // system de PRODUÇÃO do pipeline de tool-calling é o BAKED do treino (asset), injetado
        // via toolSystemPrompt. Mantido aqui para os testes JVM (FakeLlm) e compatibilidade.
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
     * Executa o processamento híbrido de tool-calling de uma pergunta.
     */
    fun execute(
        userQuestion: String,
        history: List<ConversationTurn> = emptyList(),
        asrMs: Long = 0L
    ): Flow<TurnEvent> = flow {
        val totalStart = System.currentTimeMillis()

        // Reconstrói o histórico no formato nativo, podado ao orçamento (troca mais antiga sai).
        val renderedHistory = buildRenderedHistory(history, userQuestion)

        // ---------------------------------------------------------------------
        // TURNO 1 — DECISÃO: o modelo decide buscar (tool_call) ou responder direto (reuso).
        // ---------------------------------------------------------------------
        emit(TurnEvent.StageChanged(PipelineStage.CLASSIFYING))
        val decisionStart = System.currentTimeMillis()

        val decisionPrompt = LfmToolRenderer.renderDecisionPrompt(
            systemPrompt = toolSystemPrompt,
            history = renderedHistory,
            userQuestion = userQuestion
        )

        val decisionBuffer = StringBuilder()
        try {
            llmEngine.generateFromPrompt(decisionPrompt, maxTokens = decisionMaxTokens).collect { chunk ->
                if (chunk is LlmResponseChunk.Text) decisionBuffer.append(chunk.delta)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Erro no turno de decisão (tool-calling)", e)
            emit(TurnEvent.Error("Falha na decisão do modelo: ${e.message}", e))
            return@flow
        }
        val decisionRaw = decisionBuffer.toString()
        val reasonMs = System.currentTimeMillis() - decisionStart

        val looksLikeCall = LfmToolCallParser.looksLikeToolCall(decisionRaw)
        val calls = if (looksLikeCall) LfmToolCallParser.parse(decisionRaw) else emptyList()
        val validCall = calls.firstOrNull { it.name == LfmToolCallParser.TOOL_NAME }
        // A intenção de buscar existe se: parseamos buscar_norma OU vimos o marcador (malformado
        // ou tool desconhecida). Em ambos, o default SEGURO é buscar com a fala bruta.
        val wantsSearch = validCall != null || looksLikeCall

        val modelConsulta = validCall?.arguments?.get("consulta")?.trim().orEmpty()
        val modelNr = validCall?.arguments?.get("nr")?.trim().orEmpty()

        emit(TurnEvent.ToolDecided(called = wantsSearch, consulta = modelConsulta, nr = modelNr, reasonMs = reasonMs))
        Log.i(TAG, "Decisão do modelo: buscar=$wantsSearch consulta='$modelConsulta' nr='$modelNr' (${reasonMs}ms)")

        // ---------------------------------------------------------------------
        // Ramo B — SEM tool_call: reuso/saudação/fora de escopo. A saída da decisão É a resposta.
        // ---------------------------------------------------------------------
        if (!wantsSearch) {
            emit(TurnEvent.StageChanged(PipelineStage.RESPONDING))
            // Emite a resposta direta acumulada (nenhuma busca; contexto já em tela é reusado).
            emit(TurnEvent.ChunksRetrieved(emptyList(), reused = true, durationMs = 0L))
            val directAnswer = stripToolResidue(decisionRaw).trim()
            emit(TurnEvent.TextDelta(directAnswer))

            val totalMs = System.currentTimeMillis() - totalStart
            val metrics = TurnMetrics(
                asrMs = asrMs, searchMs = 0L, ttftMs = reasonMs, decodeMs = 0L,
                totalMs = totalMs, contextTokens = estimateTokens(decisionPrompt),
                completionTokens = estimateTokens(directAnswer), tokPerSec = 0.0,
                nrTop1 = "-", nrTop1Prob = 0f, nrTop2 = "", gateMode = "reuso",
                boostNrs = "", chunksReused = true, retrievalMode = "reuso",
                denseEncodeMs = 0L, calledTool = false, toolConsulta = "", toolNr = "",
                reasonMs = reasonMs
            )
            emit(
                TurnEvent.Done(
                    finalAnswer = directAnswer, chunksUsed = emptyList(), metrics = metrics
                )
            )
            logTurnSafely(history.size + 1, userQuestion, directAnswer, emptyList(), metrics)
            return@flow
        }

        // ---------------------------------------------------------------------
        // Ramo A — COM tool_call: busca EXTERNA (IGNORA a consulta do modelo) + síntese.
        // ---------------------------------------------------------------------
        val searchStart = System.currentTimeMillis()
        var decision: HybridRetriever.Decision? = null
        val chunks: List<Chunk> = try {
            // CRÍTICO: busca com a FALA BRUTA (userQuestion), não com modelConsulta. O
            // classificador de NR + RRF v4 já vivem no HybridRetriever. Limite: 1 busca/turno.
            if (retriever is HybridRetriever) {
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
            Log.e(TAG, "Erro na busca externa (tool)", e)
            emptyList()
        }
        val searchMs = System.currentTimeMillis() - searchStart

        decision?.let { d ->
            emit(
                TurnEvent.Classified(
                    nrTop1 = d.top1, nrTop1Prob = d.top1Prob, nrTop2 = d.top2,
                    gateMode = d.mode, boostNrs = d.boostNrs
                )
            )
        }
        emit(TurnEvent.ChunksRetrieved(chunks, reused = false, durationMs = searchMs))
        Log.i(TAG, "Busca externa (fala bruta) recuperou ${chunks.size} chunks em ${searchMs}ms")

        // ---------------------------------------------------------------------
        // SÍNTESE: injeta a tool_call do modelo + o turno `tool` com os chunks, gera resposta.
        // A tool_call reconstruída usa a `consulta` do modelo (ou a fala bruta se malformada),
        // pois é o que o modelo "acha" que pediu — a coerência do prompt exige espelhá-la.
        // ---------------------------------------------------------------------
        emit(TurnEvent.StageChanged(PipelineStage.RESPONDING))
        val consultaForPrompt = modelConsulta.ifBlank { userQuestion }
        val nrForPrompt = modelNr.ifBlank { null }
        val toolContext = LfmToolRenderer.formatToolResult(chunks)

        val synthesisPrompt = LfmToolRenderer.renderSynthesisPrompt(
            systemPrompt = toolSystemPrompt,
            history = renderedHistory,
            userQuestion = userQuestion,
            consulta = consultaForPrompt,
            nr = nrForPrompt,
            toolContext = toolContext
        )

        val t2Start = System.currentTimeMillis()
        val answerBuilder = StringBuilder()
        var ttftMs = 0L
        var firstTokenReceived = false
        var completionTokens = 0

        try {
            llmEngine.generateFromPrompt(synthesisPrompt, maxTokens = synthesisMaxTokens).collect { chunk ->
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
            Log.e(TAG, "Erro na síntese do turno de tool", e)
            emit(TurnEvent.Error("Falha na síntese de resposta: ${e.message}", e))
            return@flow
        }

        val t2End = System.currentTimeMillis()
        val totalMs = t2End - totalStart
        val decodeMs = maxOf(0L, (t2End - t2Start) - ttftMs)
        val tokPerSec = if (decodeMs > 0) (completionTokens.toDouble() / (decodeMs / 1000.0)) else 0.0
        // A síntese pode, por engano, emitir uma nova tool_call: descartamos (1 busca/turno).
        val finalAnswer = stripToolResidue(answerBuilder.toString()).trim()

        val metrics = TurnMetrics(
            asrMs = asrMs,
            searchMs = searchMs,
            ttftMs = ttftMs,
            decodeMs = decodeMs,
            totalMs = totalMs,
            contextTokens = estimateTokens(synthesisPrompt),
            completionTokens = completionTokens,
            tokPerSec = Math.round(tokPerSec * 100.0) / 100.0,
            nrTop1 = decision?.top1 ?: "",
            nrTop1Prob = decision?.top1Prob ?: 0f,
            nrTop2 = decision?.top2 ?: "",
            gateMode = decision?.mode ?: "-",
            boostNrs = decision?.boostNrs?.joinToString(",").orEmpty(),
            chunksReused = false,
            retrievalMode = decision?.retrieval ?: "bm25",
            denseEncodeMs = if (retriever is HybridRetriever && decision?.retrieval == "rrf3")
                retriever.lastDenseEncodeMs else 0L,
            calledTool = true,
            toolConsulta = modelConsulta,
            toolNr = modelNr,
            reasonMs = reasonMs
        )

        emit(TurnEvent.Done(finalAnswer = finalAnswer, chunksUsed = chunks, metrics = metrics))
        logTurnSafely(history.size + 1, userQuestion, finalAnswer, chunks, metrics)
    }

    /**
     * Reconstrói o histórico no formato nativo (RenderedTurn) e poda a troca mais antiga até o
     * prompt estimado caber no orçamento (n_ctx 2048). O contexto normativo recente e a fala
     * atual são preservados; some sempre a troca MAIS ANTIGA primeiro.
     */
    fun buildRenderedHistory(
        history: List<ConversationTurn>,
        currentQuestion: String
    ): List<LfmToolRenderer.RenderedTurn> {
        val recent = history.takeLast(maxTurnsT2).toMutableList()
        // Poda por orçamento: estima o prompt (system + histórico + fala + folga da geração).
        val baseTokens = estimateTokens(toolSystemPrompt) + estimateTokens(currentQuestion) + 32
        while (recent.isNotEmpty()) {
            val histTokens = recent.sumOf { renderedTurnTokens(it) }
            if (baseTokens + histTokens <= maxPromptTokens) break
            recent.removeAt(0) // remove a troca MAIS ANTIGA
        }
        return recent.map { turn ->
            LfmToolRenderer.RenderedTurn(
                userQuestion = turn.question,
                calledTool = turn.calledTool,
                consulta = turn.toolConsulta,
                nr = turn.toolNr,
                toolContext = if (turn.calledTool) LfmToolRenderer.formatToolResult(turn.chunks) else "",
                answer = turn.answer
            )
        }
    }

    /** Estima os tokens de um turno reconstruído (fala + [tool_call + chunks] + resposta). */
    private fun renderedTurnTokens(turn: ConversationTurn): Int {
        var t = estimateTokens(turn.question) + estimateTokens(turn.answer)
        if (turn.calledTool) {
            t += estimateTokens(turn.toolConsulta ?: "") + 8
            t += turn.chunks.sumOf { estimateTokens(it.content) + 12 }
        }
        return t
    }

    /** Remove resíduo de tokens/sintaxe de tool-call de um texto de resposta. */
    fun stripToolResidue(text: String): String {
        var out = text
        for (mk in TOOL_MARKERS) {
            val idx = out.indexOf(mk)
            if (idx >= 0) out = out.substring(0, idx)
        }
        return out.replace("<|tool_call_end|>", "").replace("<|im_end|>", "")
    }

    private fun logTurnSafely(
        turnIndex: Int,
        question: String,
        answer: String,
        chunks: List<Chunk>,
        metrics: TurnMetrics
    ) {
        try {
            telemetryLogger?.logTurn(
                turnIndex = turnIndex,
                question = question,
                transcription = question,
                finalAnswer = answer,
                chunksUsed = chunks,
                metrics = metrics
            )
        } catch (e: Throwable) {
            Log.w(TAG, "Falha ao gravar telemetria", e)
        }
    }

    /**
     * Estimador rápido de contagem de tokens para controle de orçamento (1 token ~ 3.8 chars).
     */
    fun estimateTokens(text: String): Int {
        if (text.isBlank()) return 0
        return maxOf(1, (text.length / 3.8).toInt())
    }
}
