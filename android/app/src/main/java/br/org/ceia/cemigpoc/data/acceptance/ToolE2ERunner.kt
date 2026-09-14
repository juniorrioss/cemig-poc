package br.org.ceia.cemigpoc.data.acceptance

import android.content.Context
import android.os.Debug
import android.os.PowerManager
import android.util.Log
import br.org.ceia.cemigpoc.BuildConfig
import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import br.org.ceia.cemigpoc.data.engine.RealLlamaEngine
import br.org.ceia.cemigpoc.data.model.ModelFileManager
import br.org.ceia.cemigpoc.data.retriever.DenseRetriever
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.llama.LlamaEmbedder
import br.org.ceia.cemigpoc.domain.pipeline.LfmToolCallParser
import br.org.ceia.cemigpoc.domain.pipeline.LfmToolRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Runner de TESTE E2E do pipeline HÍBRIDO de tool-calling NO APARELHO (task poc-app-tools,
 * Parte 2). O capitão quer PROVA de que o fluxo está correto — não afirmação. Este runner
 * dirige o MESMO motor (1.2B tool Q4), retriever (RRF v4) e renderizador (paridade byte-a-byte)
 * do app, e grava o LOG BRUTO de cada caso: prompt renderizado (recortado), tool_call literal,
 * chunks retornados e resposta final; além de TTFT/decode/latência e RAM PSS.
 *
 * Dispara via:
 *   adb shell am broadcast -a br.org.ceia.cemigpoc.RUN_TOOL_E2E
 * Saída: getExternalFilesDir(null)/tool_e2e_results.json  e no logcat (tag CEMIG_TOOL_E2E).
 *
 * Casos (cobrem a exigência do brief a–e + os casos do capitão):
 *  A. pergunta que DEVE buscar (13,8 kV)                -> tool_call + busca externa
 *  B. followup do MESMO assunto                          -> reuso (SEM nova busca)
 *  C. mudança de assunto (poste)                         -> nova busca com chunks diferentes
 *  D. saudação/agradecimento                             -> ZERO busca
 *  E. contexto insuficiente (camiseta rasgada)           -> recusa/delimitação, sem invenção
 *
 * O runner é single-turn por caso, EXCETO B, que reusa o histórico de A (multiturno real).
 *
 * Comentários PT-BR; identificadores em inglês.
 */
object ToolE2ERunner {

    private const val TAG = "CEMIG_TOOL_E2E"
    private const val DECISION_MAX_TOKENS = 220
    private const val SYNTHESIS_MAX_TOKENS = 384

    data class E2ECase(
        val id: String,
        val label: String,
        val question: String,
        val reuseFrom: String? = null // id do caso anterior cujo histórico é injetado
    )

    val CASES = listOf(
        E2ECase("A_deve_buscar", "Pergunta técnica -> DEVE buscar", "Qual a distância segura pra trabalhar perto de rede de 13,8 kV?"),
        E2ECase("B_followup_reuso", "Followup mesmo assunto -> REUSO", "E se eu não conseguir desligar a rede?", reuseFrom = "A_deve_buscar"),
        E2ECase("C_muda_assunto", "Mudança de assunto -> NOVA busca", "Beleza. E o poste que não parece firme, posso subir?"),
        E2ECase("D_saudacao", "Saudação -> ZERO busca", "Valeu, muito obrigado pela ajuda!"),
        E2ECase("E_recusa", "Contexto insuficiente -> RECUSA", "Posso trabalhar de camiseta de algodão rasgada no serviço?")
    )

    data class CaseResult(
        val id: String,
        val label: String,
        val question: String,
        val calledTool: Boolean,
        val toolCallLiteral: String,
        val consulta: String,
        val nr: String,
        val chunks: List<String>,
        val chunksDetail: String,
        val decisionPromptTail: String,
        val synthesisPromptHead: String,
        val finalAnswer: String,
        val reasonMs: Long,
        val searchMs: Long,
        val ttftMs: Long,
        val decodeMs: Long,
        val totalMs: Long,
        val completionTokens: Int
    )

    suspend fun run(context: Context): List<CaseResult> = withContext(Dispatchers.IO) {
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val wake = pm?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cemigpoc:ToolE2EWake")?.apply {
            acquire(20 * 60 * 1000L)
        }
        try {
            Log.i(TAG, "==================== E2E TOOL-CALLING HÍBRIDO ====================")
            Log.i(TAG, "Sintetizador=${ModelFileManager.LLM_MODEL_NAME} ASR=${BuildConfig.ASR_ENGINE}")

            val fm = ModelFileManager(context)
            val dbFile = fm.getFts5DatabaseFile()
            val llmFile = fm.getLlmModelFile()
            val embedFile = fm.getEmbedModelFile()

            val toolSystemPrompt = runCatching {
                context.assets.open("tools_system_prompt.txt").bufferedReader(Charsets.UTF_8).use { it.readText() }
            }.getOrElse {
                Log.e(TAG, "Falha ao ler tools_system_prompt.txt", it); ""
            }
            Log.i(TAG, "System prompt tool: ${toolSystemPrompt.length} chars (esperado 2381)")

            // Retriever RRF v4 (idêntico ao app).
            val embedder = LlamaEmbedder().also {
                if (!it.load(embedFile.absolutePath)) Log.w(TAG, "Encoder denso não carregou")
            }
            val dense = DenseRetriever(context, embedder, embedModelPath = embedFile.absolutePath).also { it.load() }
            val retriever = HybridRetriever(
                delegate = Fts5Retriever(context, dbFile),
                classifier = NrClassifier.load(context),
                denseRetriever = dense
            )

            val llama = RealLlamaEngine().also {
                it.engine.suppressReasoning = BuildConfig.SUPPRESS_REASONING
                it.engine.load(llmFile.absolutePath, ctxSize = 2048, nThreads = 6)
            }

            val results = mutableListOf<CaseResult>()
            // Histórico reconstruído por caso (para o reuso multiturno de B).
            val historyById = mutableMapOf<String, LfmToolRenderer.RenderedTurn>()

            for (c in CASES) {
                Log.i(TAG, "-----------------------------------------------------------------")
                Log.i(TAG, "[${c.id}] ${c.label}")
                Log.i(TAG, "  PERGUNTA: ${c.question}")

                val history = c.reuseFrom?.let { historyById[it]?.let { t -> listOf(t) } } ?: emptyList()
                if (history.isEmpty()) llama.clearKvCache()

                val totalStart = System.currentTimeMillis()

                // ---------- TURNO 1: DECISÃO ----------
                val decisionPrompt = LfmToolRenderer.renderDecisionPrompt(toolSystemPrompt, history, c.question)
                val decisionStart = System.currentTimeMillis()
                val decisionBuf = StringBuilder()
                llama.generateFromPrompt(decisionPrompt, DECISION_MAX_TOKENS).collect { ch ->
                    if (ch is LlmResponseChunk.Text) decisionBuf.append(ch.delta)
                }
                val decisionRaw = decisionBuf.toString()
                val reasonMs = System.currentTimeMillis() - decisionStart

                val looksLikeCall = LfmToolCallParser.looksLikeToolCall(decisionRaw)
                val calls = if (looksLikeCall) LfmToolCallParser.parse(decisionRaw) else emptyList()
                val validCall = calls.firstOrNull { it.name == LfmToolCallParser.TOOL_NAME }
                val wantsSearch = validCall != null || looksLikeCall
                val consulta = validCall?.arguments?.get("consulta")?.trim().orEmpty()
                val nr = validCall?.arguments?.get("nr")?.trim().orEmpty()
                val toolCallLiteral = decisionRaw.trim().replace("\n", "\\n").take(300)

                Log.i(TAG, "  DECISÃO: buscar=$wantsSearch (${reasonMs}ms)")
                Log.i(TAG, "  TOOL_CALL LITERAL: $toolCallLiteral")

                var searchMs = 0L
                var chunks: List<Chunk> = emptyList()
                var ttftMs = 0L
                var decodeMs = 0L
                var completionTokens = 0
                var finalAnswer: String
                var synthesisPromptHead = ""

                if (wantsSearch) {
                    // ---------- BUSCA EXTERNA (IGNORA a consulta do modelo; usa fala bruta) ----------
                    val searchStart = System.currentTimeMillis()
                    chunks = try {
                        if (retriever.hasDense) retriever.searchV3(c.question, 2)
                        else retriever.searchWithRaw(c.question, c.question, 2)
                    } catch (e: Throwable) { Log.e(TAG, "busca falhou", e); emptyList() }
                    searchMs = System.currentTimeMillis() - searchStart
                    val d = retriever.lastDecision
                    Log.i(TAG, "  BUSCA(fala bruta): ${chunks.size} chunks em ${searchMs}ms · gate=${d?.mode} top1=${d?.top1}(${d?.top1Prob})")
                    chunks.forEachIndexed { i, ch -> Log.i(TAG, "    chunk[$i] ${ch.doc} ${ch.section} :: ${ch.content.take(90)}") }

                    // ---------- TURNO 2: SÍNTESE ----------
                    val consultaForPrompt = consulta.ifBlank { c.question }
                    val nrForPrompt = nr.ifBlank { null }
                    val toolCtx = LfmToolRenderer.formatToolResult(chunks)
                    val synthesisPrompt = LfmToolRenderer.renderSynthesisPrompt(
                        toolSystemPrompt, history, c.question, consultaForPrompt, nrForPrompt, toolCtx
                    )
                    synthesisPromptHead = synthesisPrompt.substringAfter("<|im_start|>user").take(600)
                    val t2Start = System.currentTimeMillis()
                    val ansBuf = StringBuilder()
                    var first = false
                    llama.generateFromPrompt(synthesisPrompt, SYNTHESIS_MAX_TOKENS).collect { ch ->
                        if (ch is LlmResponseChunk.Text) {
                            if (!first) { first = true; ttftMs = System.currentTimeMillis() - t2Start }
                            completionTokens++
                            ansBuf.append(ch.delta)
                        }
                    }
                    val t2End = System.currentTimeMillis()
                    decodeMs = maxOf(0L, (t2End - t2Start) - ttftMs)
                    finalAnswer = stripResidue(ansBuf.toString()).trim()
                } else {
                    // reuso/saudação: a saída da decisão É a resposta.
                    ttftMs = reasonMs
                    finalAnswer = stripResidue(decisionRaw).trim()
                    Log.i(TAG, "  SEM BUSCA (reuso/direto)")
                }

                val totalMs = System.currentTimeMillis() - totalStart
                Log.i(TAG, "  RESPOSTA: $finalAnswer")
                Log.i(TAG, "  TEMPOS: reason=${reasonMs}ms search=${searchMs}ms ttft=${ttftMs}ms decode=${decodeMs}ms TOTAL=${totalMs}ms (${completionTokens} tok)")

                // Registra o turno para reuso multiturno (histórico nativo).
                historyById[c.id] = LfmToolRenderer.RenderedTurn(
                    userQuestion = c.question,
                    calledTool = wantsSearch,
                    consulta = consulta.ifBlank { c.question },
                    nr = nr.ifBlank { null },
                    toolContext = if (wantsSearch) LfmToolRenderer.formatToolResult(chunks) else "",
                    answer = finalAnswer
                )

                results.add(
                    CaseResult(
                        id = c.id, label = c.label, question = c.question,
                        calledTool = wantsSearch, toolCallLiteral = toolCallLiteral,
                        consulta = consulta, nr = nr,
                        chunks = chunks.map { "${it.doc} ${it.section}" },
                        chunksDetail = chunks.joinToString(" || ") { "${it.doc} ${it.section}: ${it.content.take(120)}" },
                        decisionPromptTail = decisionPrompt.takeLast(280),
                        synthesisPromptHead = synthesisPromptHead,
                        finalAnswer = finalAnswer,
                        reasonMs = reasonMs, searchMs = searchMs, ttftMs = ttftMs,
                        decodeMs = decodeMs, totalMs = totalMs, completionTokens = completionTokens
                    )
                )
            }

            // RAM PSS (com Nemotron + 1.2B residentes).
            val mi = Debug.MemoryInfo()
            Debug.getMemoryInfo(mi)
            val pssMb = mi.totalPss / 1024.0
            Log.i(TAG, "RAM PSS total: %.1f MB".format(pssMb))

            saveJson(context, results, pssMb)
            llama.engine.close()
            embedder.close()
            Log.i(TAG, "==================== E2E CONCLUÍDO (${results.size} casos) ====================")
            results
        } finally {
            try { if (wake?.isHeld == true) wake.release() } catch (_: Exception) {}
        }
    }

    private fun stripResidue(text: String): String {
        var out = text
        for (mk in listOf("<|tool_call_start|>", "buscar_norma(")) {
            val idx = out.indexOf(mk); if (idx >= 0) out = out.substring(0, idx)
        }
        return out.replace("<|tool_call_end|>", "").replace("<|im_end|>", "")
    }

    private fun saveJson(context: Context, results: List<CaseResult>, pssMb: Double) {
        val f = context.getExternalFilesDir(null)?.resolve("tool_e2e_results.json")
            ?: File(context.filesDir, "tool_e2e_results.json")
        val json = buildString {
            append("{\n")
            append("  \"timestamp\": \"${SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date())}\",\n")
            append("  \"device\": \"Samsung Galaxy S24+ (SM-S926B, Exynos 2400)\",\n")
            append("  \"synthesizer\": \"${esc(ModelFileManager.LLM_MODEL_NAME)}\",\n")
            append("  \"asr_engine\": \"${esc(BuildConfig.ASR_ENGINE)}\",\n")
            append("  \"ram_pss_mb\": ${"%.1f".format(Locale.US, pssMb)},\n")
            append("  \"cases\": [\n")
            results.forEachIndexed { i, r ->
                append("    {\n")
                append("      \"id\": \"${esc(r.id)}\",\n")
                append("      \"label\": \"${esc(r.label)}\",\n")
                append("      \"question\": \"${esc(r.question)}\",\n")
                append("      \"called_tool\": ${r.calledTool},\n")
                append("      \"tool_call_literal\": \"${esc(r.toolCallLiteral)}\",\n")
                append("      \"consulta\": \"${esc(r.consulta)}\",\n")
                append("      \"nr\": \"${esc(r.nr)}\",\n")
                append("      \"chunks\": [${r.chunks.joinToString(",") { "\"${esc(it)}\"" }}],\n")
                append("      \"chunks_detail\": \"${esc(r.chunksDetail)}\",\n")
                append("      \"decision_prompt_tail\": \"${esc(r.decisionPromptTail)}\",\n")
                append("      \"synthesis_prompt_head\": \"${esc(r.synthesisPromptHead)}\",\n")
                append("      \"final_answer\": \"${esc(r.finalAnswer)}\",\n")
                append("      \"reason_ms\": ${r.reasonMs},\n")
                append("      \"search_ms\": ${r.searchMs},\n")
                append("      \"ttft_ms\": ${r.ttftMs},\n")
                append("      \"decode_ms\": ${r.decodeMs},\n")
                append("      \"total_ms\": ${r.totalMs},\n")
                append("      \"completion_tokens\": ${r.completionTokens}\n")
                append("    }${if (i < results.size - 1) "," else ""}\n")
            }
            append("  ]\n")
            append("}\n")
        }
        f.writeText(json)
        Log.i(TAG, "Resultados E2E salvos em: ${f.absolutePath}")
    }

    private fun esc(s: String): String = s
        .replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
}
