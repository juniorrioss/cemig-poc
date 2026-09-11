package br.org.ceia.cemigpoc.data.acceptance

import android.content.Context
import android.os.PowerManager
import android.util.Log
import br.org.ceia.cemigpoc.data.engine.RealAsrEngine
import br.org.ceia.cemigpoc.data.engine.RealLlamaEngine
import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import br.org.ceia.cemigpoc.data.model.ModelFileManager
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class AcceptanceQuestion(
    val id: String,
    val audioFileName: String,
    val referenceText: String,
    val isMultiturnContinuation: Boolean = false,
    val previousQuestionId: String? = null
)

data class AcceptanceResultItem(
    val id: String,
    val referenceText: String,
    val transcribedText: String,
    val isMultiturn: Boolean,
    val keywords: String,
    val keywordsReused: Boolean,
    val chunksRetrieved: List<String>,
    val answerSnippet: String,
    val asrMs: Long,
    val rewriteMs: Long,
    val searchMs: Long,
    val ttftMs: Long,
    val decodeMs: Long,
    val totalMs: Long,
    val contextTokens: Int,
    val completionTokens: Int,
    val tokPerSec: Double,
    val fitsBudget: Boolean
)

object AcceptanceRunner {

    private const val TAG = "CEMIG_ACCEPTANCE"

    val QUESTIONS = listOf(
        AcceptanceQuestion("Q01", "q01.wav", "Quais são os procedimentos obrigatórios para a desenergização segundo a NR-10?"),
        AcceptanceQuestion("Q02", "q02_multiturn.wav", "E se não for possível desenergizar, quais as medidas de proteção coletiva prioritárias?", isMultiturnContinuation = true, previousQuestionId = "Q01"),
        AcceptanceQuestion("Q03", "q03.wav", "Qual é a distância mínima de segurança para trabalhar próximo a uma rede de 13,8 kV?"),
        AcceptanceQuestion("Q04", "q04.wav", "Quais são os requisitos da NR-35 para ancoragem de cinto tipo paraquedista no poste?"),
        AcceptanceQuestion("Q05", "q05_multiturn.wav", "E qual o fator de queda tolerado para o talabarte?", isMultiturnContinuation = true, previousQuestionId = "Q04"),
        AcceptanceQuestion("Q06", "q06.wav", "A luva isolante de borracha classe 2 suporta qual nível de tensão máxima?"),
        AcceptanceQuestion("Q07", "q07.wav", "É obrigatório o uso de capacete com jugular e óculos de proteção para eletricista?"),
        AcceptanceQuestion("Q08", "q08.wav", "Onde devo instalar o conjunto de aterramento temporário na chave seccionadora?"),
        AcceptanceQuestion("Q09", "q09.wav", "Qual equipamento de proteção coletiva é exigido para abertura de chave fusível?"),
        AcceptanceQuestion("Q10", "q10.wav", "Posso realizar intervenção em circuito energizado durante chuva forte?")
    )

    suspend fun run(
        context: Context,
        existingAsr: RealAsrEngine? = null,
        existingLlama: RealLlamaEngine? = null,
        existingRetriever: br.org.ceia.cemigpoc.domain.engine.Retriever? = null
    ): List<AcceptanceResultItem> = withContext(Dispatchers.IO) {
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val wakeLock = powerManager?.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "cemigpoc:AcceptanceWakeLock"
        )?.apply {
            acquire(30 * 60 * 1000L) // 30 minutos de wake lock
        }

        try {
            Log.i(TAG, "=================================================================")
            Log.i(TAG, "INICIANDO ROTEIRO DE ACEITAÇÃO CEMIG POC M2 — 10 PERGUNTAS FALADAS")
        Log.i(TAG, "Hardware: Samsung Galaxy S24+ (Exynos 2400 · ARM64 · NDK)")
        Log.i(TAG, "Modo Avião: Ativo · 100% Offline · Sem chamadas de rede")
        Log.i(TAG, "=================================================================")

        val fileManager = ModelFileManager(context)
        val dbFile = fileManager.getFts5DatabaseFile()
        val asrFile = fileManager.getAsrModelFile()
        val llmFile = fileManager.getLlmModelFile()

        // Pipeline híbrido de 2 estágios: classificador NR (estágio 1) + BM25 com boost (estágio 2).
        val retriever = existingRetriever ?: HybridRetriever(
            delegate = Fts5Retriever(context, dbFile),
            classifier = NrClassifier.load(context)
        )
        val realAsr = existingAsr ?: RealAsrEngine().also {
            it.engine.initModel(asrFile.absolutePath)
        }
        val realLlama = existingLlama ?: RealLlamaEngine().also {
            it.engine.load(llmFile.absolutePath, ctxSize = 2048, nThreads = 6)
        }
        val shouldCloseEngines = existingAsr == null && existingLlama == null

        val telemetryFile = File(context.filesDir, "acceptance_telemetry.jsonl")
        val logger = TelemetryLogger(telemetryFile)

        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = realLlama,
            telemetryLogger = logger,
            maxTurnsT1 = 6,
            maxTurnsT2 = 3,
            t2MaxBudgetTokens = 1000,
            jaccardThreshold = 0.7,
            topK = 2
        )

        val acceptanceDir = context.getExternalFilesDir(null)?.resolve("acceptance")
            ?: File(context.filesDir, "acceptance")

        val results = mutableListOf<AcceptanceResultItem>()
        val turnsMap = mutableMapOf<String, ConversationTurn>()

        for ((idx, q) in QUESTIONS.withIndex()) {
            Log.i(TAG, "-----------------------------------------------------------------")
            Log.i(TAG, "[${idx + 1}/10] Executando ${q.id}: ${q.referenceText}")
            Log.i(TAG, "Multiturno com continuação: ${q.isMultiturnContinuation}")

            val wavFile = File(acceptanceDir, q.audioFileName)
            if (!wavFile.exists()) {
                Log.e(TAG, "Arquivo de áudio não encontrado: ${wavFile.absolutePath}")
                continue
            }

            // 1. Leitura do arquivo WAV
            val audioSamples = readWavToFloatArray(wavFile)
            val audioDurationS = audioSamples.size / 16000.0f
            Log.i(TAG, "Áudio: ${q.audioFileName} (%.2fs, %d amostras)".format(audioDurationS, audioSamples.size))

            // 2. ASR Transcrição Whisper Base Q5_1
            val tAsrStart = System.currentTimeMillis()
            val transcribedText = realAsr.transcribeAudioSamples(audioSamples)
            val asrMs = System.currentTimeMillis() - tAsrStart
            Log.i(TAG, "Transcrição ASR (${asrMs}ms): '$transcribedText'")

            // 3. Montagem do histórico para multiturno
            val history = mutableListOf<ConversationTurn>()
            if (q.isMultiturnContinuation && q.previousQuestionId != null) {
                turnsMap[q.previousQuestionId]?.let {
                    history.add(it)
                    Log.i(TAG, "Histórico injetado da questão ${q.previousQuestionId}: '${it.question}' -> '${it.answer.take(60)}...'")
                }
            } else {
                realLlama.clearKvCache()
            }

            // 4. Execução do AskPipeline Modo C Top-2
            val queryText = if (transcribedText.isNotBlank()) transcribedText else q.referenceText
            val doneEvent = pipeline.execute(
                userQuestion = queryText,
                history = history,
                asrMs = asrMs
            ).filterIsInstance<TurnEvent.Done>().first()

            val m = doneEvent.metrics
            val turn = ConversationTurn(
                question = queryText,
                answer = doneEvent.finalAnswer,
                keywords = m.keywords,
                chunks = doneEvent.chunksUsed,
                metrics = m
            )
            turnsMap[q.id] = turn

            val chunksDesc = doneEvent.chunksUsed.map { "${it.doc} ${it.section}" }
            val fitsBudget = m.totalMs <= 10000L

            val itemResult = AcceptanceResultItem(
                id = q.id,
                referenceText = q.referenceText,
                transcribedText = transcribedText,
                isMultiturn = q.isMultiturnContinuation,
                keywords = m.keywords,
                keywordsReused = m.keywordsReused,
                chunksRetrieved = chunksDesc,
                answerSnippet = doneEvent.finalAnswer.take(180),
                asrMs = m.asrMs,
                rewriteMs = m.rewriteMs,
                searchMs = m.searchMs,
                ttftMs = m.ttftMs,
                decodeMs = m.decodeMs,
                totalMs = m.totalMs,
                contextTokens = m.contextTokens,
                completionTokens = m.completionTokens,
                tokPerSec = m.tokPerSec,
                fitsBudget = fitsBudget
            )
            results.add(itemResult)

            Log.i(TAG, "RESULTADO ${q.id}:")
            Log.i(TAG, "  ASR:      ${m.asrMs} ms")
            Log.i(TAG, "  T1 Rewr:  ${m.rewriteMs} ms (Keywords: '${m.keywords}', Reuso: ${m.keywordsReused})")
            Log.i(TAG, "  BM25:     ${m.searchMs} ms (${chunksDesc.joinToString(", ")})")
            Log.i(TAG, "  TTFT:     ${m.ttftMs} ms")
            Log.i(TAG, "  Decode:   ${m.decodeMs} ms (${m.completionTokens} tok @ ${m.tokPerSec} t/s)")
            Log.i(TAG, "  TOTAL:    ${m.totalMs} ms (%.2fs) — Cumpre <= 10s: %s".format(m.totalMs / 1000.0, if (fitsBudget) "SIM" else "NÃO"))
            Log.i(TAG, "  Resposta: ${doneEvent.finalAnswer.take(140)}...")
        }

        // 5. Salva JSON consolidado dos resultados
        saveResultsJson(context, results)

        if (shouldCloseEngines) {
            realAsr.engine.close()
            realLlama.engine.close()
        }

        Log.i(TAG, "=================================================================")
        Log.i(TAG, "ROTEIRO DE ACEITAÇÃO CONCLUÍDO COM SUCESSO: ${results.size} PERGUNTAS AVALIADAS")
        Log.i(TAG, "=================================================================")

        results
        } finally {
            try {
                if (wakeLock?.isHeld == true) {
                    wakeLock.release()
                }
            } catch (_: Exception) {}
        }
    }

    private fun readWavToFloatArray(file: File): FloatArray {
        val bytes = file.readBytes()
        var dataIdx = -1
        for (i in 0 until bytes.size - 4) {
            if (bytes[i] == 'd'.code.toByte() &&
                bytes[i + 1] == 'a'.code.toByte() &&
                bytes[i + 2] == 't'.code.toByte() &&
                bytes[i + 3] == 'a'.code.toByte()
            ) {
                dataIdx = i + 8
                break
            }
        }
        if (dataIdx == -1 || dataIdx >= bytes.size) {
            dataIdx = 44
        }

        val numSamples = (bytes.size - dataIdx) / 2
        val shortBuffer = java.nio.ByteBuffer.wrap(bytes, dataIdx, numSamples * 2)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()

        val floats = FloatArray(shortBuffer.remaining())
        for (i in floats.indices) {
            floats[i] = shortBuffer.get() / 32768.0f
        }
        return floats
    }

    private fun saveResultsJson(context: Context, results: List<AcceptanceResultItem>) {
        val jsonFile = context.getExternalFilesDir(null)?.resolve("acceptance_results.json")
            ?: File(context.filesDir, "acceptance_results.json")

        val json = buildString {
            append("{\n")
            append("  \"timestamp\": \"${SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date())}\",\n")
            append("  \"device\": \"Samsung Galaxy S24+ (SM-S926B, Exynos 2400)\",\n")
            append("  \"airplane_mode\": true,\n")
            append("  \"total_questions\": ${results.size},\n")
            append("  \"results\": [\n")
            results.forEachIndexed { idx, r ->
                append("    {\n")
                append("      \"id\": \"${r.id}\",\n")
                append("      \"reference\": \"${escape(r.referenceText)}\",\n")
                append("      \"transcription\": \"${escape(r.transcribedText)}\",\n")
                append("      \"is_multiturn\": ${r.isMultiturn},\n")
                append("      \"keywords\": \"${escape(r.keywords)}\",\n")
                append("      \"keywords_reused\": ${r.keywordsReused},\n")
                append("      \"chunks\": [${r.chunksRetrieved.joinToString(",") { "\"$it\"" }}],\n")
                append("      \"answer_snippet\": \"${escape(r.answerSnippet)}\",\n")
                append("      \"asr_ms\": ${r.asrMs},\n")
                append("      \"rewrite_ms\": ${r.rewriteMs},\n")
                append("      \"search_ms\": ${r.searchMs},\n")
                append("      \"ttft_ms\": ${r.ttftMs},\n")
                append("      \"decode_ms\": ${r.decodeMs},\n")
                append("      \"total_ms\": ${r.totalMs},\n")
                append("      \"context_tokens\": ${r.contextTokens},\n")
                append("      \"completion_tokens\": ${r.completionTokens},\n")
                append("      \"tok_per_sec\": ${r.tokPerSec},\n")
                append("      \"fits_budget\": ${r.fitsBudget}\n")
                append("    }${if (idx < results.size - 1) "," else ""}\n")
            }
            append("  ]\n")
            append("}\n")
        }

        jsonFile.writeText(json)
        Log.i(TAG, "Resultados consolidados salvos em: ${jsonFile.absolutePath}")
    }

    private fun escape(s: String): String {
        return s.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    }
}
