package br.org.ceia.cemigpoc.data.telemetry

import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Registrador de telemetria local em formato JSONL (armazenado em filesDir/telemetry.jsonl).
 *
 * Registra breakdown de cada etapa: ASR, Rewrite (T1), Busca BM25, Prefill/TTFT e Decode (T2).
 */
class TelemetryLogger(
    private val telemetryFile: File
) {
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US)

    @Synchronized
    fun logTurn(
        turnIndex: Int,
        question: String,
        transcription: String,
        finalAnswer: String,
        chunksUsed: List<Chunk>,
        metrics: TurnMetrics
    ) {
        val chunksJson = chunksUsed.joinToString(prefix = "[", postfix = "]") { chunk ->
            """{"id":${chunk.id},"doc":"${escape(chunk.doc)}","section":"${escape(chunk.section)}","score":${chunk.score},"title":"${escape(chunk.title)}"}"""
        }

        val jsonLine = buildString {
            append("{")
            append("\"timestamp\":\"${dateFormat.format(Date())}\",")
            append("\"turn_index\":$turnIndex,")
            append("\"question\":\"${escape(question)}\",")
            append("\"transcription\":\"${escape(transcription)}\",")
            append("\"keywords\":\"${escape(metrics.keywords)}\",")
            append("\"keywords_reused\":${metrics.keywordsReused},")
            append("\"chunks_used\":$chunksJson,")
            append("\"response\":\"${escape(finalAnswer)}\",")
            append("\"asr_ms\":${metrics.asrMs},")
            append("\"rewrite_ms\":${metrics.rewriteMs},")
            append("\"search_ms\":${metrics.searchMs},")
            append("\"ttft_ms\":${metrics.ttftMs},")
            append("\"decode_ms\":${metrics.decodeMs},")
            append("\"total_duration_ms\":${metrics.totalMs},")
            append("\"context_tokens\":${metrics.contextTokens},")
            append("\"completion_tokens\":${metrics.completionTokens},")
            append("\"tok_per_sec\":${metrics.tokPerSec}")
            append("}")
        }

        telemetryFile.parentFile?.mkdirs()
        FileWriter(telemetryFile, true).use { writer ->
            writer.appendLine(jsonLine)
        }
    }

    private fun escape(s: String): String {
        return s.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\b", "\\b")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    }

    fun getLogCount(): Int {
        if (!telemetryFile.exists()) return 0
        return telemetryFile.readLines().count { it.isNotBlank() }
    }
}
