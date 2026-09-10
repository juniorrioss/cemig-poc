package br.org.ceia.cemigpoc.data.telemetry

import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ToolCall
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Registrador de telemetria local em formato JSONL (armazenado em filesDir/telemetry.jsonl).
 *
 * Implementado em Kotlin puro sem dependência de stubs do Android, garantindo
 * funcionamento idêntico na JVM de testes e no dispositivo Android em produção.
 */
class TelemetryLogger(
    private val telemetryFile: File
) {
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US)

    @Synchronized
    fun log(
        userQuestion: String,
        transcription: String,
        toolCallsMade: List<ToolCall>,
        chunksUsed: List<Chunk>,
        finalAnswer: String,
        totalDurationMs: Long
    ) {
        val toolCallsJson = toolCallsMade.joinToString(prefix = "[", postfix = "]") { call ->
            """{"tool":"${escape(call.toolName)}","query":"${escape(call.query)}"}"""
        }

        val chunksJson = chunksUsed.joinToString(prefix = "[", postfix = "]") { chunk ->
            """{"id":${chunk.id},"doc":"${escape(chunk.doc)}","section":"${escape(chunk.section)}","score":${chunk.score},"preview":"${escape(chunk.content.take(120))}"}"""
        }

        val jsonLine = """{"timestamp":"${dateFormat.format(Date())}","question":"${escape(userQuestion)}","transcription":"${escape(transcription)}","tool_calls":$toolCallsJson,"chunks_used":$chunksJson,"response":"${escape(finalAnswer)}","total_duration_ms":$totalDurationMs}"""

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

    /**
     * Retorna a quantidade de registros gravados no arquivo.
     */
    fun getLogCount(): Int {
        if (!telemetryFile.exists()) return 0
        return telemetryFile.readLines().count { it.isNotBlank() }
    }
}
