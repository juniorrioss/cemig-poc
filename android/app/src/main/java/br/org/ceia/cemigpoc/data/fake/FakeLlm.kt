package br.org.ceia.cemigpoc.data.fake

import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.domain.model.ToolCall
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

enum class FakeLlmMode {
    AUTO,
    FORCE_TOOL,
    FORCE_NO_TOOL
}

/**
 * Mock configurável do SLM com suporte a Modo C Top-2 (Rewrite + Síntese com Citação).
 */
class FakeLlm(
    private val mode: FakeLlmMode = FakeLlmMode.AUTO,
    private val streamDelayMs: Long = 0L,
    var customRewriteOutput: String? = null,
    var customSynthesisOutput: String? = null
) : LlmEngine {

    override fun streamChat(
        messages: List<Message>,
        systemPrompt: String
    ): Flow<LlmResponseChunk> = flow {
        // 1. Turno 1: Extração / Query Rewrite de Palavras-Chave
        if (systemPrompt == AskPipeline.REWRITE_SYSTEM_PROMPT) {
            val userContent = messages.lastOrNull { it.role == Message.Role.USER }?.content ?: ""
            val output = customRewriteOutput ?: reformulateKeywords(userContent)
            emitStreamingText(output)
            return@flow
        }

        // 2. Turno 2: Síntese de resposta baseada no contexto e histórico
        if (systemPrompt == AskPipeline.SYNTHESIS_SYSTEM_PROMPT) {
            val userContent = messages.lastOrNull { it.role == Message.Role.USER }?.content ?: ""
            val responseText = customSynthesisOutput ?: synthesizeAnswer(userContent)
            emitStreamingText(responseText)
            return@flow
        }

        // Fallback genérico para compatibilidade
        val userContent = messages.lastOrNull { it.role == Message.Role.USER }?.content ?: ""
        emitStreamingText(synthesizeAnswer(userContent))
    }

    private suspend fun kotlinx.coroutines.flow.FlowCollector<LlmResponseChunk>.emitStreamingText(text: String) {
        val tokens = text.split(" ")
        for (i in tokens.indices) {
            val token = if (i < tokens.size - 1) "${tokens[i]} " else tokens[i]
            emit(LlmResponseChunk.Text(token))
            if (streamDelayMs > 0) {
                delay(streamDelayMs)
            }
        }
    }

    fun reformulateKeywords(prompt: String): String {
        val q = prompt.lowercase()
        return when {
            q.contains("desenergiz") -> "NR-10 desenergizacao etapas seccionamento aterramento"
            q.contains("altura") || q.contains("nr-35") || q.contains("queda") -> "NR-35 trabalho altura protecao cinturão"
            q.contains("epi") || q.contains("luva") || q.contains("capacete") -> "NR-06 EPI equipamento protecao individual"
            q.contains("distancia") || q.contains("13,8") || q.contains("kv") -> "NR-10 delimitacao zonas risco controlada"
            else -> "NR-10 seguranca instalacoes eletricas procedimentos"
        }
    }

    fun synthesizeAnswer(userContent: String): String {
        return when {
            userContent.contains("Nenhum contexto normativo recuperado", ignoreCase = true) ->
                "Não sei com base nas normas consultadas. Nenhuma informação foi localizada para o procedimento solicitado."
            userContent.contains("NR-35", ignoreCase = true) ->
                "De acordo com a NR-35, item 35.5.1, todo trabalho em altura acima de dois metros deve utilizar sistema de proteção contra quedas com cinturão tipo paraquedista e ponto de ancoragem seguro."
            userContent.contains("NR-06", ignoreCase = true) ->
                "Conforme a NR-06, item 6.3, a empresa é obrigada a fornecer aos empregados, gratuitamente, EPI adequado ao risco, em perfeito estado de conservação e funcionamento."
            else ->
                "Conforme estabelece expressamente a NR-10, item 10.5.1, somente serão consideradas desenergizadas as instalações elétricas liberadas para trabalho mediante a sequência: seccionamento, impedimento de reenergização, constatação de ausência de tensão, instalação de aterramento temporário, proteção dos elementos energizados e sinalização de impedimento."
        }
    }
}
