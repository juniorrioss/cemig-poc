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
 * Mock configurável do SLM com suporte à síntese com citação (pipeline consolidado, sem T1).
 */
class FakeLlm(
    private val mode: FakeLlmMode = FakeLlmMode.AUTO,
    private val streamDelayMs: Long = 0L,
    var customSynthesisOutput: String? = null
) : LlmEngine {

    override fun streamChat(
        messages: List<Message>,
        systemPrompt: String
    ): Flow<LlmResponseChunk> = flow {
        // Síntese de resposta baseada no contexto e histórico (único turno de LLM restante).
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

    /**
     * Simula a geração a partir de prompt nativo pré-renderizado (pipeline de tool-calling).
     *
     * Heurística do mock (determinística, para os testes JVM do laço híbrido):
     * - Prompt de SÍNTESE (contém um turno `tool`): sintetiza a resposta a partir dos chunks.
     * - Prompt de DECISÃO (termina numa fala do usuário): decide chamar a ferramenta OU
     *   responder direto conforme `mode` e o conteúdo da última fala (saudação/agradecimento
     *   -> sem tool). A tool_call é emitida no formato nativo do LFM2.5.
     */
    override fun generateFromPrompt(
        prompt: String,
        maxTokens: Int
    ): Flow<LlmResponseChunk> = flow {
        // Prompt de síntese: já tem o turno `tool` com o contexto injetado.
        if (prompt.contains("<|im_start|>tool")) {
            val toolCtx = prompt.substringAfterLast("<|im_start|>tool\n").substringBefore("<|im_end|>")
            emitStreamingText(synthesizeAnswer(toolCtx))
            return@flow
        }

        // Prompt de decisão: a última fala do usuário está entre o último user e o assistant final.
        val lastUser = prompt.substringAfterLast("<|im_start|>user\n").substringBefore("<|im_end|>").trim()
        val decideCall = when (mode) {
            FakeLlmMode.FORCE_TOOL -> true
            FakeLlmMode.FORCE_NO_TOOL -> false
            FakeLlmMode.AUTO -> !isSmallTalk(lastUser)
        }
        if (decideCall) {
            // Emite a tool_call nativa; a `consulta` do mock imita a reescrita do modelo.
            val call = "<|tool_call_start|>[buscar_norma(consulta='${lastUser.replace("'", "\\'")}')]<|tool_call_end|>"
            emit(LlmResponseChunk.Text(call))
        } else {
            // Resposta direta (reuso/saudação): sem busca.
            emitStreamingText(directReply(lastUser))
        }
    }

    private fun isSmallTalk(text: String): Boolean {
        val t = text.lowercase()
        return listOf("bom dia", "boa tarde", "boa noite", "obrigad", "valeu", "tudo bem",
            "tudo certo", "ok", "beleza").any { t.contains(it) }
    }

    private fun directReply(text: String): String {
        return if (isSmallTalk(text)) {
            "Tudo certo, estou aqui pra ajudar com as normas de segurança. Pode mandar sua dúvida."
        } else {
            "Como já vimos, mantenha o procedimento das normas consultadas."
        }
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
