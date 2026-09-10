package br.org.ceia.cemigpoc.data.fake

import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.domain.model.ToolCall
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Modo de operação do FakeLlm para controle estrito em testes unitários.
 */
enum class FakeLlmMode {
    AUTO,
    FORCE_TOOL,
    FORCE_NO_TOOL
}

/**
 * Implementação mock do SLM que exercita de verdade o loop de turnos e tool-calling.
 *
 * @property mode Modo de controle (AUTO analisa a pergunta; FORCE força chamada ou não)
 * @property streamDelayMs Intervalo simulado entre emissão de tokens de streaming (0 para testes instantâneos)
 */
class FakeLlm(
    private val mode: FakeLlmMode = FakeLlmMode.AUTO,
    private val streamDelayMs: Long = 15L
) : LlmEngine {

    override fun streamChat(
        messages: List<Message>,
        systemPrompt: String
    ): Flow<LlmResponseChunk> = flow {
        val lastMessage = messages.lastOrNull() ?: return@flow

        // ---------------------------------------------------------------------
        // TURNO 2: O modelo recebeu o retorno da ferramenta (Role.TOOL)
        // ---------------------------------------------------------------------
        if (lastMessage.role == Message.Role.TOOL) {
            val toolContent = lastMessage.content

            val responseText = if (toolContent.contains("Nenhum trecho", ignoreCase = true) || toolContent.isBlank()) {
                "Não encontrei informações suficientes nas normas regulamentadoras consultadas para responder a essa pergunta com segurança. Recomenda-se consultar o responsável técnico da equipe de campo."
            } else if (toolContent.contains("NR-35", ignoreCase = true)) {
                "Conforme a [NR-35, Item 35.5.1], o sistema de proteção contra quedas é obrigatório para qualquer trabalho em altura acima de dois metros do nível inferior. É indispensável o uso de cinturão tipo paraquedista com talabarte duplo ou trava-quedas."
            } else {
                "Conforme a [NR-10, Item 10.4.1], as etapas obrigatórias para considerar uma instalação desenergizada são: seccionamento, impedimento de reenergização, constatação de ausência de tensão, aterramento temporário, proteção de elementos energizados e sinalização de impedimento. Somente após todos esses passos o circuito está seguro para intervenção."
            }

            // Simula streaming token a token (palavras com pontuação)
            emitStreamingText(responseText)
            return@flow
        }

        // ---------------------------------------------------------------------
        // TURNO 1: Decisão inicial com base na pergunta do usuário
        // ---------------------------------------------------------------------
        val userQuestion = messages.lastOrNull { it.role == Message.Role.USER }?.content ?: ""
        val normalizedQuestion = userQuestion.lowercase().trim()

        val shouldCallTool = when (mode) {
            FakeLlmMode.FORCE_TOOL -> true
            FakeLlmMode.FORCE_NO_TOOL -> false
            FakeLlmMode.AUTO -> {
                // Perguntas conversacionais não chamam tool
                val isConversational = normalizedQuestion.startsWith("olá") ||
                        normalizedQuestion.startsWith("ola") ||
                        normalizedQuestion.startsWith("bom dia") ||
                        normalizedQuestion.startsWith("boa tarde") ||
                        normalizedQuestion.contains("quem é você") ||
                        normalizedQuestion.contains("ajuda") && normalizedQuestion.length < 15

                !isConversational
            }
        }

        if (shouldCallTool) {
            // Reformulação assertiva para busca BM25
            val assertiveQuery = reformulateQueryForBm25(userQuestion)
            emit(
                LlmResponseChunk.Call(
                    ToolCall(
                        toolName = "retriever",
                        query = assertiveQuery
                    )
                )
            )
        } else {
            // Resposta conversacional direta sem RAG
            val directAnswer = "Olá! Sou o assistente de segurança operacional da CEMIG. Posso consultar normas técnicas como NR-10 e NR-35 para tirar dúvidas de segurança e procedimentos de campo."
            emitStreamingText(directAnswer)
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

    /**
     * Simula a capacidade do LLM de transformar perguntas longas em termos assertivos BM25.
     */
    private fun reformulateQueryForBm25(question: String): String {
        val q = question.lowercase()
        return when {
            q.contains("desenergiz") || q.contains("passos") || q.contains("etapas") -> "NR-10 desenergizacao etapas seccionamento aterramento"
            q.contains("altura") || q.contains("queda") || q.contains("nr-35") || q.contains("cinto") -> "NR-35 protecao quedas altura cinturão"
            q.contains("epi") || q.contains("vestimenta") || q.contains("adornos") -> "NR-10 vestimentas EPI adornos"
            q.contains("coletiv") || q.contains("tensao de seguranca") -> "NR-10 protecao coletiva desenergizacao"
            else -> question.replace(Regex("[^a-zA-Z0-9À-ÿ\\s]"), " ").trim()
        }
    }
}
