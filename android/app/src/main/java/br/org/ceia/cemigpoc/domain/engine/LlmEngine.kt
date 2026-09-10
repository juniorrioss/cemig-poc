package br.org.ceia.cemigpoc.domain.engine

import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.domain.model.ToolCall
import kotlinx.coroutines.flow.Flow

/**
 * Pedaço de resposta gerado pelo modelo de linguagem SLM.
 */
sealed interface LlmResponseChunk {
    /**
     * Fragmento de texto gerado para streaming da resposta ao usuário.
     */
    data class Text(val delta: String) : LlmResponseChunk

    /**
     * Decisão do modelo de invocar uma ferramenta com parâmetros específicos.
     */
    data class Call(val toolCall: ToolCall) : LlmResponseChunk
}

/**
 * Interface do modelo de linguagem (SLM) com suporte a turnos e tool-calling nativo.
 */
interface LlmEngine {
    /**
     * Gera resposta em streaming a partir do histórico conversacional e system prompt.
     *
     * @param messages Histórico de mensagens da conversa atual
     * @param systemPrompt Diretrizes do sistema e especificação da ferramenta de busca
     * @return Fluxo de emissões contendo deltas de texto ou chamada de ferramenta
     */
    fun streamChat(
        messages: List<Message>,
        systemPrompt: String
    ): Flow<LlmResponseChunk>
}
