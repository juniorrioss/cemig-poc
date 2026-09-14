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

    /**
     * Gera a partir de um prompt JÁ renderizado no formato nativo (ChatML do LFM2.5), sem
     * aplicar chat template no engine. Necessário para o pipeline híbrido de tool-calling:
     * a renderização (system com `List of tools:`, chamada `<|tool_call_start|>[...]`, turno
     * `tool`) é feita em Kotlin (LfmToolRenderer) para bater byte-a-byte com o treino, o que
     * o `llama_chat_apply_template` do JNI não consegue reproduzir.
     *
     * @param prompt Prompt completo no formato nativo, terminando em `<|im_start|>assistant\n`
     * @param maxTokens Teto de tokens gerados neste turno
     * @return Fluxo de deltas de texto (inclui tokens especiais reemitidos, ex.: tool-call)
     */
    fun generateFromPrompt(
        prompt: String,
        maxTokens: Int = 384
    ): Flow<LlmResponseChunk>
}
