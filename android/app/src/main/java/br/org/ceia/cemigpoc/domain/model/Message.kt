package br.org.ceia.cemigpoc.domain.model

/**
 * Representa uma mensagem no histórico conversacional do modelo com suporte a tool-calling.
 */
data class Message(
    val role: Role,
    val content: String,
    val toolCall: ToolCall? = null
) {
    enum class Role {
        SYSTEM,
        USER,
        ASSISTANT,
        TOOL
    }
}
