package br.org.ceia.cemigpoc.data.engine

import android.util.Log
import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.LlmResponseChunk
import br.org.ceia.cemigpoc.domain.model.Message
import br.org.ceia.cemigpoc.llama.LlamaCppEngine
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Implementação real do motor de linguagem LLM executando Liquid LFM2.5 localmente via llama.cpp.
 */
class RealLlamaEngine(
    val engine: LlamaCppEngine = LlamaCppEngine()
) : LlmEngine {

    companion object {
        private const val TAG = "RealLlamaEngine"
    }

    override fun streamChat(
        messages: List<Message>,
        systemPrompt: String
    ): Flow<LlmResponseChunk> {
        val pairs = mutableListOf<Pair<String, String>>()
        if (systemPrompt.isNotBlank()) {
            pairs.add("system" to systemPrompt)
        }
        for (m in messages) {
            val roleStr = when (m.role) {
                Message.Role.SYSTEM -> "system"
                Message.Role.USER -> "user"
                Message.Role.ASSISTANT -> "assistant"
                Message.Role.TOOL -> "user" // Injeção de contexto normativo tratada como entrada
            }
            pairs.add(roleStr to m.content)
        }

        val prompt = engine.applyChatTemplate(pairs, addAssistant = true)
        Log.i(TAG, "Prompt gerado via chat_template (${prompt.length} chars)")

        return engine.generateStream(prompt).map { delta ->
            LlmResponseChunk.Text(delta)
        }
    }

    suspend fun generateComplete(prompt: String, maxTokens: Int = 50): String {
        return engine.generateComplete(prompt, maxTokens)
    }

    fun applyChatTemplate(pairs: List<Pair<String, String>>, addAssistant: Boolean = true): String {
        return engine.applyChatTemplate(pairs, addAssistant)
    }

    fun clearKvCache() {
        engine.clearKvCache()
    }
}
