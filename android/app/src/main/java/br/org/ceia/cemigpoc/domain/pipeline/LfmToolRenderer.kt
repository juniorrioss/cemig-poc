package br.org.ceia.cemigpoc.domain.pipeline

import br.org.ceia.cemigpoc.domain.model.Chunk

/**
 * Renderização NATIVA do chat template do LFM2.5 no formato de tool-calling — réplica
 * determinística e byte-a-byte do `chat_template.jinja` oficial usado no TREINO (tools_v1).
 *
 * Motivo (crítico, emenda 2 do brief): o system + a gramática de tool-call servidos ao modelo
 * têm de ser IDÊNTICOS aos do dataset de treino; caso contrário treinamos num prompt e
 * servimos noutro. A C-API `llama_chat_apply_template` (path JNI) NÃO renderiza `tools=` nem o
 * papel `tool`, então montamos o ChatML aqui, à mão, e provamos a paridade em
 * `LfmToolRendererParityTest` contra os fixtures gerados pelo template oficial (jinja2).
 *
 * O SYSTEM já vem com o sufixo `List of tools: [...]` BAKED (idêntico byte-a-byte a passar
 * `tools=` ao template — ver tools_v1/prep_train). Guardamos essa string exata como asset
 * `tools_system_prompt.txt` (SHA256 363185b7…, 2381 chars) e a injetamos como system.
 *
 * Formato (validado):
 *   <|startoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n
 *   <|im_start|>user\n{fala}<|im_end|>\n
 *   <|im_start|>assistant\n<|tool_call_start|>[buscar_norma(consulta='…', nr='NR-10')]<|tool_call_end|><|im_end|>\n
 *   <|im_start|>tool\n[1] (NR-10 - 10.x - título):\n{texto}<|im_end|>\n
 *   <|im_start|>assistant\n{resposta}<|im_end|>\n
 *   (add_generation_prompt -> ...<|im_start|>assistant\n)
 *
 * Comentários em português; identificadores em inglês.
 */
object LfmToolRenderer {

    const val BOS = "<|startoftext|>"
    const val TOOL_CALL_START = "<|tool_call_start|>"
    const val TOOL_CALL_END = "<|tool_call_end|>"
    const val TOOL_NAME = "buscar_norma"

    /** Um turno já materializado na conversa, para reconstrução do histórico. */
    data class RenderedTurn(
        val userQuestion: String,
        val calledTool: Boolean,
        val consulta: String?,   // argumento reescrito pelo modelo (só telemetria/reconstrução)
        val nr: String?,         // norma citada pelo modelo (null = ausente)
        val toolContext: String, // conteúdo do turno `tool` (chunks formatados); vazio se não buscou
        val answer: String
    )

    /**
     * Escapa uma string exatamente como o macro `format_arg_value` do template oficial:
     * `\` -> `\\`, `'` -> `\'`, `\n` -> `\n` (literal), `\r` -> `\r` (literal). A ordem importa
     * (a barra primeiro) para não re-escapar as barras introduzidas.
     */
    fun escapeArg(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
    }

    /**
     * Reconstrói a chamada canônica `<|tool_call_start|>[buscar_norma(consulta='…', nr='…')]
     * <|tool_call_end|>`. `nr` só entra quando não-nulo (assinatura enxuta do treino).
     */
    fun renderToolCall(consulta: String, nr: String?): String {
        val args = StringBuilder("consulta='").append(escapeArg(consulta)).append("'")
        if (!nr.isNullOrBlank()) {
            args.append(", nr='").append(escapeArg(nr)).append("'")
        }
        return "$TOOL_CALL_START[$TOOL_NAME($args)]$TOOL_CALL_END"
    }

    /** Formata os chunks recuperados como conteúdo do turno `tool` (== format_tool_result). */
    fun formatToolResult(chunks: List<Chunk>): String {
        if (chunks.isEmpty()) return "Nenhum trecho normativo encontrado para a consulta."
        return chunks.mapIndexed { idx, c ->
            "[${idx + 1}] (${c.doc} - ${c.section} - ${c.title}):\n${c.content}"
        }.joinToString("\n\n")
    }

    private fun imBlock(role: String, content: String): String =
        "<|im_start|>$role\n$content<|im_end|>\n"

    /**
     * Renderiza o prompt do PRIMEIRO turno de geração (decisão): system + histórico + a nova
     * fala do usuário, terminando em `<|im_start|>assistant\n` (add_generation_prompt=true).
     * O modelo decide sozinho: emite uma tool_call OU responde direto.
     */
    fun renderDecisionPrompt(
        systemPrompt: String,
        history: List<RenderedTurn>,
        userQuestion: String
    ): String {
        val sb = StringBuilder(BOS)
        sb.append(imBlock("system", systemPrompt))
        appendHistory(sb, history)
        sb.append(imBlock("user", userQuestion))
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    /**
     * Renderiza o prompt do SEGUNDO turno de geração (síntese): igual ao de decisão, mas com
     * o turno assistant da tool_call + o turno `tool` (chunks) já injetados, terminando em
     * `<|im_start|>assistant\n` para o modelo sintetizar a resposta final.
     */
    fun renderSynthesisPrompt(
        systemPrompt: String,
        history: List<RenderedTurn>,
        userQuestion: String,
        consulta: String,
        nr: String?,
        toolContext: String
    ): String {
        val sb = StringBuilder(BOS)
        sb.append(imBlock("system", systemPrompt))
        appendHistory(sb, history)
        sb.append(imBlock("user", userQuestion))
        sb.append(imBlock("assistant", renderToolCall(consulta, nr)))
        sb.append(imBlock("tool", toolContext))
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    /** Anexa os turnos completos do histórico (user + [tool_call + tool] + assistant). */
    private fun appendHistory(sb: StringBuilder, history: List<RenderedTurn>) {
        for (turn in history) {
            sb.append(imBlock("user", turn.userQuestion))
            if (turn.calledTool && !turn.consulta.isNullOrBlank()) {
                sb.append(imBlock("assistant", renderToolCall(turn.consulta, turn.nr)))
                sb.append(imBlock("tool", turn.toolContext))
            }
            sb.append(imBlock("assistant", turn.answer))
        }
    }
}
