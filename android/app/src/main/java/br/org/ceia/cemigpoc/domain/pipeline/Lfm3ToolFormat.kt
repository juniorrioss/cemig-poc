package br.org.ceia.cemigpoc.domain.pipeline

import br.org.ceia.cemigpoc.domain.model.Chunk

/**
 * Renderização NATIVA do chat template do LFM3 (f4-r32-ep2) no formato de tool-calling XML,
 * réplica determinística e byte-a-byte do `chat_template.jinja` do checkpoint (o MESMO do
 * treino F4 e da avaliação F5/F6, `apply_chat_template(..., enable_thinking=False)`).
 *
 * A C-API `llama_chat_apply_template` do JNI não executa o jinja (não renderiza `tools`, o papel
 * `tool` nem `<think></think>`), então montamos o prompt aqui e PROVAMOS a paridade em
 * `Lfm3ToolFormatParityTest` contra fixtures gerados pelo template oficial (jinja2).
 *
 * System = string BAKED (tools embutidas: `\n\n<tools>\n{json}\n</tools>`), idêntica a passar
 * `tools=` ao template (provado no PASSO 0 da F4) — asset `lfm3_tools_system_prompt.txt`.
 *
 * Formato (enable_thinking=False):
 *   <|startoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n
 *   <|im_start|>user\n{fala}<|im_end|>\n
 *   <|im_start|>assistant\n[<think></think> só nos turnos APÓS o último user]{texto}
 *       <tool_call>\n<function=buscar_norma>\n<parameter=consulta>\n…\n</parameter>\n
 *       <parameter=nr>\nNR-10|null\n</parameter>\n</function>\n</tool_call><|im_end|>\n
 *   <|im_start|>tool\n<tool_response>\n{trechos}\n</tool_response><|im_end|>\n
 *   (generation prompt) <|im_start|>assistant\n<think></think>
 *
 * Detalhe do template (keep_thinking): turnos assistant ANTERIORES ao último user recebem
 * `<think></think>` antes do conteúdo também (preserve_thinking=false -> content prefixado com
 * `<think></think>`), então TODO assistant histórico começa com `<think></think>`.
 *
 * Comentários em português; identificadores em inglês.
 */
object Lfm3ToolFormat {

    const val BOS = "<|startoftext|>"
    const val THINK_EMPTY = "<think></think>"
    const val TOOL_NAME = "buscar_norma"

    /** Um turno já materializado (histórico multiturno). */
    data class Turn(
        val userQuestion: String,
        val calledTool: Boolean,
        val consulta: String?,
        val nr: String?,
        val toolContext: String,
        val answer: String
    )

    private fun im(role: String, content: String): String = "<|im_start|>$role\n$content<|im_end|>\n"

    /** Bloco XML da chamada (render_tool_calls do template; args SEMPRE consulta e nr). */
    fun renderToolCall(consulta: String, nr: String?): String =
        "<tool_call>\n<function=$TOOL_NAME>\n" +
            "<parameter=consulta>\n$consulta\n</parameter>\n" +
            "<parameter=nr>\n${nr ?: "null"}\n</parameter>\n" +
            "</function>\n</tool_call>"

    /** Turno tool (render_tool_response de string). */
    fun renderToolResponse(content: String): String =
        if (content.contains("<tool_response>")) content else "<tool_response>\n$content\n</tool_response>"

    /** Formata trechos crus (== tool_schema_f4.format_tool_result / Compressor.render_raw). */
    fun formatRawChunks(chunks: List<Chunk>): String {
        if (chunks.isEmpty()) return "Nenhum trecho normativo encontrado para a consulta."
        return chunks.mapIndexed { i, c -> "[${i + 1}] (${sourceLabel(c)}):\n${c.content}" }
            .joinToString("\n\n")
    }

    /** 'doc - seção - título' (título omitido se vazio) == Compressor.source_label. */
    fun sourceLabel(c: Chunk): String {
        val base = "${c.doc} - ${c.section}"
        return if (c.title.isNotEmpty()) "$base - ${c.title}" else base
    }

    private fun appendHistory(sb: StringBuilder, history: List<Turn>) {
        for (t in history) {
            sb.append(im("user", t.userQuestion))
            if (t.calledTool) {
                sb.append(im("assistant", THINK_EMPTY + renderToolCall(t.consulta ?: t.userQuestion, t.nr)))
                sb.append(im("tool", renderToolResponse(t.toolContext)))
            }
            sb.append(im("assistant", THINK_EMPTY + t.answer))
        }
    }

    /** Prompt de DECISÃO: system + histórico + fala; o modelo chama a ferramenta ou responde. */
    fun renderDecisionPrompt(system: String, history: List<Turn>, userQuestion: String): String {
        val sb = StringBuilder(BOS)
        sb.append(im("system", system))
        appendHistory(sb, history)
        sb.append(im("user", userQuestion))
        sb.append("<|im_start|>assistant\n").append(THINK_EMPTY)
        return sb.toString()
    }

    /** Prompt de SÍNTESE: decisão + chamada + turno tool com o contexto; gera a resposta final. */
    fun renderSynthesisPrompt(
        system: String,
        history: List<Turn>,
        userQuestion: String,
        consulta: String,
        nr: String?,
        toolContext: String
    ): String {
        val sb = StringBuilder(BOS)
        sb.append(im("system", system))
        appendHistory(sb, history)
        sb.append(im("user", userQuestion))
        sb.append(im("assistant", THINK_EMPTY + renderToolCall(consulta, nr)))
        sb.append(im("tool", renderToolResponse(toolContext)))
        sb.append("<|im_start|>assistant\n").append(THINK_EMPTY)
        return sb.toString()
    }

    // ------------------------------------------------------------------ parser
    // Inverso determinístico do render (== bench/lfm3_f4/tool_parser.py).
    private val CALL_RE = Regex("<tool_call>\\s*(.*?)\\s*</tool_call>", RegexOption.DOT_MATCHES_ALL)
    private val FUNC_RE = Regex("<function=([^>\\s]+)>\\s*(.*?)\\s*</function>", RegexOption.DOT_MATCHES_ALL)
    private val PARAM_RE = Regex("<parameter=([^>\\s]+)>\\n(.*?)\\n</parameter>", RegexOption.DOT_MATCHES_ALL)

    data class ParsedCall(val name: String, val arguments: Map<String, String?>)

    fun looksLikeToolCall(text: String): Boolean = text.contains("<tool_call>") || text.contains("<function=")

    fun parse(text: String): List<ParsedCall> {
        val out = mutableListOf<ParsedCall>()
        for (m in CALL_RE.findAll(text)) {
            val f = FUNC_RE.find(m.groupValues[1]) ?: continue
            val args = LinkedHashMap<String, String?>()
            for (p in PARAM_RE.findAll(f.groupValues[2] + "\n")) {
                args[p.groupValues[1]] = if (p.groupValues[2] == "null") null else p.groupValues[2]
            }
            out.add(ParsedCall(f.groupValues[1], args))
        }
        return out
    }

    /** Remove thinking residual, blocos de tool_call e tokens de fim da resposta final. */
    fun cleanAnswer(text: String): String {
        var t = text
        val th = Regex("^\\s*<think>.*?</think>", RegexOption.DOT_MATCHES_ALL).find(t)
        if (th != null) t = t.substring(th.range.last + 1)
        val i = t.indexOf("<tool_call>")
        if (i >= 0) t = t.substring(0, i)
        return t.replace("<|im_end|>", "").trim()
    }
}
