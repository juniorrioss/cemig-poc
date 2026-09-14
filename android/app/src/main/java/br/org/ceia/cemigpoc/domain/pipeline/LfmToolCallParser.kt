package br.org.ceia.cemigpoc.domain.pipeline

/**
 * Parser TOLERANTE da saída do modelo em inferência — porte fiel de
 * `tools_v1/render.py::parse_tool_calls_runtime`.
 *
 * O engine nativo (llama.cpp) frequentemente NÃO reemite os tokens especiais
 * `<|tool_call_start|>`/`<|tool_call_end|>` como texto — o modelo (treinado) emite a chamada
 * como `[buscar_norma(consulta='…', nr='NR-10')]`. Este parser aceita a forma COM ou SEM os
 * wrappers e exige que o corpo contenha `buscar_norma(` (evita casar listas comuns na prosa).
 *
 * Gramática (do template oficial): `[func(arg=valor, arg2=valor2)]`, strings entre aspas
 * simples com escape `\\`, `\'`, `\n`, `\r`. Retorna [] se não houver chamada reconhecível.
 *
 * Comentários em português; identificadores em inglês.
 */
object LfmToolCallParser {

    const val TOOL_NAME = "buscar_norma"

    data class ParsedCall(val name: String, val arguments: Map<String, String?>)

    // Captura o corpo entre colchetes, com ou sem os wrappers de tool-call. Não-guloso.
    private val CALL_RE = Regex(
        "(?:<\\|tool_call_start\\|>)?\\s*\\[(.*?)]\\s*(?:<\\|tool_call_end\\|>)?",
        setOf(RegexOption.DOT_MATCHES_ALL)
    )

    /** true se o texto cru contém um marcador de chamada de ferramenta. */
    fun looksLikeToolCall(text: String): Boolean =
        text.contains("<|tool_call_start|>") || text.contains("$TOOL_NAME(")

    /**
     * Extrai as chamadas reconhecíveis. Ignora corpos que não contenham `buscar_norma(`.
     */
    fun parse(text: String): List<ParsedCall> {
        val calls = mutableListOf<ParsedCall>()
        for (m in CALL_RE.findAll(text)) {
            val body = m.groupValues[1].trim()
            if (body.isEmpty() || !body.contains("$TOOL_NAME(")) continue
            try {
                for (callStr in splitCalls(body)) {
                    if (!callStr.startsWith(TOOL_NAME)) continue
                    val (name, inner) = splitNameArgs(callStr) ?: continue
                    val args = LinkedHashMap<String, String?>()
                    for (piece in splitTopLevelArgs(inner)) {
                        val eq = piece.indexOf('=')
                        if (eq < 0) continue
                        val k = piece.substring(0, eq).trim()
                        val v = piece.substring(eq + 1)
                        args[k] = parseArgValue(v)
                    }
                    calls.add(ParsedCall(name, args))
                }
            } catch (_: Exception) {
                // corpo malformado: ignora esta ocorrência (tolerante)
            }
        }
        return calls
    }

    /** Converte um valor renderizado de volta ao Python-esque: string entre aspas, None/null. */
    private fun parseArgValue(raw: String): String? {
        val s = raw.trim()
        if (s.length >= 2 && s.first() == '\'' && s.last() == '\'') {
            return unescapeSingleQuoted(s.substring(1, s.length - 1))
        }
        if (s == "None" || s == "null") return null
        return s
    }

    /** Inverte o escape do format_arg_value (`\\`, `\'`, `\n`, `\r`). */
    private fun unescapeSingleQuoted(s: String): String {
        val out = StringBuilder()
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    '\\' -> out.append('\\')
                    '\'' -> out.append('\'')
                    'n' -> out.append('\n')
                    'r' -> out.append('\r')
                    else -> out.append('\\').append(s[i + 1])
                }
                i += 2
            } else {
                out.append(c)
                i += 1
            }
        }
        return out.toString()
    }

    /** Separa múltiplas chamadas func(...) no topo da lista, respeitando parênteses/strings. */
    private fun splitCalls(body: String): List<String> {
        val parts = mutableListOf<String>()
        val buf = StringBuilder()
        var depth = 0
        var inStr = false
        var esc = false
        for (ch in body) {
            if (inStr) {
                buf.append(ch)
                when {
                    esc -> esc = false
                    ch == '\\' -> esc = true
                    ch == '\'' -> inStr = false
                }
                continue
            }
            when (ch) {
                '\'' -> { inStr = true; buf.append(ch) }
                '(' -> { depth++; buf.append(ch) }
                ')' -> { depth--; buf.append(ch) }
                ',' -> if (depth == 0) { parts.add(buf.toString()); buf.clear() } else buf.append(ch)
                else -> buf.append(ch)
            }
        }
        if (buf.isNotEmpty()) parts.add(buf.toString())
        return parts.map { it.trim() }.filter { it.isNotEmpty() }
    }

    /** 'func(a=1, b=2)' -> ('func', 'a=1, b=2'). Retorna null se malformado. */
    private fun splitNameArgs(callStr: String): Pair<String, String>? {
        val s = callStr.trim()
        val lp = s.indexOf('(')
        if (lp < 0 || !s.endsWith(")")) return null
        return s.substring(0, lp).trim() to s.substring(lp + 1, s.length - 1)
    }

    /** Divide 'a=1, b=2' respeitando vírgulas dentro de strings entre aspas. */
    private fun splitTopLevelArgs(inner: String): List<String> {
        val parts = mutableListOf<String>()
        val buf = StringBuilder()
        var inStr = false
        var esc = false
        for (ch in inner) {
            if (inStr) {
                buf.append(ch)
                when {
                    esc -> esc = false
                    ch == '\\' -> esc = true
                    ch == '\'' -> inStr = false
                }
            } else {
                when (ch) {
                    '\'' -> { inStr = true; buf.append(ch) }
                    ',' -> { parts.add(buf.toString()); buf.clear() }
                    else -> buf.append(ch)
                }
            }
        }
        if (buf.isNotEmpty()) parts.add(buf.toString())
        return parts.map { it.trim() }.filter { it.isNotEmpty() }
    }
}
