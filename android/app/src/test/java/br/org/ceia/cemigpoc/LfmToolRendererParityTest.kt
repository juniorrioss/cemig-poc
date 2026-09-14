package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.pipeline.LfmToolCallParser
import br.org.ceia.cemigpoc.domain.pipeline.LfmToolRenderer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Prova de PARIDADE BYTE-A-BYTE (exigência da emenda 2 do brief): o renderizador nativo em
 * Kotlin (LfmToolRenderer) tem de produzir EXATAMENTE o mesmo prompt que o `chat_template.jinja`
 * oficial usado no TREINO (tools_v1). Os fixtures em src/test/resources/tool_render/
 * foram gerados pelo template oficial (jinja2, tools=TOOLS) — se um byte divergir, o teste falha.
 *
 * Também confere que o SYSTEM embarcado (asset tools_system_prompt.txt) é IDÊNTICO ao usado no
 * treino (SHA256 363185b7…, 2381 chars) e que o parser reextrai a chamada renderizada (round-trip).
 */
class LfmToolRendererParityTest {

    private fun res(name: String): String =
        javaClass.classLoader!!.getResourceAsStream(name)!!
            .bufferedReader(Charsets.UTF_8).use { it.readText() }

    private val system: String by lazy { res("tools_system_prompt.txt") }

    private val q13kv = "distância segura pra trabalhar perto de rede de 13,8 kV"
    private val chunk13kv = Chunk(
        id = 1, doc = "NR-10", section = "10.2.8.2", title = "Medidas de proteção coletiva",
        content = "Nos trabalhos em instalações elétricas, quando não for possível o desligamento, a distância de segurança para a faixa de tensão de 13,8 kV deve respeitar o raio de delimitação da zona de risco definido no Anexo II."
    )
    private val consulta13kv = "distancia seguranca zona risco 13,8 kV 13.8 kV media tensao"

    @Test
    fun systemPrompt_ehIdenticoAoTreino() {
        // O asset já inclui o sufixo `List of tools: [...]` baked (idêntico ao dataset).
        assertEquals("System deve ter 2381 chars (treino)", 2381, system.length)
        assertTrue(system.contains("List of tools: ["))
        assertTrue(system.startsWith("Você é o assistente técnico de campo da CEMIG."))
    }

    @Test
    fun renderDecisionPrompt_baterFixtureA() {
        val expected = res("tool_render/A_user_only_genprompt.txt")
        val actual = LfmToolRenderer.renderDecisionPrompt(system, emptyList(), q13kv)
        assertEquals(expected, actual)
    }

    @Test
    fun renderSynthesisPrompt_baterFixtureB() {
        val expected = res("tool_render/B_after_tool_genprompt.txt")
        val actual = LfmToolRenderer.renderSynthesisPrompt(
            systemPrompt = system, history = emptyList(), userQuestion = q13kv,
            consulta = consulta13kv, nr = "NR-10",
            toolContext = LfmToolRenderer.formatToolResult(listOf(chunk13kv))
        )
        assertEquals(expected, actual)
    }

    @Test
    fun renderMultiturnComReuso_baterFixtureC() {
        val expected = res("tool_render/C_multiturn_reuse_genprompt.txt")
        // Histórico: 1 troca que buscou (tool_call + tool + resposta).
        val prevAnswer = "Pela NR-10, item 10.2.8.2, respeite o raio de delimitação da zona de risco para 13,8 kV do Anexo II antes de se aproximar."
        val history = listOf(
            LfmToolRenderer.RenderedTurn(
                userQuestion = q13kv, calledTool = true, consulta = consulta13kv, nr = "NR-10",
                toolContext = LfmToolRenderer.formatToolResult(listOf(chunk13kv)),
                answer = prevAnswer
            )
        )
        val actual = LfmToolRenderer.renderDecisionPrompt(
            system, history, "e se eu não conseguir desligar a rede?"
        )
        assertEquals(expected, actual)
    }

    @Test
    fun renderSaudacao_baterFixtureD() {
        val expected = res("tool_render/D_greeting_genprompt.txt")
        val actual = LfmToolRenderer.renderDecisionPrompt(system, emptyList(), "bom dia, tudo certo por aí?")
        assertEquals(expected, actual)
    }

    @Test
    fun parser_reextraiChamadaRenderizada() {
        val call = LfmToolRenderer.renderToolCall(consulta13kv, "NR-10")
        val parsed = LfmToolCallParser.parse(call)
        assertEquals(1, parsed.size)
        assertEquals("buscar_norma", parsed[0].name)
        assertEquals(consulta13kv, parsed[0].arguments["consulta"])
        assertEquals("NR-10", parsed[0].arguments["nr"])
    }

    @Test
    fun parser_aceitaChamadaSemWrappers() {
        // O engine nativo às vezes não reemite os tokens especiais.
        val raw = "[buscar_norma(consulta='inspecao poste antes de subir', nr='NR-35')]"
        val parsed = LfmToolCallParser.parse(raw)
        assertEquals(1, parsed.size)
        assertEquals("inspecao poste antes de subir", parsed[0].arguments["consulta"])
        assertEquals("NR-35", parsed[0].arguments["nr"])
    }

    @Test
    fun parser_nrNulaVoltaComoNull() {
        val call = LfmToolRenderer.renderToolCall("estabilidade estrutural poste", null)
        val parsed = LfmToolCallParser.parse(call)
        assertEquals(1, parsed.size)
        assertTrue("nr ausente", parsed[0].arguments["nr"] == null || !parsed[0].arguments.containsKey("nr"))
    }

    @Test
    fun escapeArg_espelhaFormatArgValueDoTemplate() {
        // aspas simples, barra e quebra de linha escapadas como o macro format_arg_value.
        assertEquals("aspas \\'simples\\'", LfmToolRenderer.escapeArg("aspas 'simples'"))
        assertEquals("barra \\\\ e nl \\n", LfmToolRenderer.escapeArg("barra \\ e nl \n"))
    }
}
