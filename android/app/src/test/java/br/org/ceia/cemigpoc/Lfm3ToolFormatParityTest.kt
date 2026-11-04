package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.domain.pipeline.Lfm3ToolFormat
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.File
import java.security.MessageDigest

/**
 * PARIDADE BYTE-A-BYTE do renderizador LFM3 (Kotlin) com o chat_template.jinja do checkpoint
 * f4-r32-ep2 (apply_chat_template(..., enable_thinking=False), transformers 5.x — o MESMO do
 * treino F4 e da régua F6). Fixtures: `bench/lfm3_f7/make_fixtures.py --template`.
 */
class Lfm3ToolFormatParityTest {

    private fun res(name: String) = File("src/test/resources/$name").readText(Charsets.UTF_8)
    private val system by lazy { res("lfm3_tools_system_prompt.txt") }
    private val inp by lazy { JSONObject(res("lfm3_render/inputs.json")) }

    @Test
    fun systemDoAsset_ehOBakedDoTreino() {
        val asset = File("src/main/assets/lfm3_tools_system_prompt.txt").readText(Charsets.UTF_8)
        assertEquals(system, asset)
        val sha = MessageDigest.getInstance("SHA-256").digest(asset.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        assertEquals("921e6a1f2c77bcf1afb0b3e44b976261adb9de4760f631ad7a8f30549cc8eb15", sha)
    }

    @Test
    fun decisao_A() {
        assertEquals(res("lfm3_render/A_decision.txt"),
            Lfm3ToolFormat.renderDecisionPrompt(system, emptyList(), inp.getString("q1")))
    }

    @Test
    fun sintese_B() {
        val q1 = inp.getString("q1")
        assertEquals(res("lfm3_render/B_synthesis.txt"),
            Lfm3ToolFormat.renderSynthesisPrompt(system, emptyList(), q1, q1, "NR-10", inp.getString("ctx1")))
    }

    @Test
    fun multiturno_C() {
        val q1 = inp.getString("q1")
        val h = listOf(Lfm3ToolFormat.Turn(q1, true, q1, "NR-10", inp.getString("ctx1"), inp.getString("a1")))
        assertEquals(res("lfm3_render/C_multiturn.txt"),
            Lfm3ToolFormat.renderDecisionPrompt(system, h, inp.getString("q2")))
    }

    @Test
    fun reusoComRespostaDireta_D() {
        val q1 = inp.getString("q1")
        val h = listOf(
            Lfm3ToolFormat.Turn(q1, true, q1, null, inp.getString("ctx1"), inp.getString("a1")),
            Lfm3ToolFormat.Turn(inp.getString("q3"), false, null, null, "", inp.getString("a3"))
        )
        assertEquals(res("lfm3_render/D_reuse_direct.txt"),
            Lfm3ToolFormat.renderDecisionPrompt(system, h, inp.getString("q2")))
    }

    @Test
    fun sinteseEstendeDecisao_prefixoExato() {
        // keepPrefix do engine depende disto: síntese = decisão (sem o `<think></think>` final
        // já está nela) + chamada + tool.
        val q1 = inp.getString("q1")
        val dec = Lfm3ToolFormat.renderDecisionPrompt(system, emptyList(), q1)
        val syn = Lfm3ToolFormat.renderSynthesisPrompt(system, emptyList(), q1, q1, "NR-10", "x")
        assertEquals(true, syn.startsWith(dec))
    }

    @Test
    fun parser_roundTrip() {
        val call = Lfm3ToolFormat.renderToolCall("13,8 kV 'aspas'\nlinha 2", null)
        val p = Lfm3ToolFormat.parse("<think></think>$call")
        assertEquals(1, p.size)
        assertEquals("buscar_norma", p[0].name)
        assertEquals("13,8 kV 'aspas'\nlinha 2", p[0].arguments["consulta"])
        assertNull(p[0].arguments["nr"])
        assertEquals("Resposta.", Lfm3ToolFormat.cleanAnswer("<think></think>Resposta.<|im_end|>"))
    }
}
