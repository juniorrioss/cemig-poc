package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.fake.FakeLlm
import br.org.ceia.cemigpoc.data.fake.FakeLlmMode
import br.org.ceia.cemigpoc.data.fake.FakeRetriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Testes de unidade JVM do AskPipeline exercitando o loop de turnos e tool-calling com os fakes.
 */
class AskPipelineTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private val systemPrompt = "Você é um assistente de normas técnicas. Chame retriever para buscar normas."
    private lateinit var telemetryFile: File
    private lateinit var telemetryLogger: TelemetryLogger

    @Before
    fun setup() {
        telemetryFile = tempFolder.newFile("telemetry_test.jsonl")
        telemetryLogger = TelemetryLogger(telemetryFile)
    }

    @Test
    fun perguntaQueDisparaTool_executaTurnoDuplo_eCitaFonte() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_TOOL, streamDelayMs = 0L)
        val pipeline = AskPipeline(
            systemPrompt = systemPrompt,
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger
        )

        val events = pipeline.execute("Quais são as etapas para desenergização de acordo com a NR-10?").toList()

        // 1. Verifica se houve chamada da tool no Turno 1
        val toolCallEvent = events.filterIsInstance<TurnEvent.ToolCallEvent>().firstOrNull()
        assertNotNull("Deveria emitir ToolCallEvent", toolCallEvent)
        assertEquals("retriever", toolCallEvent!!.call.toolName)
        assertTrue(
            "Query de busca deve conter termos assertivos",
            toolCallEvent.call.query.contains("desenergiz", ignoreCase = true)
        )

        // 2. Verifica se houve retorno da tool
        val toolResultEvent = events.filterIsInstance<TurnEvent.ToolResultEvent>().firstOrNull()
        assertNotNull("Deveria emitir ToolResultEvent", toolResultEvent)
        assertFalse("Deveria retornar chunks relevantes", toolResultEvent!!.chunks.isEmpty())

        // 3. Verifica streaming de texto no Turno 2
        val textDeltas = events.filterIsInstance<TurnEvent.TextDelta>()
        assertTrue("Deveria conter deltas de texto da resposta", textDeltas.isNotEmpty())

        // 4. Verifica conclusão e obrigatoriedade de citação de fonte
        val doneEvent = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull("Deveria emitir evento Done final", doneEvent)
        assertTrue(
            "A resposta final deve citar a norma e seção",
            doneEvent!!.finalAnswer.contains("NR-10", ignoreCase = true)
        )
        assertTrue(
            "A resposta final deve citar a seção/item",
            doneEvent.finalAnswer.contains("10.4.1")
        )
        assertFalse("Chunks usados devem estar presentes no Done", doneEvent.chunksUsed.isEmpty())
        assertEquals(1, doneEvent.toolCallsMade.size)

        // 5. Verifica se a telemetria foi gravada
        assertEquals(1, telemetryLogger.getLogCount())
    }

    @Test
    fun perguntaQueNaoDisparaTool_respondeDiretamenteEmTurnoUnico() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_NO_TOOL, streamDelayMs = 0L)
        val pipeline = AskPipeline(
            systemPrompt = systemPrompt,
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger
        )

        val events = pipeline.execute("Olá, bom dia! Como você funciona?").toList()

        // Verifica ausência de tool call
        val toolCallEvents = events.filterIsInstance<TurnEvent.ToolCallEvent>()
        assertTrue("Não deveria chamar nenhuma ferramenta para saudações", toolCallEvents.isEmpty())

        val toolResultEvents = events.filterIsInstance<TurnEvent.ToolResultEvent>()
        assertTrue("Não deveria haver resultado de ferramenta", toolResultEvents.isEmpty())

        // Verifica resposta direta
        val doneEvent = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull("Deveria emitir Done", doneEvent)
        assertTrue("Chunks usados devem ser vazios", doneEvent!!.chunksUsed.isEmpty())
        assertTrue("Tool calls feitas devem ser vazias", doneEvent.toolCallsMade.isEmpty())
        assertTrue(
            "Resposta deve ser conversacional",
            doneEvent.finalAnswer.contains("assistente", ignoreCase = true)
        )

        // Telemetria registrada mesmo para respostas diretas
        assertEquals(1, telemetryLogger.getLogCount())
    }

    @Test
    fun toolSemResultados_respondeNaoSeiComSeguranca() = runTest {
        // Retriever configurado para simular ausência de trechos encontrados
        val retriever = FakeRetriever(returnEmpty = true)
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_TOOL, streamDelayMs = 0L)
        val pipeline = AskPipeline(
            systemPrompt = systemPrompt,
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger
        )

        val events = pipeline.execute("Qual a regra para um tópico inexistente xyz?").toList()

        // A ferramenta foi acionada
        val toolCallEvent = events.filterIsInstance<TurnEvent.ToolCallEvent>().firstOrNull()
        assertNotNull(toolCallEvent)

        // Mas retornou vazio
        val toolResultEvent = events.filterIsInstance<TurnEvent.ToolResultEvent>().firstOrNull()
        assertNotNull(toolResultEvent)
        assertTrue("Retorno da ferramenta deve ser vazio", toolResultEvent!!.chunks.isEmpty())

        // LLM deve responder "Não encontrei informações suficientes..."
        val doneEvent = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull(doneEvent)
        assertTrue(
            "Deve informar com segurança que não encontrou informações na norma",
            doneEvent!!.finalAnswer.contains("Não encontrei informações suficientes", ignoreCase = true)
        )
    }

    @Test
    fun modoAuto_decideAdequadamenteEntreToolERespostaDireta() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.AUTO, streamDelayMs = 0L)
        val pipeline = AskPipeline(
            systemPrompt = systemPrompt,
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger
        )

        // Pergunta conversacional -> Sem tool
        val greetingEvents = pipeline.execute("Olá, quem é você?").toList()
        assertTrue(
            "Saudação não deve acionar tool",
            greetingEvents.filterIsInstance<TurnEvent.ToolCallEvent>().isEmpty()
        )

        // Pergunta técnica -> Com tool
        val technicalEvents = pipeline.execute("Qual a proteção obrigatória contra quedas na NR-35?").toList()
        val toolEvent = technicalEvents.filterIsInstance<TurnEvent.ToolCallEvent>().firstOrNull()
        assertNotNull("Pergunta técnica deve acionar tool", toolEvent)
        assertTrue(toolEvent!!.call.query.contains("NR-35", ignoreCase = true))
    }
}
