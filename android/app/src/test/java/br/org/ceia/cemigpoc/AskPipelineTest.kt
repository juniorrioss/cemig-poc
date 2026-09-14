package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.fake.FakeLlm
import br.org.ceia.cemigpoc.data.fake.FakeLlmMode
import br.org.ceia.cemigpoc.data.fake.FakeRetriever
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
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
 * Bateria de testes de unidade JVM do AskPipeline HÍBRIDO de tool-calling (task poc-app-tools):
 * 1. Pergunta que DEVE buscar -> houve tool_call e busca externa (chunks não vazios);
 * 2. Saudação/agradecimento -> ZERO busca (modelo não chama a ferramenta);
 * 3. Reuso: sem tool_call, sem nova busca (chunks vazios, gate "reuso");
 * 4. Poda de histórico por orçamento de tokens (troca mais antiga sai primeiro);
 * 5. Sanitização FTS5 de normas com traços e diacríticos;
 * 6. Telemetria local registra a decisão de tool-calling.
 */
class AskPipelineTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var telemetryFile: File
    private lateinit var telemetryLogger: TelemetryLogger

    @Before
    fun setup() {
        telemetryFile = tempFolder.newFile("telemetry_test.jsonl")
        telemetryLogger = TelemetryLogger(telemetryFile)
    }

    @Test
    fun perguntaTecnica_disparaToolCallEBuscaExterna() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_TOOL)
        val pipeline = AskPipeline(retriever = retriever, llmEngine = llm, telemetryLogger = telemetryLogger)

        val events = pipeline.execute("Qual o EPI para trabalho em altura na NR-35?").toList()

        val toolDecided = events.filterIsInstance<TurnEvent.ToolDecided>().firstOrNull()
        assertNotNull("Deveria emitir ToolDecided", toolDecided)
        assertTrue("O modelo deveria ter chamado a ferramenta", toolDecided!!.called)

        val chunksEvent = events.filterIsInstance<TurnEvent.ChunksRetrieved>().firstOrNull()
        assertNotNull(chunksEvent)
        assertFalse("A busca externa NÃO é reuso", chunksEvent!!.reused)
        assertTrue("A busca externa deve retornar chunks", chunksEvent.chunks.isNotEmpty())

        val done = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull(done)
        assertTrue("Métricas devem marcar called_tool", done!!.metrics.calledTool)
        assertTrue("Resposta deve ser fundamentada", done.finalAnswer.isNotBlank())
    }

    @Test
    fun saudacao_naoDisparaBusca() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.AUTO)
        val pipeline = AskPipeline(retriever = retriever, llmEngine = llm, telemetryLogger = telemetryLogger)

        val events = pipeline.execute("Bom dia, tudo certo por aí?").toList()

        val toolDecided = events.filterIsInstance<TurnEvent.ToolDecided>().firstOrNull()
        assertNotNull(toolDecided)
        assertFalse("Saudação não deve chamar a ferramenta", toolDecided!!.called)

        val chunksEvent = events.filterIsInstance<TurnEvent.ChunksRetrieved>().firstOrNull()
        assertNotNull(chunksEvent)
        assertTrue("Saudação: chunks vazios", chunksEvent!!.chunks.isEmpty())

        val done = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull(done)
        assertFalse("Saudação não busca", done!!.metrics.calledTool)
        assertEquals("reuso", done.metrics.gateMode)
    }

    @Test
    fun reuso_semToolCall_naoRefazBusca() = runTest {
        val retriever = FakeRetriever()
        // FORCE_NO_TOOL simula o modelo decidindo reusar o contexto já em tela.
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_NO_TOOL)
        val pipeline = AskPipeline(retriever = retriever, llmEngine = llm, telemetryLogger = telemetryLogger)

        val previousChunk = Chunk(id = 42, doc = "NR-10", section = "10.5.1", content = "Regras de desenergização", score = 0.8)
        val history = listOf(
            ConversationTurn(
                question = "Quais as etapas de desenergização da NR-10?",
                answer = "Conforme NR-10 item 10.5.1...",
                chunks = listOf(previousChunk),
                calledTool = true,
                toolConsulta = "etapas desenergizacao NR-10",
                toolNr = "NR-10"
            )
        )

        val events = pipeline.execute("E o próximo passo depois do seccionamento?", history = history).toList()

        val toolDecided = events.filterIsInstance<TurnEvent.ToolDecided>().firstOrNull()
        assertNotNull(toolDecided)
        assertFalse("Reuso: modelo não chama a ferramenta", toolDecided!!.called)

        val chunksEvent = events.filterIsInstance<TurnEvent.ChunksRetrieved>().firstOrNull()
        assertNotNull(chunksEvent)
        assertTrue("Reuso: marcado como reused", chunksEvent!!.reused)
        assertEquals(0L, chunksEvent.durationMs)

        val done = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull(done)
        assertTrue("Reuso registrado nas métricas", done!!.metrics.chunksReused)
    }

    @Test
    fun podaDeHistorico_quandoExcedeOrcamentoTokens() {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            maxPromptTokens = 300, // limite baixo p/ forçar poda
            maxTurnsT2 = 3
        )

        val longAnswer = "Texto longo com muitas palavras detalhando normas regulamentadoras de segurança ".repeat(10)
        val history = listOf(
            ConversationTurn(question = "Pergunta 1", answer = longAnswer),
            ConversationTurn(question = "Pergunta 2", answer = longAnswer),
            ConversationTurn(question = "Pergunta 3", answer = longAnswer)
        )

        val pruned = pipeline.buildRenderedHistory(history, "Pergunta atual do operário")
        assertTrue("O histórico deve ter sido podado", pruned.size < history.size)
    }

    @Test
    fun sanitizacaoDeBuscaFts5_comNormasETracos() {
        val retriever = Fts5Retriever()

        val queryWithHyphen = retriever.sanitizeFts5Query("NR-10 desenergização 10.5.1")
        assertTrue("Deve conter aspas em nr-10", queryWithHyphen.contains("\"nr-10\"*"))
        assertTrue("Deve conter aspas em 10.5.1", queryWithHyphen.contains("\"10.5.1\"*"))
        assertTrue("Deve remover acentuação de desenergização", queryWithHyphen.contains("desene*"))

        val queryWithStopwords = retriever.sanitizeFts5Query("como fazer a desenergização com segurança")
        assertFalse(queryWithStopwords.contains("como"))
        assertFalse(queryWithStopwords.contains("seguranca"))
        assertTrue(queryWithStopwords.contains("desene*"))
    }

    @Test
    fun telemetria_registraDecisaoDeToolCalling() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm(mode = FakeLlmMode.FORCE_TOOL)
        val pipeline = AskPipeline(retriever = retriever, llmEngine = llm, telemetryLogger = telemetryLogger)

        pipeline.execute("Qual a regra para desenergização na NR-10?", asrMs = 1500L).toList()

        assertEquals(1, telemetryLogger.getLogCount())
        val logContent = telemetryFile.readText()
        assertTrue("Log deve conter asr_ms", logContent.contains("\"asr_ms\":1500"))
        assertTrue("Log deve conter called_tool", logContent.contains("\"called_tool\":"))
        assertTrue("Log deve conter tool_consulta", logContent.contains("\"tool_consulta\":"))
        assertTrue("Log deve conter reason_ms", logContent.contains("\"reason_ms\":"))
        assertTrue("Log deve conter chunks_used", logContent.contains("\"chunks_used\":"))
    }
}
