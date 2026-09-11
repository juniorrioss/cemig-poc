package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.fake.FakeLlm
import br.org.ceia.cemigpoc.data.fake.FakeRetriever
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
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
 * Bateria de testes de unidade JVM do AskPipeline (pipeline consolidado, sem Turno 1 de rewrite):
 * 1. Multiturno com histórico bruto no orçamento da síntese;
 * 2. Reuso de chunks via similaridade Jaccard (> 0.7) entre falas BRUTAS consecutivas;
 * 3. Poda estrita de orçamento na síntese (<= 1000 tokens);
 * 4. Sanitização FTS5 de normas com traços e diacríticos;
 * 5. Telemetria local com breakdown de tempos por etapa.
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
    fun multiturno_continuaComHistoricoNaSintese() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger,
            maxTurnsT2 = 3
        )

        // Histórico prévio da conversa
        val turn1 = ConversationTurn(
            question = "Como faço a desenergização do painel?",
            answer = "Segundo a NR-10, item 10.5.1, você deve seguir o seccionamento e impedimento de reenergização.",
            chunks = listOf(
                Chunk(id = 1, doc = "NR-10", section = "10.5.1", content = "Etapas de desenergização...", score = 1.2)
            )
        )
        val history = listOf(turn1)

        // Pergunta de continuação dependente de contexto
        val continuationQuestion = "E qual o próximo passo depois do seccionamento?"

        val events = pipeline.execute(continuationQuestion, history = history).toList()

        val doneEvent = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull("Deveria emitir evento Done final", doneEvent)
        assertTrue("A resposta deve ser fundamentada", doneEvent!!.finalAnswer.isNotBlank())
        assertTrue("Deve citar norma regulamentadora", doneEvent.finalAnswer.contains("NR-10"))
    }

    @Test
    fun reusoDeChunksPorJaccard_quandoFalasBrutasSemelhantes() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger,
            jaccardThreshold = 0.7
        )

        val previousChunk = Chunk(id = 42, doc = "NR-10", section = "Item 10.5.1", content = "Regras de desenergização", score = 0.8)
        val prevQuestion = "Quais as etapas de desenergização da NR-10 no painel?"
        val turn1 = ConversationTurn(
            question = prevQuestion,
            answer = "Conforme NR-10 item 10.5.1...",
            chunks = listOf(previousChunk)
        )
        val history = listOf(turn1)

        // Nova fala BRUTA quase idêntica à anterior (Jaccard > 0.7) -> reuso de chunks
        val events = pipeline.execute("Quais as etapas de desenergização da NR-10 no painel agora?", history = history).toList()

        val chunksEvent = events.filterIsInstance<TurnEvent.ChunksRetrieved>().firstOrNull()
        assertNotNull(chunksEvent)
        assertTrue("Deveria reutilizar os chunks do turno anterior", chunksEvent!!.reused)
        assertEquals(1, chunksEvent.chunks.size)
        assertEquals(42L, chunksEvent.chunks.first().id)

        val doneEvent = events.filterIsInstance<TurnEvent.Done>().firstOrNull()
        assertNotNull(doneEvent)
        assertTrue("Métricas devem registrar reuso de chunks", doneEvent!!.metrics.chunksReused)
    }

    @Test
    fun semReusoDeChunks_quandoFalasBrutasDiferentes() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger,
            jaccardThreshold = 0.7
        )

        val previousChunk = Chunk(id = 10, doc = "NR-10", section = "10.4.1", content = "Regras elétricas", score = 0.5)
        val turn1 = ConversationTurn(
            question = "Quais as etapas de desenergização?",
            answer = "Conforme NR-10...",
            chunks = listOf(previousChunk)
        )
        val history = listOf(turn1)

        val events = pipeline.execute("Qual o EPI para subir no poste?", history = history).toList()

        val chunksEvent = events.filterIsInstance<TurnEvent.ChunksRetrieved>().firstOrNull()
        assertNotNull(chunksEvent)
        assertFalse("Não deve reutilizar chunks quando o tema diverge", chunksEvent!!.reused)
        assertTrue("Busca BM25 deve levar tempo maior ou igual a zero", chunksEvent.durationMs >= 0L)
    }

    @Test
    fun podaDeOrcamento_quandoHistoricoExcedeLimiteTokens() {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            t2MaxBudgetTokens = 300 // Limite artificialmente baixo para testar poda
        )

        // Cria 3 trocas longas que excedem 300 tokens
        val longAnswer = "Este é um texto longo com muitas palavras repetidas detalhando normas regulamentadoras de segurança do trabalho ".repeat(10)
        val turn1 = ConversationTurn(question = "Pergunta 1", answer = longAnswer)
        val turn2 = ConversationTurn(question = "Pergunta 2", answer = longAnswer)
        val turn3 = ConversationTurn(question = "Pergunta 3", answer = longAnswer)
        val fullHistory = listOf(turn1, turn2, turn3)

        val pruned = pipeline.pruneHistoryForBudget(
            history = fullHistory,
            systemPrompt = AskPipeline.SYNTHESIS_SYSTEM_PROMPT,
            userContent = "Pergunta atual do operário",
            maxBudgetTokens = 300
        )

        // Deve ter podado trocas antigas para caber no limite
        assertTrue("O histórico deve ter sido podado", pruned.size < fullHistory.size)
    }

    @Test
    fun sanitizacaoDeBuscaFts5_comNormasETracos() {
        val retriever = Fts5Retriever()

        // 1. Termos com traço devem receber aspas para evitar sintaxe inválida no SQLite FTS5
        val queryWithHyphen = retriever.sanitizeFts5Query("NR-10 desenergização 10.5.1")
        assertTrue("Deve conter aspas em nr-10", queryWithHyphen.contains("\"nr-10\"*"))
        assertTrue("Deve conter aspas em 10.5.1", queryWithHyphen.contains("\"10.5.1\"*"))
        assertTrue("Deve remover acentuação de desenergização", queryWithHyphen.contains("desene*"))

        // 2. Stopwords em português devem ser eliminadas
        val queryWithStopwords = retriever.sanitizeFts5Query("como fazer a desenergização com segurança")
        assertFalse(queryWithStopwords.contains("como"))
        assertFalse(queryWithStopwords.contains("seguranca"))
        assertTrue(queryWithStopwords.contains("desene*"))
    }

    @Test
    fun telemetria_registraMetricasDetalhadasPorTurno() = runTest {
        val retriever = FakeRetriever()
        val llm = FakeLlm()
        val pipeline = AskPipeline(
            retriever = retriever,
            llmEngine = llm,
            telemetryLogger = telemetryLogger
        )

        pipeline.execute("Qual a regra para desenergização?", asrMs = 1500L).toList()

        assertEquals(1, telemetryLogger.getLogCount())
        val logContent = telemetryFile.readText()
        assertTrue("Log deve conter asr_ms", logContent.contains("\"asr_ms\":1500"))
        assertTrue("Log deve conter search_ms", logContent.contains("\"search_ms\":"))
        assertTrue("Log deve conter gate_mode", logContent.contains("\"gate_mode\":"))
        assertTrue("Log deve conter ttft_ms", logContent.contains("\"ttft_ms\":"))
        assertTrue("Log deve conter chunks_used", logContent.contains("\"chunks_used\":"))
    }
}
