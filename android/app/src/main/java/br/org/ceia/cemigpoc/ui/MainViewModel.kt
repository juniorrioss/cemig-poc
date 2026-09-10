package br.org.ceia.cemigpoc.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import br.org.ceia.cemigpoc.data.fake.FakeAsr
import br.org.ceia.cemigpoc.data.fake.FakeLlm
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.LlmEngine
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ToolCall
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

/**
 * Estados do fluxo operacional da interface.
 */
enum class AppStatus(val label: String) {
    PRONTO("OFFLINE · PRONTO"),
    OUVINDO("OUVINDO VOZ..."),
    BUSCANDO("CONSULTANDO NORMAS (BM25)..."),
    GERANDO("SINTETIZANDO RESPOSTA..."),
    CONCLUIDO("RESPOSTA CONCLUÍDA"),
    ERRO("ERRO NO PROCESSAMENTO")
}

/**
 * Estado da tela única do assistente.
 */
data class MainUiState(
    val status: AppStatus = AppStatus.PRONTO,
    val transcription: String = "",
    val finalTranscription: String = "",
    val isListening: Boolean = false,
    val streamingResponse: String = "",
    val activeToolCall: ToolCall? = null,
    val sources: List<Chunk> = emptyList(),
    val totalDurationMs: Long? = null,
    val errorMessage: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    // Inicialização dos motores e pipeline offline
    private val asrEngine: AsrEngine
    private val retriever: Retriever
    private val llmEngine: LlmEngine
    private val askPipeline: AskPipeline
    private val telemetryLogger: TelemetryLogger

    init {
        val context = application.applicationContext
        val telemetryFile = File(context.filesDir, "telemetry.jsonl")
        telemetryLogger = TelemetryLogger(telemetryFile)

        // Lê system prompt dos assets
        val systemPrompt = try {
            context.assets.open("system_prompt_pt.md").bufferedReader().use { it.readText() }
        } catch (_: Exception) {
            "Você é um assistente de normas técnicas. Use a ferramenta retriever para buscar regras."
        }

        // Fts5Retriever com fallback transparente
        retriever = Fts5Retriever(context)
        llmEngine = FakeLlm(streamDelayMs = 20L)
        asrEngine = FakeAsr()

        askPipeline = AskPipeline(
            systemPrompt = systemPrompt,
            retriever = retriever,
            llmEngine = llmEngine,
            telemetryLogger = telemetryLogger
        )

        // Observa eventos contínuos do ASR
        viewModelScope.launch {
            asrEngine.events.collect { event ->
                when (event) {
                    is AsrEvent.Partial -> {
                        _uiState.update { it.copy(transcription = event.text) }
                    }
                    is AsrEvent.Final -> {
                        _uiState.update {
                            it.copy(
                                transcription = event.text,
                                finalTranscription = event.text,
                                isListening = false
                            )
                        }
                        processQuestion(event.text)
                    }
                    is AsrEvent.Error -> {
                        _uiState.update {
                            it.copy(
                                isListening = false,
                                status = AppStatus.ERRO,
                                errorMessage = event.message
                            )
                        }
                    }
                }
            }
        }
    }

    /**
     * Invocado quando o operário pressiona e segura o botão PTT.
     */
    fun startPushToTalk() {
        _uiState.update {
            it.copy(
                isListening = true,
                status = AppStatus.OUVINDO,
                transcription = "",
                streamingResponse = "",
                activeToolCall = null,
                sources = emptyList(),
                totalDurationMs = null,
                errorMessage = null
            )
        }
        asrEngine.startListening()
    }

    /**
     * Invocado quando o operário solta o botão PTT.
     */
    fun stopPushToTalk() {
        asrEngine.stopListening()
    }

    /**
     * Processa uma consulta (via voz ou clique de exemplo).
     */
    fun processQuestion(question: String) {
        if (question.isBlank()) return

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    transcription = question,
                    finalTranscription = question,
                    streamingResponse = "",
                    activeToolCall = null,
                    sources = emptyList(),
                    status = AppStatus.BUSCANDO,
                    errorMessage = null
                )
            }

            val responseAccumulator = StringBuilder()

            askPipeline.execute(question).collect { turnEvent ->
                when (turnEvent) {
                    is TurnEvent.ToolCallEvent -> {
                        _uiState.update {
                            it.copy(
                                activeToolCall = turnEvent.call,
                                status = AppStatus.BUSCANDO
                            )
                        }
                    }
                    is TurnEvent.ToolResultEvent -> {
                        _uiState.update {
                            it.copy(
                                sources = turnEvent.chunks,
                                status = AppStatus.GERANDO
                            )
                        }
                    }
                    is TurnEvent.TextDelta -> {
                        responseAccumulator.append(turnEvent.text)
                        _uiState.update {
                            it.copy(
                                streamingResponse = responseAccumulator.toString(),
                                status = AppStatus.GERANDO
                            )
                        }
                    }
                    is TurnEvent.Done -> {
                        _uiState.update {
                            it.copy(
                                streamingResponse = turnEvent.finalAnswer,
                                sources = turnEvent.chunksUsed,
                                totalDurationMs = turnEvent.totalDurationMs,
                                status = AppStatus.CONCLUIDO
                            )
                        }
                    }
                    is TurnEvent.Error -> {
                        _uiState.update {
                            it.copy(
                                status = AppStatus.ERRO,
                                errorMessage = turnEvent.message
                            )
                        }
                    }
                }
            }
        }
    }

    /**
     * Reinicia para o estado inicial pronto.
     */
    fun resetState() {
        _uiState.update {
            MainUiState(status = AppStatus.PRONTO)
        }
    }
}
