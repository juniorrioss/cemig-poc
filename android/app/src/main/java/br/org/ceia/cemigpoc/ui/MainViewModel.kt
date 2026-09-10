package br.org.ceia.cemigpoc.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import br.org.ceia.cemigpoc.data.acceptance.AcceptanceRunner
import br.org.ceia.cemigpoc.data.engine.RealAsrEngine
import br.org.ceia.cemigpoc.data.engine.RealLlamaEngine
import br.org.ceia.cemigpoc.data.model.ModelFileManager
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

/**
 * Estado imutável da interface principal conversacional.
 */
data class MainUiState(
    val stage: PipelineStage = PipelineStage.IDLE,
    val history: List<ConversationTurn> = emptyList(),
    val currentQuestionInput: String = "",
    val currentStreamingAnswer: String = "",
    val currentKeywords: String = "",
    val currentChunks: List<Chunk> = emptyList(),
    val isListening: Boolean = false,
    val isModelReady: Boolean = false,
    val isDebugMode: Boolean = false,
    val modelStatusMessage: String = "Inicializando motores nativos ARM64...",
    val errorMessage: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainViewModel"
    }

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val fileManager = ModelFileManager(application.applicationContext)
    private val retriever = Fts5Retriever(application.applicationContext)
    private val realLlamaEngine = RealLlamaEngine()
    private val realAsrEngine = RealAsrEngine()
    private val telemetryLogger = TelemetryLogger(File(application.applicationContext.filesDir, "telemetry.jsonl"))

    private val askPipeline = AskPipeline(
        retriever = retriever,
        llmEngine = realLlamaEngine,
        telemetryLogger = telemetryLogger,
        maxTurnsT1 = 6,
        maxTurnsT2 = 3,
        t2MaxBudgetTokens = 1000,
        jaccardThreshold = 0.7,
        topK = 2
    )

    private var asrStartTimeMs = 0L
    private var lastAsrDurationMs = 0L

    init {
        // Inicialização assíncrona dos motores nativos (Whisper Base Q5_1 + Liquid LFM2.5)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _uiState.update { it.copy(modelStatusMessage = "Preparando índice de normas (36 NRs)...") }
                val dbFile = fileManager.getFts5DatabaseFile()
                Log.i(TAG, "Índice FTS5 pronto em: ${dbFile.absolutePath}")

                _uiState.update { it.copy(modelStatusMessage = "Carregando modelo ASR Whisper Base Q5_1...") }
                val asrFile = fileManager.getAsrModelFile { progress ->
                    _uiState.update {
                        it.copy(modelStatusMessage = "Copiando Whisper Base: ${(progress * 100).toInt()}%")
                    }
                }
                val asrOk = realAsrEngine.engine.initModel(asrFile.absolutePath)
                if (!asrOk) {
                    Log.e(TAG, "Falha ao inicializar Whisper Base")
                }

                _uiState.update { it.copy(modelStatusMessage = "Carregando modelo LFM2.5 1.2B Instruct...") }
                val llmFile = fileManager.getLlmModelFile { progress ->
                    _uiState.update {
                        it.copy(modelStatusMessage = "Copiando LFM2.5 Q4: ${(progress * 100).toInt()}%")
                    }
                }
                val llmOk = realLlamaEngine.engine.load(
                    modelPath = llmFile.absolutePath,
                    ctxSize = 2048,
                    nThreads = 6
                )
                if (!llmOk) {
                    Log.e(TAG, "Falha ao inicializar LFM2.5")
                }

                _uiState.update {
                    it.copy(
                        isModelReady = asrOk && llmOk,
                        modelStatusMessage = if (asrOk && llmOk) "Pronto (Whisper + LFM2.5)" else "Falha no carregamento dos modelos",
                        stage = PipelineStage.IDLE
                    )
                }
                Log.i(TAG, "Inicialização dos motores concluída com sucesso")
            } catch (e: Exception) {
                Log.e(TAG, "Erro na inicialização dos motores", e)
                _uiState.update {
                    it.copy(
                        isModelReady = false,
                        modelStatusMessage = "Erro: ${e.message}",
                        errorMessage = e.message
                    )
                }
            }
        }

        // Observa eventos contínuos do ASR
        viewModelScope.launch {
            realAsrEngine.events.collect { event ->
                when (event) {
                    is AsrEvent.Partial -> {
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.TRANSCRIBING,
                                currentQuestionInput = event.text
                            )
                        }
                    }
                    is AsrEvent.Final -> {
                        lastAsrDurationMs = System.currentTimeMillis() - asrStartTimeMs
                        _uiState.update {
                            it.copy(
                                isListening = false,
                                stage = PipelineStage.REWRITING,
                                currentQuestionInput = event.text
                            )
                        }
                        processQuestion(event.text, asrMs = lastAsrDurationMs)
                    }
                    is AsrEvent.Error -> {
                        _uiState.update {
                            it.copy(
                                isListening = false,
                                stage = PipelineStage.ERROR,
                                errorMessage = event.message
                            )
                        }
                    }
                }
            }
        }
    }

    /**
     * Atualiza o texto editável da caixa de entrada do operário.
     */
    fun onQuestionInputChanged(newText: String) {
        _uiState.update { it.copy(currentQuestionInput = newText) }
    }

    /**
     * Alterna o Modo Engenharia / Debug inline na tela.
     */
    fun toggleDebugMode() {
        _uiState.update { it.copy(isDebugMode = !it.isDebugMode) }
    }

    /**
     * Inicia gravação Push-to-Talk via microfone.
     */
    fun startPushToTalk() {
        if (!_uiState.value.isModelReady) return

        asrStartTimeMs = System.currentTimeMillis()
        _uiState.update {
            it.copy(
                isListening = true,
                stage = PipelineStage.LISTENING,
                errorMessage = null
            )
        }
        realAsrEngine.startListening()
    }

    /**
     * Encerra gravação Push-to-Talk.
     */
    fun stopPushToTalk() {
        realAsrEngine.stopListening()
    }

    /**
     * Envia a pergunta digitada ou editada pelo operário.
     */
    fun submitQuestion() {
        val question = _uiState.value.currentQuestionInput.trim()
        if (question.isNotBlank()) {
            processQuestion(question, asrMs = 0L)
        }
    }

    /**
     * Processa a pergunta completa pelo pipeline multiturno.
     */
    fun processQuestion(question: String, asrMs: Long = 0L) {
        if (question.isBlank()) return

        val currentHistory = _uiState.value.history

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    stage = PipelineStage.REWRITING,
                    currentQuestionInput = "",
                    currentStreamingAnswer = "",
                    currentKeywords = "",
                    currentChunks = emptyList(),
                    errorMessage = null
                )
            }

            val streamingAccumulator = StringBuilder()

            askPipeline.execute(
                userQuestion = question,
                history = currentHistory,
                asrMs = asrMs
            ).collect { turnEvent ->
                when (turnEvent) {
                    is TurnEvent.StageChanged -> {
                        _uiState.update { it.copy(stage = turnEvent.stage) }
                    }
                    is TurnEvent.KeywordsExtracted -> {
                        _uiState.update { it.copy(currentKeywords = turnEvent.keywords) }
                    }
                    is TurnEvent.ChunksRetrieved -> {
                        _uiState.update { it.copy(currentChunks = turnEvent.chunks) }
                    }
                    is TurnEvent.TextDelta -> {
                        streamingAccumulator.append(turnEvent.text)
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.RESPONDING,
                                currentStreamingAnswer = streamingAccumulator.toString()
                            )
                        }
                    }
                    is TurnEvent.Done -> {
                        val newTurn = ConversationTurn(
                            question = question,
                            answer = turnEvent.finalAnswer,
                            keywords = turnEvent.metrics.keywords,
                            chunks = turnEvent.chunksUsed,
                            metrics = turnEvent.metrics,
                            isStreaming = false
                        )
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.IDLE,
                                history = it.history + newTurn,
                                currentStreamingAnswer = "",
                                currentKeywords = "",
                                currentChunks = emptyList()
                            )
                        }
                    }
                    is TurnEvent.Error -> {
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.ERROR,
                                errorMessage = turnEvent.message
                            )
                        }
                    }
                    else -> {}
                }
            }
        }
    }

    /**
     * Inicia uma nova conversa: limpa histórico de trocas e reseta cache KV do Llama.
     */
    fun newConversation() {
        realLlamaEngine.clearKvCache()
        _uiState.update {
            it.copy(
                stage = PipelineStage.IDLE,
                history = emptyList(),
                currentQuestionInput = "",
                currentStreamingAnswer = "",
                currentKeywords = "",
                currentChunks = emptyList(),
                errorMessage = null
            )
        }
        Log.i(TAG, "Nova conversa iniciada: histórico e KV cache limpos")
    }

    /**
     * Dispara o roteiro de aceitação de 10 perguntas faladas (Modo Avião).
     */
    fun runAcceptanceTest() {
        viewModelScope.launch(Dispatchers.IO) {
            Log.i(TAG, "runAcceptanceTest: Aguardando motores estarem prontos...")
            while (!_uiState.value.isModelReady) {
                kotlinx.coroutines.delay(500)
            }
            Log.i(TAG, "runAcceptanceTest: Motores prontos, iniciando AcceptanceRunner...")
            AcceptanceRunner.run(
                context = getApplication(),
                existingAsr = realAsrEngine,
                existingLlama = realLlamaEngine,
                existingRetriever = retriever
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        realLlamaEngine.engine.close()
        realAsrEngine.engine.close()
    }
}
