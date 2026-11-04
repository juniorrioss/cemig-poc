package br.org.ceia.cemigpoc.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import br.org.ceia.cemigpoc.BuildConfig
import br.org.ceia.cemigpoc.data.acceptance.AcceptanceRunner
import br.org.ceia.cemigpoc.data.engine.RealAsrEngine
import br.org.ceia.cemigpoc.data.engine.RealNemotronAsrEngine
import br.org.ceia.cemigpoc.data.engine.RealLlamaEngine
import br.org.ceia.cemigpoc.data.model.ModelFileManager
import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import br.org.ceia.cemigpoc.data.retriever.ContextCompressor
import br.org.ceia.cemigpoc.data.retriever.DenseMatrix
import br.org.ceia.cemigpoc.data.retriever.DenseRetriever
import br.org.ceia.cemigpoc.data.retriever.ManualIndex
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.llama.LlamaEmbedder
import br.org.ceia.cemigpoc.data.telemetry.TelemetryLogger
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.AsrStatus
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
    val currentChunks: List<Chunk> = emptyList(),
    val isListening: Boolean = false,
    // true entre o Final do ASR e a confirmação do operário: a transcrição está
    // exibida e editável, aguardando o operário revisar antes de enviar ao pipeline.
    val awaitingConfirmation: Boolean = false,
    val isModelReady: Boolean = false,
    val isDebugMode: Boolean = false,
    val modelStatusMessage: String = "Inicializando motores nativos ARM64...",
    val errorMessage: String? = null,
    // F7: perguntas-exemplo do benchmark operacional (asset example_questions.json).
    val exampleQuestions: List<ExampleQuestion> = emptyList()
)

/** Pergunta-exemplo clicável (kind = norma | manual; source = norma/manual esperado). */
data class ExampleQuestion(val id: String, val kind: String, val source: String, val question: String)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainViewModel"
    }

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val fileManager = ModelFileManager(application.applicationContext)
    // Estágio 1 do pipeline híbrido: classificador leve de NR (TF-IDF+LogReg, Kotlin puro).
    private val nrClassifier = NrClassifier.load(application.applicationContext)
    // Retrieval v3: encoder de embeddings on-device + busca densa (fusão RRF 3-sinais).
    // RAM medida no S24+ (dumpsys): encoder RESIDENTE -> pico 2.44 GB PSS / 2.58 GB RSS;
    // encoder ON-DEMAND (carrega+libera por pergunta) -> pico 1.92 GB PSS / 2.02 GB RSS.
    //
    // DECISÃO POR NÚMERO (ver README_V3_INTEGRATION.md):
    // - Residente: E2E 10/10 <=10s, MÉDIA 7,6 s (encode 20-40 ms). RAM 2.44 GB cabe folgada
    //   no S24+ (12 GB) e em qualquer aparelho >=8 GB -> DEFAULT.
    // - On-demand: RAM -520 MB (cabe no S21 6 GB coexistindo com Whisper ~197 MB + LFM 1.4
    //   GB), MAS o cold-load de ~2,1 s/pergunta sobe a média p/ ~9,1 s e estoura 10 s em
    //   2/10. Fallback SÓ para aparelhos <8 GB, aceitando a latência maior.
    private val embedder = LlamaEmbedder()
    private val useLazyEncoder = false
    private var embedModelPathCache: String? = null
    private val denseRetriever = DenseRetriever(
        context = application.applicationContext,
        embedder = embedder,
        lazyEncoder = useLazyEncoder,
        embedModelPath = null  // preenchido após resolver o arquivo (init)
    )
    private val fts5 = Fts5Retriever(application.applicationContext)
    // F7: manuais + compressor do ponto de operação; criados no init (assets copiados) e
    // injetados no retriever. Se faltarem, o pipeline cai no RRF v4 top-2 (fallback honesto).
    @Volatile
    private var retriever = HybridRetriever(
        delegate = fts5,
        classifier = nrClassifier,
        denseRetriever = denseRetriever
    )
    private val realLlamaEngine = RealLlamaEngine()
    // Motor ASR selecionável por build (Parte 6 do poc-sft-v3): Nemotron 3.5 (sherpa-onnx) ou
    // Whisper Base. Ambos implementam AsrEngine; o Nemotron só é usado se BuildConfig pedir E
    // os pesos externos existirem (senão cai no Whisper — fallback honesto).
    private val useNemotron = BuildConfig.USE_NEMOTRON_ASR
    private val realAsrEngine = RealAsrEngine()
    private val nemotronAsrEngine = if (useNemotron) RealNemotronAsrEngine() else null
    // Motor ativo (resolvido no init conforme disponibilidade dos pesos).
    @Volatile
    private var activeAsrEngine: br.org.ceia.cemigpoc.domain.engine.AsrEngine = realAsrEngine
    private val telemetryLogger = TelemetryLogger(File(application.applicationContext.filesDir, "telemetry.jsonl"))

    // System prompt de tool-calling IDÊNTICO ao do treino (asset tools_system_prompt.txt,
    // SHA256 363185b7…). Lido dos assets no construtor; se faltar, usa o fallback conciso
    // (nunca deve faltar — está force-adicionado aos assets).
    private val useLfm3 = BuildConfig.USE_LFM3
    private val toolSystemPrompt: String = runCatching {
        // LFM3 (F7): system BAKED do f4-r32-ep2 (tool_schema_f4.baked_system_content, SHA256 921e6a1f…).
        application.applicationContext.assets.open(if (useLfm3) "lfm3_tools_system_prompt.txt" else "tools_system_prompt.txt")
            .bufferedReader(Charsets.UTF_8).use { it.readText() }
    }.getOrElse {
        Log.e(TAG, "Falha ao ler tools_system_prompt.txt dos assets; usando fallback", it)
        AskPipeline.SYNTHESIS_SYSTEM_PROMPT
    }

    // LFM3: n_ctx 4096 (prompt de síntese do ponto op ~1100 tok + histórico + até 384 gerados;
    // o system LFM3 sozinho tem ~550 tok). LFM2.5 legado: 2048.
    private val ctxSize = if (useLfm3) 4096 else 2048

    @Volatile
    private var askPipeline = buildPipeline()

    private fun buildPipeline() = AskPipeline(
        retriever = retriever,
        llmEngine = realLlamaEngine,
        telemetryLogger = telemetryLogger,
        toolSystemPrompt = toolSystemPrompt,
        maxTurnsT2 = 3,
        maxPromptTokens = if (useLfm3) 3200 else 1700,
        topK = 2,
        format = if (useLfm3) AskPipeline.ToolFormat.LFM3 else AskPipeline.ToolFormat.LFM25
    )

    private var asrStartTimeMs = 0L
    private var lastAsrDurationMs = 0L
    // Id da gravação corrente aceita pela UI. Eventos de ASR de gravações antigas
    // (recordingId != este) são descartados — trava anti-reprocessamento do S24+.
    @Volatile
    private var activeRecordingId = 0L

    init {
        // F7: perguntas-exemplo (benchmark operacional, aprovadas nas 3 rodadas da F6).
        runCatching {
            val arr = org.json.JSONArray(
                application.applicationContext.assets.open("example_questions.json")
                    .bufferedReader(Charsets.UTF_8).use { it.readText() }
            )
            val list = (0 until arr.length()).map {
                val o = arr.getJSONObject(it)
                ExampleQuestion(o.getString("id"), o.getString("kind"), o.getString("source"), o.getString("question"))
            }
            _uiState.update { it.copy(exampleQuestions = list) }
        }.onFailure { Log.w(TAG, "example_questions.json ausente", it) }

        // F7: gancho do E2E no aparelho (broadcast RUN_F7_E2E) — usa ESTE pipeline carregado.
        br.org.ceia.cemigpoc.data.acceptance.AcceptanceReceiver.f7Hook = { ids, keepPrefix ->
            viewModelScope.launch(Dispatchers.IO) {
                while (!_uiState.value.isModelReady) kotlinx.coroutines.delay(500)
                keepPrefix?.let { realLlamaEngine.engine.keepPrefix = it }
                Log.i(TAG, "F7 E2E: keepPrefix=${realLlamaEngine.engine.keepPrefix}")
                try {
                    br.org.ceia.cemigpoc.data.acceptance.F7E2ERunner.run(getApplication(), askPipeline, ids)
                } catch (e: Throwable) {
                    Log.e(TAG, "F7 E2E falhou", e)
                }
            }
        }

        // Inicialização assíncrona dos motores nativos (Whisper Base Q5_1 + Liquid LFM2.5)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _uiState.update { it.copy(modelStatusMessage = "Preparando índice de normas (36 NRs)...") }
                val dbFile = fileManager.getFts5DatabaseFile()
                Log.i(TAG, "Índice FTS5 pronto em: ${dbFile.absolutePath}")

                // Motor ASR: Nemotron 3.5 (sherpa-onnx) se pedido por build E os pesos externos
                // existirem; senão Whisper Base (fallback honesto).
                val nemotronDir = if (useNemotron) fileManager.getNemotronModelDir() else null
                val asrOk: Boolean
                if (useNemotron && nemotronDir != null && nemotronAsrEngine != null) {
                    _uiState.update { it.copy(modelStatusMessage = "Carregando ASR Nemotron 3.5 INT8...") }
                    asrOk = nemotronAsrEngine.engine.initModel(nemotronDir.absolutePath, numThreads = 4)
                    if (asrOk) {
                        activeAsrEngine = nemotronAsrEngine
                        Log.i(TAG, "ASR ativo: Nemotron 3.5 INT8 (sherpa-onnx)")
                    } else {
                        Log.e(TAG, "Falha ao inicializar Nemotron; caindo no Whisper Base")
                    }
                } else {
                    if (useNemotron) Log.w(TAG, "Nemotron pedido mas pesos ausentes; usando Whisper Base")
                    asrOk = false
                }
                // Whisper Base como motor principal (default) ou fallback do Nemotron.
                val whisperOk: Boolean
                if (activeAsrEngine === realAsrEngine) {
                    _uiState.update { it.copy(modelStatusMessage = "Carregando modelo ASR Whisper Base Q5_1...") }
                    val asrFile = fileManager.getAsrModelFile { progress ->
                        _uiState.update {
                            it.copy(modelStatusMessage = "Copiando Whisper Base: ${(progress * 100).toInt()}%")
                        }
                    }
                    whisperOk = realAsrEngine.engine.initModel(asrFile.absolutePath)
                    if (!whisperOk) Log.e(TAG, "Falha ao inicializar Whisper Base")
                } else {
                    whisperOk = false
                }
                val anyAsrOk = asrOk || whisperOk

                _uiState.update { it.copy(modelStatusMessage = if (useLfm3) "Carregando LFM3 1.7B (f4-r32-ep2 Q4_0)..." else "Carregando modelo LFM2.5 1.2B Instruct...") }
                val llmFile = fileManager.getLlmModelFile { progress ->
                    _uiState.update {
                        it.copy(modelStatusMessage = "Copiando modelo Q4: ${(progress * 100).toInt()}%")
                    }
                }
                // Thinking-OFF (task poc-engine-upgrade): configurado por build. O 2.6B força
                // <think> no template; a supressão (template + logit-bias no JNI) o torna
                // viável para voz. No 1.2B default o flag é false (no-op). Setar ANTES de gerar.
                realLlamaEngine.engine.suppressReasoning = BuildConfig.SUPPRESS_REASONING
                if (useLfm3) {
                    // Sampling da série LFM3 (treino/avaliação F4-F6 e smoke da F7):
                    // temp 0.1 / top_k 50 / rep 1.1 / min_p 0. O `<think></think>` já vem no
                    // prompt (Lfm3ToolFormat), então não há supressão extra.
                    realLlamaEngine.engine.repPenalty = 1.1f
                    realLlamaEngine.engine.minP = 0f
                    realLlamaEngine.engine.keepPrefix = true
                }
                val llmOk = realLlamaEngine.engine.load(
                    modelPath = llmFile.absolutePath,
                    ctxSize = ctxSize,
                    nThreads = 6
                )
                if (!llmOk) {
                    Log.e(TAG, "Falha ao inicializar LFM2.5")
                }
                Log.i(TAG, "Sintetizador: ${ModelFileManager.LLM_MODEL_NAME} (thinking-OFF=${BuildConfig.SUPPRESS_REASONING})")

                // Retrieval v3: prepara o encoder de embeddings + índices densos. Se algo
                // falhar, o HybridRetriever degrada para BM25-gated (fallback honesto).
                _uiState.update { it.copy(modelStatusMessage = "Preparando encoder denso (EmbeddingGemma)...") }
                val embedFile = fileManager.getEmbedModelFile { progress ->
                    _uiState.update {
                        it.copy(modelStatusMessage = "Copiando EmbeddingGemma: ${(progress * 100).toInt()}%")
                    }
                }
                embedModelPathCache = embedFile.absolutePath
                denseRetriever.embedModelPath = embedFile.absolutePath
                // Índices densos (.bin ~13.5 MB) sempre carregados; o ENCODER é carregado
                // sob demanda quando useLazyEncoder=true (economiza ~300 MB de pico).
                val denseOk = denseRetriever.load()
                if (!useLazyEncoder) {
                    val embedOk = embedder.load(embedFile.absolutePath)
                    if (!embedOk) Log.w(TAG, "Encoder denso não carregou; BM25-gated será usado")
                }
                if (!denseOk) {
                    Log.w(TAG, "Busca densa indisponível; pipeline usará BM25-gated (fallback)")
                }

                // F7: manuais (FTS5 + denso próprios) + compressor de contexto (ponto op F6).
                if (useLfm3) {
                    _uiState.update { it.copy(modelStatusMessage = "Preparando manuais e compressor de contexto...") }
                    try {
                        val manDb = fileManager.getAssetFile(ModelFileManager.MANUALS_DB)
                        val manDense = DenseMatrix.load(fileManager.getAssetFile(ModelFileManager.MANUALS_DENSE_BIN))
                        val manual = ManualIndex(manDb, manDense) { fts5.sanitizeFts5Query(it) }
                        val comp = ContextCompressor.load(
                            fileManager.getAssetFile(ModelFileManager.UNITS_BIN),
                            fileManager.getAssetFile(ModelFileManager.UNITS_VEC)
                        )
                        retriever = HybridRetriever(
                            delegate = fts5, classifier = nrClassifier, denseRetriever = denseRetriever,
                            manualIndex = manual, compressor = comp
                        )
                        askPipeline = buildPipeline()
                        Log.i(TAG, "Ponto de operação F6 pronto: manuais=${manDense.count} janelas=${comp.dim}d")
                    } catch (e: Exception) {
                        Log.e(TAG, "Falha ao preparar manuais/compressor; usando RRF v4 top-2", e)
                    }
                }

                val asrLabel = if (activeAsrEngine === nemotronAsrEngine) "Nemotron 3.5" else "Whisper"
                _uiState.update {
                    it.copy(
                        isModelReady = anyAsrOk && llmOk,
                        modelStatusMessage = if (anyAsrOk && llmOk) "Pronto ($asrLabel + ${if (useLfm3) "LFM3 1.7B" else "LFM2.5"})" else "Falha no carregamento dos modelos",
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

        // Observa eventos contínuos do ASR.
        //
        // Correção do bug do S24+ (task poc-asr-fix):
        // 1) Descarta qualquer evento cujo recordingId != gravação corrente, então uma
        //    transcrição antiga NUNCA é reaplicada nem reprocessada.
        // 2) STATUS (Status) só mexe no indicador de estágio, jamais em currentQuestionInput.
        // 3) CONTEÚDO (Partial/Final) vai para currentQuestionInput; o Final NÃO dispara
        //    processQuestion automaticamente — o operário revisa/edita e confirma (botão
        //    Enviar / submitQuestion). Assim a resposta sempre corresponde à fala atual.
        // Coleta os eventos dos DOIS motores (só o ativo emite, pois só nele chamamos
        // startListening). Assim independe de qual motor o init resolveu (Nemotron/Whisper).
        viewModelScope.launch {
            realAsrEngine.events.collect { event -> handleAsrEvent(event) }
        }
        nemotronAsrEngine?.let { neng ->
            viewModelScope.launch {
                neng.events.collect { event -> handleAsrEvent(event) }
            }
        }
    }

    /** Processa um evento de ASR (comum aos motores Whisper e Nemotron). */
    private fun handleAsrEvent(event: AsrEvent) {
        // Trava anti-reprocessamento: ignora eventos de gravações que não são a corrente.
        if (event.recordingId != activeRecordingId) {
            Log.d(TAG, "Evento ASR obsoleto ignorado (rec=${event.recordingId} != ativo=$activeRecordingId)")
            return
        }
        when (event) {
                    is AsrEvent.Status -> {
                        // Apenas STATUS -> indicador de estágio. Nunca escreve no texto.
                        val stage = when (event.stage) {
                            AsrStatus.RECORDING -> PipelineStage.LISTENING
                            AsrStatus.TRANSCRIBING -> PipelineStage.TRANSCRIBING
                        }
                        _uiState.update { it.copy(stage = stage) }
                    }
                    is AsrEvent.Partial -> {
                        // CONTEÚDO parcial -> preview editável, sem disparar o pipeline.
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.TRANSCRIBING,
                                currentQuestionInput = event.text
                            )
                        }
                    }
                    is AsrEvent.Final -> {
                        // CONTEÚDO final -> exibe para revisão. NÃO envia automaticamente.
                        lastAsrDurationMs = System.currentTimeMillis() - asrStartTimeMs
                        _uiState.update {
                            it.copy(
                                isListening = false,
                                awaitingConfirmation = true,
                                stage = PipelineStage.IDLE,
                                currentQuestionInput = event.text
                            )
                        }
                    }
                    is AsrEvent.Error -> {
                        _uiState.update {
                            it.copy(
                                isListening = false,
                                awaitingConfirmation = false,
                                stage = PipelineStage.ERROR,
                                errorMessage = event.message
                            )
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
        // Marca esta gravação como a corrente ANTES de emitir eventos: qualquer evento
        // pendente de uma gravação anterior será descartado no coletor pelo recordingId.
        activeRecordingId = activeAsrEngine.startListening()
        _uiState.update {
            it.copy(
                isListening = true,
                awaitingConfirmation = false,
                stage = PipelineStage.LISTENING,
                currentQuestionInput = "",
                errorMessage = null
            )
        }
    }

    /**
     * Encerra gravação Push-to-Talk.
     */
    fun stopPushToTalk() {
        activeAsrEngine.stopListening()
    }

    /**
     * Envia a pergunta digitada ou editada pelo operário.
     */
    fun submitQuestion() {
        val question = _uiState.value.currentQuestionInput.trim()
        if (question.isNotBlank()) {
            // Aproveita a latência de ASR medida se a pergunta veio da transcrição (voz).
            val asrMs = if (_uiState.value.awaitingConfirmation) lastAsrDurationMs else 0L
            _uiState.update { it.copy(awaitingConfirmation = false) }
            processQuestion(question, asrMs = asrMs)
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
                    stage = PipelineStage.CLASSIFYING,
                    currentQuestionInput = "",
                    awaitingConfirmation = false,
                    currentStreamingAnswer = "",
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
                    is TurnEvent.ToolDecided -> {
                        // Decisão de tool-calling do modelo (visível no Modo Engenharia).
                        Log.i(TAG, "ToolDecided: buscar=${turnEvent.called} consulta='${turnEvent.consulta}' nr='${turnEvent.nr}'")
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
                        val m = turnEvent.metrics
                        val newTurn = ConversationTurn(
                            question = question,
                            answer = turnEvent.finalAnswer,
                            chunks = turnEvent.chunksUsed,
                            metrics = m,
                            isStreaming = false,
                            calledTool = m.calledTool,
                            toolConsulta = m.toolConsulta.ifBlank { null },
                            toolNr = m.toolNr.ifBlank { null },
                            toolContext = turnEvent.toolContext
                        )
                        _uiState.update {
                            it.copy(
                                stage = PipelineStage.IDLE,
                                history = it.history + newTurn,
                                currentStreamingAnswer = "",
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
                awaitingConfirmation = false,
                currentStreamingAnswer = "",
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
            // Se o ASR ativo for o Nemotron 3.5, roteia a transcrição do roteiro por ele.
            val nemo = nemotronAsrEngine
            val transcribeOverride: (suspend (FloatArray) -> String)? =
                if (nemo != null && activeAsrEngine === nemo) { s -> nemo.transcribeAudioSamples(s) } else null
            AcceptanceRunner.run(
                context = getApplication(),
                existingAsr = realAsrEngine,
                existingLlama = realLlamaEngine,
                existingRetriever = retriever,
                transcribeOverride = transcribeOverride
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        realLlamaEngine.engine.close()
        realAsrEngine.engine.close()
        nemotronAsrEngine?.engine?.close()
        embedder.close()
    }
}
