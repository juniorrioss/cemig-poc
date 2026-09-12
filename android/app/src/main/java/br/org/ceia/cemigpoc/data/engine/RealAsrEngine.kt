package br.org.ceia.cemigpoc.data.engine

import android.util.Log
import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.AsrStatus
import br.org.ceia.cemigpoc.whisper.WhisperCppEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Implementação real do motor ASR executando Whisper Base Q5_1 localmente via whisper.cpp.
 *
 * Correção do bug de reprocessamento do S24+ (task poc-asr-fix):
 * - Canal de eventos SEM replay (replay = 0): um coletor que reassina NÃO reobtém o
 *   último Final antigo, então uma transcrição anterior nunca é reprocessada.
 * - Cada gravação recebe um recordingId monotônico; eventos carregam esse id para o
 *   consumidor descartar o que não for da gravação corrente.
 * - STATUS (gravando/transcrevendo) é emitido como AsrEvent.Status, separado do
 *   CONTEÚDO transcrito (AsrEvent.Final) — nunca mais mistura estágio com texto.
 */
class RealAsrEngine(
    val engine: WhisperCppEngine = WhisperCppEngine(),
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
) : AsrEngine {

    companion object {
        private const val TAG = "RealAsrEngine"
    }

    // replay = 0: nenhum evento antigo é re-entregue a novos coletores.
    // extraBufferCapacity garante emissões não-suspensas a partir de escopos IO.
    private val _events = MutableSharedFlow<AsrEvent>(replay = 0, extraBufferCapacity = 8)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()

    // Sequência monotônica de gravações; o consumidor usa para ignorar eventos obsoletos.
    private val recordingSeq = AtomicLong(0L)

    @Volatile
    private var isListening = false

    @Volatile
    private var currentRecordingId = 0L

    override fun startListening(): Long {
        if (isListening) return currentRecordingId
        isListening = true
        currentRecordingId = recordingSeq.incrementAndGet()
        val recId = currentRecordingId
        engine.startRecording()

        scope.launch {
            _events.emit(AsrEvent.Status(AsrStatus.RECORDING, recId))
        }
        return recId
    }

    override fun stopListening() {
        if (!isListening) return
        isListening = false
        val recId = currentRecordingId

        scope.launch {
            _events.emit(AsrEvent.Status(AsrStatus.TRANSCRIBING, recId))
            try {
                val transcription = engine.transcribeRecordedAudio(language = "pt")
                Log.i(TAG, "Transcrição final (rec=$recId): '$transcription'")
                if (transcription.isBlank()) {
                    _events.emit(
                        AsrEvent.Error("Nenhuma fala detectada ou áudio muito curto.", recordingId = recId)
                    )
                } else {
                    _events.emit(AsrEvent.Final(transcription, recId))
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Erro na transcrição Whisper (rec=$recId)", e)
                _events.emit(AsrEvent.Error("Falha na transcrição: ${e.message}", e, recId))
            }
        }
    }

    suspend fun transcribeAudioSamples(samples: FloatArray): String {
        return engine.transcribe(samples, language = "pt")
    }
}
