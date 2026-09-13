package br.org.ceia.cemigpoc.data.engine

import android.util.Log
import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.AsrStatus
import br.org.ceia.cemigpoc.sherpa.NemotronAsrEngine
import br.org.ceia.cemigpoc.whisper.AudioRecordRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Implementação do motor ASR usando o Nemotron 3.5 streaming INT8 (sherpa-onnx), alternativa
 * ao Whisper Base ([RealAsrEngine]) selecionável por build (BuildConfig.USE_NEMOTRON_ASR).
 *
 * Reusa o [AudioRecordRecorder] do módulo :whisper (16 kHz mono PCM) — a captura é idêntica;
 * só o motor de transcrição muda. Mantém o MESMO contrato anti-reprocessamento do S24+ que o
 * [RealAsrEngine]: canal sem replay (replay=0), recordingId monotônico, STATUS separado do
 * CONTEÚDO, e Final NÃO auto-dispara o pipeline (o operário confirma).
 *
 * Comentários PT-BR; código em inglês.
 */
class RealNemotronAsrEngine(
    val engine: NemotronAsrEngine = NemotronAsrEngine(),
    private val recorder: AudioRecordRecorder = AudioRecordRecorder(),
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
) : AsrEngine {

    companion object {
        private const val TAG = "RealNemotronAsrEngine"
        // Mínimo de áudio (300 ms a 16 kHz) para valer a pena transcrever.
        private const val MIN_SAMPLES = 4800
    }

    private val _events = MutableSharedFlow<AsrEvent>(replay = 0, extraBufferCapacity = 8)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()

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
        recorder.start()
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
                val samples = recorder.stop()
                if (samples.size < MIN_SAMPLES) {
                    Log.w(TAG, "Áudio muito curto (${samples.size} amostras)")
                    _events.emit(
                        AsrEvent.Error("Nenhuma fala detectada ou áudio muito curto.", recordingId = recId)
                    )
                    return@launch
                }
                val transcription = engine.transcribe(samples)
                Log.i(TAG, "Transcrição Nemotron (rec=$recId): '$transcription'")
                if (transcription.isBlank()) {
                    _events.emit(
                        AsrEvent.Error("Nenhuma fala detectada ou áudio muito curto.", recordingId = recId)
                    )
                } else {
                    _events.emit(AsrEvent.Final(transcription, recId))
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Erro na transcrição Nemotron (rec=$recId)", e)
                _events.emit(AsrEvent.Error("Falha na transcrição: ${e.message}", e, recId))
            }
        }
    }

    suspend fun transcribeAudioSamples(samples: FloatArray): String {
        return engine.transcribe(samples)
    }
}
