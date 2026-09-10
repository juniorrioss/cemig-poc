package br.org.ceia.cemigpoc.data.engine

import android.util.Log
import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.whisper.WhisperCppEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch

/**
 * Implementação real do motor ASR executando Whisper Base Q5_1 localmente via whisper.cpp.
 */
class RealAsrEngine(
    val engine: WhisperCppEngine = WhisperCppEngine(),
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
) : AsrEngine {

    companion object {
        private const val TAG = "RealAsrEngine"
    }

    private val _events = MutableSharedFlow<AsrEvent>(replay = 1)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()

    @Volatile
    private var isListening = false

    override fun startListening() {
        if (isListening) return
        isListening = true
        engine.startRecording()

        scope.launch {
            _events.emit(AsrEvent.Partial("Gravando áudio..."))
        }
    }

    override fun stopListening() {
        if (!isListening) return
        isListening = false

        scope.launch {
            _events.emit(AsrEvent.Partial("Transcrevendo via Whisper Base Q5_1..."))
            try {
                val transcription = engine.transcribeRecordedAudio(language = "pt")
                Log.i(TAG, "Transcrição final concluída: '$transcription'")
                if (transcription.isBlank()) {
                    _events.emit(AsrEvent.Error("Nenhuma fala detectada ou áudio muito curto."))
                } else {
                    _events.emit(AsrEvent.Final(transcription))
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Erro na transcrição Whisper", e)
                _events.emit(AsrEvent.Error("Falha na transcrição: ${e.message}", e))
            }
        }
    }

    suspend fun transcribeAudioSamples(samples: FloatArray): String {
        return engine.transcribe(samples, language = "pt")
    }
}
