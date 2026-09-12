package br.org.ceia.cemigpoc.data.fake

import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.AsrStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Implementação simulada do motor ASR para validação imediata da UI e testes no emulador/dispositivo.
 * Segue o mesmo contrato do RealAsrEngine (replay=0, recordingId, STATUS separado de CONTEÚDO).
 */
class FakeAsr(
    private var simulatedPhrase: String = "Quais são as etapas obrigatórias para desenergização de acordo com a NR-10?",
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default)
) : AsrEngine {

    private val _events = MutableSharedFlow<AsrEvent>(replay = 0, extraBufferCapacity = 8)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()

    private val recordingSeq = AtomicLong(0L)
    private var listeningJob: Job? = null
    private var isListening = false
    private var currentRecordingId = 0L

    fun setSimulatedPhrase(phrase: String) {
        simulatedPhrase = phrase
    }

    override fun startListening(): Long {
        if (isListening) return currentRecordingId
        isListening = true
        currentRecordingId = recordingSeq.incrementAndGet()
        val recId = currentRecordingId

        listeningJob = scope.launch {
            _events.emit(AsrEvent.Status(AsrStatus.RECORDING, recId))
            val words = simulatedPhrase.split(" ")
            val partialAccumulator = StringBuilder()

            for (word in words) {
                if (!isListening) break
                if (partialAccumulator.isNotEmpty()) partialAccumulator.append(" ")
                partialAccumulator.append(word)
                _events.emit(AsrEvent.Partial(partialAccumulator.toString(), recId))
                delay(120) // Simulação realista de cadência de fala
            }
        }
        return recId
    }

    override fun stopListening() {
        if (!isListening) return
        isListening = false
        listeningJob?.cancel()
        listeningJob = null
        val recId = currentRecordingId

        scope.launch {
            _events.emit(AsrEvent.Status(AsrStatus.TRANSCRIBING, recId))
            _events.emit(AsrEvent.Final(simulatedPhrase, recId))
        }
    }
}
