package br.org.ceia.cemigpoc.data.fake

import br.org.ceia.cemigpoc.domain.engine.AsrEngine
import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch

/**
 * Implementação simulada do motor ASR para validação imediata da UI e testes no emulador/dispositivo.
 */
class FakeAsr(
    private var simulatedPhrase: String = "Quais são as etapas obrigatórias para desenergização de acordo com a NR-10?",
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default)
) : AsrEngine {

    private val _events = MutableSharedFlow<AsrEvent>(replay = 1)
    override val events: Flow<AsrEvent> = _events.asSharedFlow()

    private var listeningJob: Job? = null
    private var isListening = false

    fun setSimulatedPhrase(phrase: String) {
        simulatedPhrase = phrase
    }

    override fun startListening() {
        if (isListening) return
        isListening = true

        listeningJob = scope.launch {
            val words = simulatedPhrase.split(" ")
            val partialAccumulator = StringBuilder()

            for (word in words) {
                if (!isListening) break
                if (partialAccumulator.isNotEmpty()) partialAccumulator.append(" ")
                partialAccumulator.append(word)
                _events.emit(AsrEvent.Partial(partialAccumulator.toString()))
                delay(120) // Simulação realista de cadência de fala
            }
        }
    }

    override fun stopListening() {
        if (!isListening) return
        isListening = false
        listeningJob?.cancel()
        listeningJob = null

        scope.launch {
            _events.emit(AsrEvent.Final(simulatedPhrase))
        }
    }
}
