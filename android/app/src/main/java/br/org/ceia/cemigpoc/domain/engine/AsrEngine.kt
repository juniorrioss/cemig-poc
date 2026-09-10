package br.org.ceia.cemigpoc.domain.engine

import kotlinx.coroutines.flow.Flow

/**
 * Eventos emitidos pelo motor de reconhecimento de fala (ASR).
 */
sealed interface AsrEvent {
    /**
     * Transcrição parcial gerada em tempo real enquanto o operário fala.
     */
    data class Partial(val text: String) : AsrEvent

    /**
     * Transcrição final confirmada após o término da gravação.
     */
    data class Final(val text: String) : AsrEvent

    /**
     * Erro ocorrido durante a captura ou transcrição de áudio.
     */
    data class Error(val message: String, val cause: Throwable? = null) : AsrEvent
}

/**
 * Interface do motor de transcrição de voz para texto (ASR) em português.
 */
interface AsrEngine {
    /**
     * Inicia a escuta/gravação do microfone.
     */
    fun startListening()

    /**
     * Encerra a gravação e finaliza a transcrição.
     */
    fun stopListening()

    /**
     * Fluxo contínuo de eventos do ASR (parciais, finais e erros).
     */
    val events: Flow<AsrEvent>
}
