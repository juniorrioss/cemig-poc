package br.org.ceia.cemigpoc.domain.engine

import kotlinx.coroutines.flow.Flow

/**
 * Estágio operacional da captura de áudio (STATUS, não conteúdo transcrito).
 * Vai para o indicador de etapa da UI — NUNCA para o campo de texto da pergunta.
 */
enum class AsrStatus { RECORDING, TRANSCRIBING }

/**
 * Eventos emitidos pelo motor de reconhecimento de fala (ASR).
 *
 * Todo evento carrega o [recordingId] da gravação que o originou. O consumidor deve
 * descartar eventos de gravações que não sejam a corrente (evita reprocessar uma
 * transcrição antiga — causa raiz do bug do S24+).
 *
 * Contrato de separação STATUS × CONTEÚDO:
 * - [Status]  -> apenas estágio (indicador da UI), sem texto transcrito.
 * - [Partial] -> CONTEÚDO parcial transcrito (preview ao vivo), quando o motor suportar.
 * - [Final]   -> CONTEÚDO final transcrito, aguardando confirmação do operário.
 */
sealed interface AsrEvent {
    /** Identificador monotônico da gravação que originou o evento. */
    val recordingId: Long

    /**
     * Mudança de estágio da captura (gravando / transcrevendo). Apenas STATUS.
     */
    data class Status(val stage: AsrStatus, override val recordingId: Long) : AsrEvent

    /**
     * Transcrição parcial (CONTEÚDO) gerada em tempo real enquanto o operário fala.
     */
    data class Partial(val text: String, override val recordingId: Long) : AsrEvent

    /**
     * Transcrição final (CONTEÚDO) após o término da gravação. NÃO é enviada
     * automaticamente ao pipeline: o operário revisa/edita e confirma.
     */
    data class Final(val text: String, override val recordingId: Long) : AsrEvent

    /**
     * Erro ocorrido durante a captura ou transcrição de áudio.
     */
    data class Error(
        val message: String,
        val cause: Throwable? = null,
        override val recordingId: Long
    ) : AsrEvent
}

/**
 * Interface do motor de transcrição de voz para texto (ASR) em português.
 */
interface AsrEngine {
    /**
     * Inicia a escuta/gravação do microfone e retorna o [AsrEvent.recordingId]
     * atribuído a esta gravação. O consumidor guarda esse id para descartar eventos
     * de gravações anteriores. Retorna o id corrente se já estiver gravando.
     */
    fun startListening(): Long

    /**
     * Encerra a gravação e finaliza a transcrição.
     */
    fun stopListening()

    /**
     * Fluxo contínuo de eventos do ASR (parciais, finais e erros).
     */
    val events: Flow<AsrEvent>
}
