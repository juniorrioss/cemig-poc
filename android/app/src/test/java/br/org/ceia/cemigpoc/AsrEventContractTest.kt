package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.domain.engine.AsrEvent
import br.org.ceia.cemigpoc.domain.engine.AsrStatus
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Reprodução automatizada do defeito do S24+ e prova da correção (task poc-asr-fix).
 *
 * Defeito relatado: "o assistente responde sem esperar a transcrição; depois a minha
 * transcrição aparece acima; ele responde referente à PRIMEIRA pergunta". Causa raiz:
 * canal de eventos do ASR com replay=1 — ao reassinar, o coletor reobtém o ÚLTIMO
 * AsrEvent.Final antigo e o pipeline dispara com a transcrição ANTERIOR.
 *
 * Estes testes provam, em JVM pura, (1) que o replay=1 re-entrega o Final velho e
 * (2) que o novo contrato (replay=0 + filtro por recordingId) elimina o reprocessamento.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AsrEventContractTest {

    /**
     * DEFEITO (antes): com replay=1, um novo coletor recebe imediatamente o último
     * Final de uma gravação anterior — exatamente o que fazia o app responder à
     * pergunta antiga.
     */
    @Test
    fun replay1_reentregaFinalAntigo_defeito() = runTest {
        val buggy = MutableSharedFlow<AsrEvent>(replay = 1)
        // Gravação 1 termina e emite seu Final.
        buggy.emit(AsrEvent.Final("primeira pergunta", recordingId = 1L))

        // Um novo coletor (reassinatura ao começar a 2ª gravação) recebe o Final VELHO.
        val reDelivered = buggy.asSharedFlow().first()
        assertTrue(reDelivered is AsrEvent.Final)
        assertEquals("primeira pergunta", (reDelivered as AsrEvent.Final).text)
        // ^ Este valor "primeira pergunta" reaparecendo é o bug reproduzido.
    }

    /**
     * CORREÇÃO (depois): com replay=0, o coletor NÃO reobtém eventos antigos, e o
     * consumidor descarta qualquer evento cujo recordingId não seja o da gravação
     * corrente. Nenhuma transcrição antiga é reprocessada.
     */
    @Test
    fun replay0_comFiltroDeId_naoReprocessa_correcao() = runTest {
        val fixed = MutableSharedFlow<AsrEvent>(replay = 0, extraBufferCapacity = 8)

        // Simula o consumidor do ViewModel: só aceita a gravação corrente.
        var activeRecordingId = 0L
        val processed = mutableListOf<String>()

        // UnconfinedTestDispatcher: o collector assina imediatamente, antes dos emits,
        // reproduzindo fielmente a ordem de eventos do app (replay=0 exige assinatura ativa).
        val job = launch(UnconfinedTestDispatcher(testScheduler)) {
            fixed.asSharedFlow().collect { event ->
                if (event.recordingId != activeRecordingId) return@collect // trava anti-obsoleto
                if (event is AsrEvent.Final) processed.add(event.text)
            }
        }

        // Gravação 1: corrente = 1.
        activeRecordingId = 1L
        fixed.emit(AsrEvent.Status(AsrStatus.RECORDING, 1L))
        fixed.emit(AsrEvent.Final("primeira pergunta", 1L))

        // Gravação 2 começa: corrente = 2. Um Final atrasado da gravação 1 NÃO deve contar.
        activeRecordingId = 2L
        fixed.emit(AsrEvent.Final("primeira pergunta", 1L)) // obsoleto -> descartado
        fixed.emit(AsrEvent.Final("segunda pergunta", 2L))  // corrente -> aceito

        job.cancel()

        // Cada Final foi processado uma única vez e sempre o da gravação corrente.
        assertEquals(listOf("primeira pergunta", "segunda pergunta"), processed)
    }

    /**
     * Status é STATUS, nunca CONTEÚDO: um AsrEvent.Status não carrega texto transcrito
     * e portanto não pode poluir o campo de pergunta do operário.
     */
    @Test
    fun status_naoCarregaTextoTranscrito() {
        val status: AsrEvent = AsrEvent.Status(AsrStatus.TRANSCRIBING, 5L)
        // Só Partial/Final expõem 'text'; Status expõe apenas o estágio.
        assertTrue(status is AsrEvent.Status)
        assertEquals(AsrStatus.TRANSCRIBING, (status as AsrEvent.Status).stage)
        val asFinal = status as? AsrEvent.Final
        assertNull(asFinal)
    }
}
